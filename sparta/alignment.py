"""
SpartaAlignment – main orchestrator for the SPARTA ALIGNMENT training loop.

Implements the full pipeline:
    1. Load instructions from dataset
    2. Combat phase (model matchmaking + response generation)
    3. Judge phase (multi-judge evaluation)
    4. Rating update (Elo-style rating system)
    5. Preference pair generation
    6. DPO fine-tuning
    7. Repeat
"""

import os
import sys
import json
import random
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional

import torch

from sparta.config import SpartaConfig
from sparta.utils import save_json, load_json, cleanup_resources
from sparta.competition.matcher import (
    Competition,
    process_model_group,
    save_judged_pairs,
    save_preference_pairs_to_json,
    save_rating_history,
    save_model_info,
    save_judge_pairs,
    load_model_info,
    filter_tie,
    judge_with_gpu,
)
from sparta.competition.data import load_instructions
from sparta.judge.judge import run_judges, calculate_judge_averages
from sparta.rating.system import (
    RatingSystem,
    RatingSystemDynamicWeighted,
    RatingSystemStaticWeighted,
)
from sparta.rating.tracker import RatingTracker, StopCriteria
from sparta.model_init.init import ModelInit, ModelInitFair
from sparta.training.dpo import train_dpo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpartaAlignment:
    """
    Generic SPARTA ALIGNMENT implementation.

    Orchestrates the full self-play arena training loop:
    combat → judge → rating update → preference extraction → DPO fine-tuning.

    Supports any set of models (Qwen, Gemma, DeepSeek, etc.) and is
    compatible with Repr-Align, ColaDLM, PersistentSparseAdam, etc.
    """

    def __init__(self, config: Optional[SpartaConfig] = None):
        self.config = config or SpartaConfig()
        self._validate_config()

        # ── Derived paths ──────────────────────────────────────────────
        self.output_dir = Path(self.config.output_base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Model mappings ─────────────────────────────────────────────
        self.model_base_mapping = dict(
            zip(self.config.model_names, self.config.base_models)
        )

        # ── Random seed ────────────────────────────────────────────────
        random.seed(self.config.random_seed)

        # ── GPU setup ──────────────────────────────────────────────────
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(g) for g in self.config.gpu_ids
        )

        # ── Early-stopping tracker ─────────────────────────────────────
        stop_criteria = StopCriteria(
            min_improvement=self.config.stop_min_improvement,
            window_size=self.config.stop_window_size,
            patience=self.config.stop_patience,
        )
        self.rating_tracker = RatingTracker(
            save_dir=str(self.output_dir / "ratings"),
            stop_criteria=stop_criteria,
        )

        logger.info(
            f"SpartaAlignment initialized: {len(self.config.model_names)} models, "
            f"{self.config.num_iterations} iterations, task={self.config.task}"
        )

    # ── Public API ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full SPARTA ALIGNMENT training loop."""
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        for iteration in range(self.config.num_iterations):
            logger.info(
                f"\n{'='*60}\n"
                f"SPARTA Iteration {iteration}/{self.config.num_iterations - 1}\n"
                f"{'='*60}"
            )

            self._run_iteration(iteration)

    # ── Iteration lifecycle ────────────────────────────────────────────

    def _run_iteration(self, iteration: int) -> None:
        """Run a single SPARTA iteration (combat → judge → rate → DPO)."""
        iteration_dir = self.output_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build model configs & info for this iteration
        model_info = self._init_model_info(iteration)
        model_configs = self._build_model_configs(iteration, model_info)

        # 2. Load instructions
        instructions = self._load_instructions()
        if not instructions:
            logger.error("No instructions loaded – skipping iteration")
            return

        # 3. Combat phase
        raw_pairs = self._run_combat_phase(instructions, model_configs, model_info)
        logger.info(f"Generated {len(raw_pairs)} combat pairs")

        # 4. Judge phase
        judged_pairs = self._run_judge_phase(raw_pairs, model_configs, iteration)
        logger.info(f"Judged {len(judged_pairs)} pairs")

        # 5. Rating update
        rating_system = self._create_rating_system(model_info, iteration)
        model_info, rating_history = self._update_ratings(
            judged_pairs, rating_system, model_info
        )

        # 6. Generate preference pairs
        preference_pairs = self._generate_preferences(raw_pairs, rating_system)

        # 7. Save state
        self._save_iteration_state(
            iteration, model_info, judged_pairs, preference_pairs, rating_history
        )

        # 8. Early stopping check
        should_stop = self.rating_tracker.update(
            {name: info for name, info in model_info.items()}
        )
        if should_stop:
            logger.info("Early stopping triggered!")
            return

        # 9. DPO fine-tuning
        self._run_dpo_phase(preference_pairs, model_configs, iteration)

        logger.info(f"Completed iteration {iteration}")

    # ── Phase: Model Init ──────────────────────────────────────────────

    def _init_model_info(self, iteration: int) -> Dict:
        """Load model info from previous iteration or initialize fresh."""
        if iteration > 0:
            model_info = load_model_info(str(self.output_dir), iteration - 1)
            if model_info is not None:
                # Update paths to point to previous iteration's output
                for model_name in self.config.model_names:
                    if model_name in model_info:
                        model_path = str(
                            self.output_dir
                            / f"iteration_{iteration - 1}"
                            / model_name
                        )
                        model_info[model_name]["path"] = model_path
                return model_info

        # First iteration – register all models
        model_info = {}
        for model_name in self.config.model_names:
            model_info[model_name] = self._register_model(model_name)
        return model_info

    def _register_model(self, model_name: str) -> Dict:
        """Initialize a model's score, deviation, and path."""
        model_base = self.model_base_mapping[model_name]
        init_model_dir = os.path.join(self.config.init_model_dir, model_base)

        if self.config.fair_init:
            model_init = ModelInitFair(
                save_dir=init_model_dir,
                task=self.config.task,
                base_model=model_base,
            )
        else:
            model_init = ModelInit(
                save_dir=init_model_dir,
                task=self.config.task,
                base_model=model_base,
            )

        init_path = os.path.join(self.config.init_model_dir, model_base)
        if os.path.exists(os.path.join(self.config.init_model_dir, model_name)):
            return model_init.init_model(model_name)
        else:
            model_init.init_model_local(model_name)
            return model_init.init_model(model_name)

    def _build_model_configs(self, iteration: int, model_info: Dict) -> List[Dict]:
        """Build model config list for competition and DPO."""
        model_configs = []
        for model_name in self.config.model_names:
            model_type = self.model_base_mapping[model_name]

            if iteration == 0:
                model_path = os.path.join(
                    self.config.init_model_dir, model_type, model_name
                )
            else:
                model_path = str(
                    self.output_dir / f"iteration_{iteration - 1}" / model_name
                )

            output_path = str(
                self.output_dir / f"iteration_{iteration}" / model_name
            )

            model_configs.append(
                {
                    "name": model_name,
                    "path": model_path,
                    "output_path": output_path,
                    "learning_rate": 1e-5,
                    "model_type": model_type,
                }
            )
        return model_configs

    # ── Phase: Instructions ────────────────────────────────────────────

    def _load_instructions(self) -> List[str]:
        """Load instructions from dataset based on config.task."""
        return load_instructions(
            task=self.config.task,
            data_dir=self.config.data_dir,
            num_instructions=self.config.prompts_per_iteration,
        )

    # ── Phase: Combat ──────────────────────────────────────────────────

    def _run_combat_phase(
        self,
        instructions: List[str],
        model_configs: List[Dict],
        model_info: Dict,
    ) -> List[Dict]:
        """Generate combat pairs using match-making."""
        competition = Competition(
            model_configs,
            model_info,
            num_opponents=self.config.top_k_match,
            random_match_prob=self.config.match_random_prob,
            is_random_select=self.config.random_select,
        )
        return competition.run(instructions)

    # ── Phase: Judge ───────────────────────────────────────────────────

    def _run_judge_phase(
        self,
        raw_pairs: List[Dict],
        model_configs: List[Dict],
        iteration: int,
    ) -> List[Dict]:
        """Run all judges on combat pairs and compute averages."""
        judged_pairs = run_judges(
            model_configs=model_configs,
            pairs=raw_pairs,
            gpu_ids=self.config.gpu_ids,
            batch_size=self.config.batch_size,
            round_num=1,
            base_dir=str(self.output_dir),
            base_model=self.config.base_models[0] if self.config.base_models else "gemma",
        )

        judged_pairs = calculate_judge_averages(judged_pairs)

        cleanup_resources()
        return judged_pairs

    # ── Phase: Rating ──────────────────────────────────────────────────

    def _create_rating_system(self, model_info: Dict, iteration: int) -> RatingSystem:
        """Instantiate the rating system based on config.score_type."""
        # Load delta history if available
        delta_history = self._load_delta_history()

        common_kwargs = dict(
            model_scores=model_info,
            initial_K=self.config.initial_k,
            min_K=self.config.min_k,
            delta_history=delta_history,
            window_size=self.config.window_size,
            min_deviation=self.config.min_deviation,
            epsilon=self.config.epsilon,
            decay_rate=self.config.decay_rate,
            decay_steps=self.config.decay_steps,
            scaling_factor=self.config.scaling_factor,
            freeze_ratings=self.config.freeze_ratings,
        )

        if self.config.score_type == "dynamic":
            return RatingSystemDynamicWeighted(
                **common_kwargs,
                base_dir=str(self.output_dir),
                current_iteration=iteration,
            )
        elif self.config.score_type == "static":
            return RatingSystemStaticWeighted(
                **common_kwargs,
                base_dir=str(self.output_dir),
                current_iteration=iteration,
            )
        else:
            return RatingSystem(**common_kwargs)

    def _load_delta_history(self) -> Dict[str, List[float]]:
        """Load or initialize delta history for rating system."""
        history_path = self.output_dir / "rating_deltas.json"
        if history_path.exists():
            try:
                delta_history = load_json(history_path)
                for model in self.config.model_names:
                    if model in delta_history:
                        delta_history[model] = delta_history[model][-10:]
                    else:
                        delta_history[model] = []
                return delta_history
            except Exception:
                pass
        return {model: [] for model in self.config.model_names}

    def _update_ratings(
        self,
        judged_pairs: List[Dict],
        rating_system: RatingSystem,
        model_info: Dict,
    ) -> tuple:
        """Update ratings from judged pairs and record history."""
        rating_history = []

        for i, pair in enumerate(judged_pairs):
            if isinstance(pair, (dict, list)):
                rating_system.update_ratings_from_judges(pair)

            current_ratings = rating_system.get_all_ratings()
            rating_history.append(
                {
                    "pair_index": i,
                    "pair": pair,
                    "ratings": {
                        model: {
                            "score": info["score"],
                            "deviation": info["deviation"],
                        }
                        for model, info in current_ratings.items()
                    },
                }
            )
            model_info = current_ratings

        logger.info("Final Ratings:")
        for model, rating in model_info.items():
            logger.info(
                f"  {model}: score={rating['score']:.2f}, "
                f"deviation={rating['deviation']:.4f}"
            )

        return model_info, rating_history

    # ── Phase: Preferences ─────────────────────────────────────────────

    def _generate_preferences(
        self,
        raw_pairs: List[Dict],
        rating_system: RatingSystem,
    ) -> List[Dict]:
        """Convert raw pairs → DPO preference pairs, filtering ties."""
        preference_pairs = []
        for pair in raw_pairs:
            preference = rating_system.select_preference_response(pair)
            if preference:
                preference_pairs.append(preference)

        old_len = len(preference_pairs)
        preference_pairs = filter_tie(preference_pairs)

        if old_len > len(preference_pairs) * 0.5:
            logger.warning(
                f"Large number of ties filtered: "
                f"{old_len} → {len(preference_pairs)}"
            )

        logger.info(f"Generated {len(preference_pairs)} preference pairs")
        return preference_pairs

    # ── Phase: DPO ─────────────────────────────────────────────────────

    def _run_dpo_phase(
        self,
        preference_pairs: List[Dict],
        model_configs: List[Dict],
        iteration: int,
    ) -> None:
        """Run DPO on all models using preference pairs, parallelized by GPU."""
        if not preference_pairs:
            logger.warning("No preference pairs – skipping DPO phase")
            return

        cleanup_resources()

        # Split models into groups of 2, one GPU per group
        models_per_group = 2
        num_groups = (
            len(model_configs) + models_per_group - 1
        ) // models_per_group

        processes = []
        for group_idx in range(num_groups):
            start = group_idx * models_per_group
            end = min(start + models_per_group, len(model_configs))
            group_configs = model_configs[start:end]
            gpu_id = self.config.gpu_ids[
                group_idx % len(self.config.gpu_ids)
            ]

            p = mp.Process(
                target=_dpo_worker,
                args=(group_configs, iteration, gpu_id, str(self.output_dir)),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        cleanup_resources()

    # ── Save state ─────────────────────────────────────────────────────

    def _save_iteration_state(
        self,
        iteration: int,
        model_info: Dict,
        judged_pairs: List[Dict],
        preference_pairs: List[Dict],
        rating_history: List[Dict],
    ) -> None:
        """Persist all state for this iteration."""
        base_dir = str(self.output_dir)

        save_model_info(model_info, base_dir, iteration)
        save_judge_pairs(judged_pairs, base_dir, iteration)
        save_judged_pairs(judged_pairs, base_dir, iteration)

        save_preference_pairs_to_json(
            preference_pairs,
            os.path.join(base_dir, f"iteration_{iteration}", "dataset"),
            "preference_pairs.json",
        )
        save_rating_history(rating_history, base_dir, iteration)

    # ── Validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate configuration consistency."""
        cfg = self.config
        if len(cfg.model_names) != len(cfg.base_models):
            if len(cfg.base_models) == 1:
                cfg.base_models = cfg.base_models * len(cfg.model_names)
            else:
                raise ValueError(
                    f"base_models length ({len(cfg.base_models)}) must match "
                    f"model_names length ({len(cfg.model_names)})"
                )
        if cfg.score_type not in ("normal", "dynamic", "static"):
            raise ValueError(
                f"Invalid score_type: {cfg.score_type}. "
                "Must be 'normal', 'dynamic', or 'static'."
            )


def _dpo_worker(
    group_configs: List[Dict],
    iteration: int,
    gpu_id: int,
    base_dir: str,
) -> None:
    """Worker function for parallel DPO training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for cfg in group_configs:
        model_name = cfg["name"]
        model_type = cfg["model_type"]
        model_path = cfg["path"]
        output_path = cfg["output_path"]

        dataset_path = os.path.join(
            base_dir, f"iteration_{iteration}", "dataset", "preference_pairs.json"
        )

        os.makedirs(output_path, exist_ok=True)

        logger.info(
            f"DPO training {model_name} on GPU {gpu_id} "
            f"(iter {iteration}, base={model_type})"
        )

        success = train_dpo(
            model_name=model_name,
            dataset_path=dataset_path,
            model_path=model_path,
            output_path=output_path,
            gpu_id=gpu_id,
            iteration=iteration,
            base_model=model_type,
        )

        if success:
            logger.info(f"Successfully trained {model_name}")
        else:
            logger.error(f"Failed to train {model_name}")
