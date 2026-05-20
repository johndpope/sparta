from dataclasses import dataclass, field
from typing import List


@dataclass
class SpartaConfig:
    """Configuration for the SPARTA ALIGNMENT training pipeline."""

    # ── Models ──────────────────────────────────────────────────────────
    model_names: List[str] = field(default_factory=lambda: [
        "code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1",
    ])
    base_models: List[str] = field(default_factory=lambda: [
        "gemma", "gemma", "gemma", "qwen", "qwen", "qwen",
    ])

    # ── Paths ───────────────────────────────────────────────────────────
    output_base_dir: str = "sparta_output"
    init_model_dir: str = "init_model"
    data_dir: str = "data"
    base_model_dir: str = "base_model"

    # ── Pipeline ────────────────────────────────────────────────────────
    num_iterations: int = 8
    task: str = "alpaca"
    fair_init: bool = True

    # ── Competition ─────────────────────────────────────────────────────
    prompts_per_iteration: int = 1000
    match_random_prob: float = 0.6       # α – probability of random opponent selection
    top_k_match: int = 5                 # number of potential opponents
    random_select: bool = True
    batch_size: int = 24
    max_new_tokens: int = 512

    # ── Rating system ───────────────────────────────────────────────────
    score_type: str = "static"           # "normal", "dynamic", "static"
    initial_k: float = 10.0
    min_k: float = 5.0
    window_size: int = 10
    min_deviation: float = 0.1
    epsilon: float = 0.01
    decay_rate: float = 0.9
    decay_steps: int = 10
    scaling_factor: float = 20.0
    freeze_ratings: bool = False

    # ── Judge ───────────────────────────────────────────────────────────
    judge_names: List[str] = field(default_factory=lambda: [
        "sharegpt", "wizardlm", "science", "open_orca",
    ])
    judge_rounds: int = 3

    # ── DPO training ────────────────────────────────────────────────────
    dpo_epochs: int = 1
    dpo_lr: float = 1e-7
    dpo_batch_size: int = 8
    dpo_beta: float = 0.1
    dpo_max_length: int = 512
    dpo_max_prompt_length: int = 256

    # ── Early stopping ──────────────────────────────────────────────────
    stop_min_improvement: float = 0.1
    stop_window_size: int = 5
    stop_patience: int = 3

    # ── Hardware ────────────────────────────────────────────────────────
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2])
    random_seed: int = 42
