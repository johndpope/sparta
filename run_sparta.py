#!/usr/bin/env python3
"""
SPARTA ALIGNMENT – Self-Play Arena for Reinforcement Training and Alignment.

Usage:
    python run_sparta.py
    python run_sparta.py --task alpaca --gpus 0,1,2 --iterations 8
"""

import argparse
from sparta import SpartaAlignment, SpartaConfig


def main():
    parser = argparse.ArgumentParser(description="SPARTA ALIGNMENT Training Pipeline")
    parser.add_argument("--task", type=str, default="alpaca",
                        help="Task: alpaca, gsm8k, truthfulqa_mc1,truthfulqa_mc2, culture_country, etc.")
    parser.add_argument("--gpus", type=str, default="0,1,2",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--iterations", type=int, default=8,
                        help="Number of training iterations")
    parser.add_argument("--output", type=str, default="sparta_output",
                        help="Output base directory")
    parser.add_argument("--score-type", type=str, default="static",
                        choices=["normal", "dynamic", "static"],
                        help="Rating system type")
    parser.add_argument("--fair", action="store_true", default=True,
                        help="Use fair model initialization")
    parser.add_argument("--no-fair", action="store_false", dest="fair",
                        help="Use unfair model initialization")
    parser.add_argument("--num-instructions", type=int, default=1000,
                        help="Instructions per iteration")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    config = SpartaConfig(
        task=args.task,
        gpu_ids=[int(g) for g in args.gpus.split(",")],
        num_iterations=args.iterations,
        output_base_dir=args.output,
        score_type=args.score_type,
        fair_init=args.fair,
        prompts_per_iteration=args.num_instructions,
        random_seed=args.seed,
    )

    sparta = SpartaAlignment(config)
    sparta.run()


if __name__ == "__main__":
    main()
