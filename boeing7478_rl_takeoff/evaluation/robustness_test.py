"""Robustness test on unseen condition combinations, with a failure table.

Usage:
    python -m evaluation.robustness_test --model models/sac/best_model.zip --episodes 100 --levels 3 4
    python -m evaluation.robustness_test --model models/sac/final_model.zip --episodes 1000 --levels 4   # final research run
    python -m evaluation.robustness_test --policy teacher --episodes 100 --levels 3 4

Seeds start at 5,000,000 by default (training seeds are random 31-bit draws; evaluation seeds start
at 1,000,000), so the sampled runway/mass/CG/weather/wind/surface/temperature/pressure combinations
are unseen by the policy during training/validation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PROJECT_ROOT, load_all_configs  # noqa: E402
from evaluation.evaluate import aggregate, build_policy, evaluate_policy  # noqa: E402
from evaluation.failure_analysis import write_reports  # noqa: E402


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--policy", type=str, default="teacher", choices=["teacher", "full_throttle", "random"])
    parser.add_argument("--episodes", type=int, default=100, help="episodes per level (>=100 initially, 1000+ for final)")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--seed-base", type=int, default=5_000_000)
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "results" / "robustness"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--config-dir", type=str, default=None)
    args = parser.parse_args()

    configs = load_all_configs(args.config_dir)
    policy = build_policy(args.policy, args.model, configs, args.device)
    label = Path(args.model).stem if args.model else args.policy
    frames = []
    for level in args.levels:
        df = evaluate_policy(policy, configs, args.episodes, level, args.seed_base + 100_000 * level)
        df.insert(0, "policy", label)
        frames.append(df)
        print(f"\n=== {label} | level {level} | {len(df)} episodes | success rate {aggregate(df)['success_rate']:.3f} ===")
    all_df = pd.concat(frames, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_csv = out_dir / f"{label}_episodes.csv"
    all_df.to_csv(episodes_csv, index=False)
    failures, breakdown = write_reports(all_df, out_dir, label)
    print("\nFAILURE TABLE")
    print(failures.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nHEADLINE METRICS (successful episodes only where applicable)")
    for key, value in aggregate(all_df).items():
        print(f"  {key:32s} {value:.4f}")
    print(f"\nWrote {episodes_csv} and failure/breakdown CSVs in {out_dir}")


if __name__ == "__main__":
    main()
