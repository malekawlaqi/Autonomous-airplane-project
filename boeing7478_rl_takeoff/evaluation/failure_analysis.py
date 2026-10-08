"""Failure analysis: failure-reason table and success rate broken down by condition.

Usage:
    python -m evaluation.failure_analysis --csv results/robustness/xyz_episodes.csv --out results/robustness
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REASON_ORDER: tuple[str, ...] = (
    "SUCCESS", "RUNWAY_EXCURSION", "FAILED_ROTATION", "INSUFFICIENT_RUNWAY", "LOSS_OF_CONTROL",
    "UNSTABLE_AFTER_LIFTOFF", "CRASH", "SIMULATOR_INVALID_STATE", "ENVELOPE_VIOLATION", "FAILED_TO_ACCELERATE",
    "TIMEOUT",
)

# column -> bin edges for the condition breakdown
CONDITION_BINS: dict[str, list[float]] = {
    "mass_lbs": [0, 600_000, 700_000, 800_000, 900_000, 1_100_000],
    "crosswind_kt": [-100, -20, -10, 10, 20, 100],
    "headwind_kt": [-100, 0, 10, 20, 100],
    "runway_length_m": [0, 3200, 3800, 4400, 10_000],
    "runway_elevation_ft": [-1000, 1000, 3000, 5000, 20_000],
    "static_friction_factor": [0, 0.5, 0.7, 0.9, 1.01],
    "delta_isa_c": [-100, -10, 10, 20, 100],
    "cg_x_in": [0, 1300, 1320, 1340, 1400, 5000],
    "turbulence_severity": [-1, 0.01, 1, 2, 10],
}


def failure_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count outcomes per reason (all known reasons are listed, including zero-count rows)."""
    counts = df["reason"].value_counts()
    reasons = list(REASON_ORDER) + [r for r in counts.index if r not in REASON_ORDER]
    table = pd.DataFrame({"reason": reasons, "count": [int(counts.get(r, 0)) for r in reasons]})
    table["percent"] = 100.0 * table["count"] / max(len(df), 1)
    return table


def condition_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Success rate and episode count for each bin of each randomised condition."""
    rows = []
    success = df["success"].astype(bool)
    for column, edges in CONDITION_BINS.items():
        if column not in df.columns:
            continue
        bins = pd.cut(df[column], bins=edges, include_lowest=True)
        for interval, group in df.groupby(bins, observed=True):
            rows.append({"condition": column, "bin": str(interval), "episodes": len(group),
                         "success_rate": float(success.loc[group.index].mean()) if len(group) else np.nan})
    return pd.DataFrame(rows)


def write_reports(df: pd.DataFrame, out_dir: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write failure and breakdown CSVs and return both tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = failure_table(df)
    breakdown = condition_breakdown(df)
    failures.to_csv(out_dir / f"{prefix}_failure_table.csv", index=False)
    breakdown.to_csv(out_dir / f"{prefix}_condition_breakdown.csv", index=False)
    return failures, breakdown


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, required=True, help="per-episode evaluation CSV")
    parser.add_argument("--out", type=str, default=None, help="output directory (default: CSV's directory)")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    failures, breakdown = write_reports(df, Path(args.out) if args.out else csv_path.parent, csv_path.stem)
    print(failures.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print()
    print(breakdown.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
