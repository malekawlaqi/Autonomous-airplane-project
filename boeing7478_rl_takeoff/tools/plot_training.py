"""Plot training progress from the CSV files written during training (separate from training itself).

    python tools/plot_training.py --runs sac
    python tools/plot_training.py --runs sac sac_teacher     # compare runs (e.g. scratch vs teacher-assisted)

Reads results/<run>/training_episodes.csv, eval_history.csv and milestones.json; writes PNGs to
results/<run>/plots/ (and results/comparison_*.png when several runs are given).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from envs import PROJECT_ROOT  # noqa: E402


def load_run(run: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    """Load training episodes, evaluation history and milestones for a run (None when missing)."""
    base = PROJECT_ROOT / "results" / run
    read = lambda name: pd.read_csv(base / name) if (base / name).exists() else None  # noqa: E731
    milestones = json.loads((base / "milestones.json").read_text()) if (base / "milestones.json").exists() else None
    return read("training_episodes.csv"), read("eval_history.csv"), milestones


def plot_run(run: str, window: int) -> None:
    """Write per-run plots."""
    episodes, evals, milestones = load_run(run)
    if episodes is None:
        print(f"[{run}] no training_episodes.csv yet - nothing to plot")
        return
    out = PROJECT_ROOT / "results" / run / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    x = episodes["num_timesteps"]
    axes[0, 0].plot(x, episodes["success"].astype(float).rolling(window, min_periods=1).mean())
    axes[0, 0].set(title=f"Training success rate (rolling {window} episodes)", xlabel="RL transitions", ylim=(-0.02, 1.02))
    axes[0, 1].plot(x, episodes["total_reward"].rolling(window, min_periods=1).mean())
    axes[0, 1].set(title="Episode return (rolling mean)", xlabel="RL transitions")
    if evals is not None and not evals.empty:
        axes[1, 0].plot(evals["num_timesteps"], evals["success_rate"], marker="o")
        axes[1, 0].set(title="Deterministic evaluation success rate", xlabel="RL transitions", ylim=(-0.02, 1.02))
        axes[1, 1].step(evals["num_timesteps"], evals["level"], where="post")
        axes[1, 1].set(title="Curriculum level", xlabel="RL transitions")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    if milestones and milestones.get("steps_to_first_success") is not None:
        axes[0, 0].axvline(milestones["steps_to_first_success"], color="g", ls="--", label="first success")
        axes[0, 0].legend()
    fig.suptitle(f"Run '{run}' (units: RL environment transitions)")
    fig.tight_layout()
    fig.savefig(out / "training_overview.png", dpi=120)
    plt.close(fig)
    print(f"[{run}] wrote {out / 'training_overview.png'}")


def plot_comparison(runs: list[str], window: int) -> None:
    """Overlay training/evaluation success of several runs and print their milestones."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for run in runs:
        episodes, evals, milestones = load_run(run)
        if episodes is not None:
            axes[0].plot(episodes["num_timesteps"], episodes["success"].astype(float).rolling(window, min_periods=1).mean(), label=run)
        if evals is not None:
            axes[1].plot(evals["num_timesteps"], evals["success_rate"], marker="o", label=run)
        first = milestones.get("steps_to_first_success") if milestones else None
        print(f"{run:20s} steps_to_first_success = {first if first is not None else 'NOT YET MEASURED'}")
    axes[0].set(title="Training success (rolling)", xlabel="RL transitions")
    axes[1].set(title="Evaluation success", xlabel="RL transitions")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = PROJECT_ROOT / "results" / f"comparison_{'_'.join(runs)}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["sac"])
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    for run in args.runs:
        plot_run(run, args.window)
    if len(args.runs) > 1:
        plot_comparison(args.runs, args.window)


if __name__ == "__main__":
    main()
