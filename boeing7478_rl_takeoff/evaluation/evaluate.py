"""Deterministic policy evaluation on a separate environment, writing CSV results.

Usage:
    python -m evaluation.evaluate --model models/sac/best_model.zip --episodes 100 --level 3 --out results/eval.csv
    python -m evaluation.evaluate --policy teacher --episodes 50 --level 1 --out results/teacher_eval.csv
    python -m evaluation.evaluate --policy full_throttle --episodes 100 --level 3   # open-loop baseline

Policies: a trained SB3 zip (SAC or PPO), ``teacher`` (conventional simulated controller),
``full_throttle`` (open-loop baseline: throttle 1, neutral surfaces) or ``random``.
Nothing here fabricates results: all numbers come from executed simulation episodes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PROJECT_ROOT, load_all_configs  # noqa: E402
from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv  # noqa: E402

Policy = Callable[[np.ndarray], np.ndarray]
SUMMARY_FAIL_COLUMNS: tuple[str, ...] = ("reason",)


def full_throttle_policy(_obs: np.ndarray) -> np.ndarray:
    """Open-loop baseline: full throttle, neutral elevator/aileron/rudder."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def make_random_policy(seed: int = 0) -> Policy:
    """Return a policy sampling uniform random actions."""
    rng = np.random.default_rng(seed)
    return lambda _obs: rng.uniform(-1.0, 1.0, size=4).astype(np.float32)


def sb3_policy(model: Any) -> Policy:
    """Wrap a Stable-Baselines3 model as a deterministic policy."""
    return lambda obs: model.predict(obs, deterministic=True)[0]


def load_sb3_model(path: str | Path, device: str = "auto") -> Any:
    """Load a SAC or PPO model, inferring the class from the file's saved metadata."""
    from stable_baselines3 import PPO, SAC

    errors: list[str] = []
    for cls in (SAC, PPO):
        try:
            return cls.load(str(path), device=device)
        except Exception as exc:  # noqa: BLE001 - deliberately re-raised below with context
            errors.append(f"{cls.__name__}: {exc!r}")
    raise RuntimeError(f"Could not load {path} as SAC or PPO:\n  " + "\n  ".join(errors))


def run_episode(env: Boeing7478TakeoffEnv, policy: Policy, eval_seed: int, level: int) -> dict[str, Any]:
    """Run one deterministic episode and return its summary dictionary."""
    obs, _ = env.reset(seed=eval_seed, options={"level": level})
    while True:
        obs, _reward, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            summary = dict(info["episode_summary"])
            summary["eval_seed"] = eval_seed
            return summary


def evaluate_policy(policy: Policy, configs: dict[str, dict[str, Any]], n_episodes: int, level: int,
                    seed_base: int, env: Boeing7478TakeoffEnv | None = None) -> pd.DataFrame:
    """Evaluate ``policy`` for ``n_episodes`` on fixed seeds ``seed_base + i``.

    Args:
        policy: Maps observation to action.
        configs: Loaded configuration dictionaries.
        n_episodes: Number of evaluation episodes.
        level: Curriculum level of the conditions to sample.
        seed_base: First evaluation seed (keep disjoint from training).
        env: Optional existing environment to reuse.

    Returns:
        One row per episode with the metrics listed in PHASE 14 plus conditions.
    """
    own_env = env is None
    env = env or Boeing7478TakeoffEnv(configs=configs)
    try:
        rows = [run_episode(env, policy, seed_base + i, level) for i in range(n_episodes)]
    finally:
        if own_env:
            env.close()
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> dict[str, float]:
    """Aggregate an evaluation DataFrame into headline metrics (NaN when not measurable)."""
    ok = df[df["success"].astype(bool)]
    mean = lambda col, frame=ok: float(frame[col].mean()) if len(frame) else float("nan")  # noqa: E731
    return {
        "episodes": float(len(df)), "success_rate": float(df["success"].astype(bool).mean()),
        "mean_takeoff_time_s": mean("takeoff_time_s"), "mean_liftoff_time_s": mean("liftoff_time_s"),
        "mean_runway_distance_m": mean("runway_distance_used_m"),
        "mean_centerline_rms_m": mean("centerline_rms_m"), "mean_centerline_max_m": mean("centerline_max_m"),
        "mean_max_roll_deg": mean("max_roll_deg"), "mean_max_pitch_deg": mean("max_pitch_deg"),
        "mean_max_alpha_deg": mean("max_alpha_deg"), "mean_max_heading_error_deg": mean("max_heading_error_deg"),
        "mean_control_roughness": mean("control_roughness"),
        "mean_cleanliness": mean("cleanliness"), "mean_rms_lateral_accel_mps2": mean("rms_lateral_accel_mps2"),
        "mean_stabilization_fraction": mean("stabilization_fraction_after_100ft", df),
        "mean_return": float(df["total_reward"].mean()),
    }


def build_policy(name: str, model_path: str | None, configs: dict[str, dict[str, Any]], device: str) -> Policy:
    """Create a policy by name (``teacher``, ``full_throttle``, ``random``) or from a model zip."""
    if model_path:
        return sb3_policy(load_sb3_model(model_path, device))
    if name == "teacher":
        from teacher.simulated_teacher import SimulatedTeacher
        return SimulatedTeacher(configs).act
    if name == "full_throttle":
        return full_throttle_policy
    if name == "random":
        return make_random_policy(0)
    raise ValueError(f"Unknown policy '{name}'")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None, help="path to an SB3 .zip model")
    parser.add_argument("--policy", type=str, default="teacher", choices=["teacher", "full_throttle", "random"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1_000_000)
    parser.add_argument("--out", type=str, default=None, help="CSV path for per-episode results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--config-dir", type=str, default=None)
    args = parser.parse_args()

    configs = load_all_configs(args.config_dir)
    policy = build_policy(args.policy, args.model, configs, args.device)
    df = evaluate_policy(policy, configs, args.episodes, args.level, args.seed_base)
    label = args.model or args.policy
    df.insert(0, "policy", label)
    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"eval_{Path(label).stem}_L{args.level}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Policy: {label} | level {args.level} | episodes {len(df)} | wrote {out}")
    for key, value in aggregate(df).items():
        print(f"  {key:32s} {value:.4f}")
    print("Failure reasons:")
    print(df["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
