"""Collect simulation-teacher demonstrations and use them to assist learning.

Usage:
    python -m teacher.demonstrations --episodes 200 --level 1 --out results/demos/teacher_demos.npz --noise 0.1

Assistance options (see training scripts): pre-fill the SAC replay buffer and/or behaviour-clone
the actor.  Whether this reaches the first success sooner than SAC from scratch is MEASURED by
running both and comparing ``steps_to_first_success`` in the milestone files - it is not assumed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PROJECT_ROOT, load_all_configs  # noqa: E402
from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv  # noqa: E402
from teacher.simulated_teacher import SimulatedTeacher  # noqa: E402


def collect_demonstrations(configs: dict[str, dict[str, Any]], n_episodes: int, level: int, out_path: str | Path,
                           action_noise_std: float = 0.0, seed_base: int = 2_000_000,
                           only_successful: bool = True) -> dict[str, Any]:
    """Run the teacher and save transitions to a ``.npz`` file.

    Args:
        configs: Loaded configuration dictionaries.
        n_episodes: Number of teacher episodes to run.
        level: Curriculum level of the episodes.
        out_path: Destination ``.npz`` path.
        action_noise_std: Gaussian noise added to executed actions (state diversity); the *noisy*
            action is stored because it is the one that produced the stored transition.
        seed_base: First episode seed (disjoint from training and evaluation seeds).
        only_successful: Keep only episodes that ended in SUCCESS.

    Returns:
        Statistics: episodes run/kept, transitions, success rate of the teacher.
    """
    env = Boeing7478TakeoffEnv(configs=configs)
    teacher = SimulatedTeacher(configs)
    rng = np.random.default_rng(seed_base)
    buffers: dict[str, list[np.ndarray]] = {k: [] for k in ("obs", "next_obs", "actions", "rewards", "dones", "truncated")}
    kept = 0
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + episode, options={"level": level})
        ep = {k: [] for k in buffers}
        while True:
            action = teacher.act(obs)
            if action_noise_std > 0.0:
                action = np.clip(action + rng.normal(0.0, action_noise_std, size=action.shape), -1.0, 1.0).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            for key, value in (("obs", obs), ("next_obs", next_obs), ("actions", action), ("rewards", reward),
                               ("dones", terminated), ("truncated", truncated)):
                ep[key].append(np.asarray(value, dtype=np.float32))
            obs = next_obs
            if terminated or truncated:
                break
        if (not only_successful) or info["episode_summary"]["success"]:
            kept += 1
            for key in buffers:
                buffers[key].extend(ep[key])
    env.close()
    if not buffers["obs"]:
        raise RuntimeError("No demonstration transitions collected (teacher never succeeded?)")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.stack(v) for k, v in buffers.items()})
    return {"episodes_run": n_episodes, "episodes_kept": kept, "transitions": len(buffers["obs"]),
            "teacher_success_rate": kept / n_episodes if only_successful else float("nan"), "path": str(out)}


def load_demonstrations(path: str | Path) -> dict[str, np.ndarray]:
    """Load a demonstration file written by :func:`collect_demonstrations`."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def add_to_replay_buffer(model: Any, demos: dict[str, np.ndarray]) -> int:
    """Insert demonstration transitions into an off-policy SB3 model's replay buffer.

    Returns:
        Number of transitions added.
    """
    buffer = model.replay_buffer
    n = len(demos["obs"])
    for i in range(n):
        truncated_only = bool(demos["truncated"][i]) and not bool(demos["dones"][i])
        done = bool(demos["dones"][i]) or bool(demos["truncated"][i])
        buffer.add(demos["obs"][i][None], demos["next_obs"][i][None], demos["actions"][i][None],
                   np.array([demos["rewards"][i]], dtype=np.float32), np.array([done]),
                   [{"TimeLimit.truncated": truncated_only}])
    return n


def behavior_clone_actor(model: Any, demos: dict[str, np.ndarray], epochs: int, batch_size: int = 256,
                         learning_rate: float = 1e-3) -> list[float]:
    """Pre-train a SAC actor by MSE regression onto the teacher's actions.

    Returns:
        Mean loss per epoch.
    """
    import torch

    actor = model.policy.actor
    device = model.device
    obs = torch.as_tensor(demos["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(demos["actions"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        permutation = torch.randperm(len(obs), device=device)
        total, batches = 0.0, 0
        for start in range(0, len(obs), batch_size):
            idx = permutation[start:start + batch_size]
            loss = torch.nn.functional.mse_loss(actor(obs[idx], deterministic=True), act[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, batches = total + float(loss.detach().cpu()), batches + 1
        losses.append(total / max(batches, 1))
    return losses


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.1, help="std of Gaussian action noise during collection")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "results" / "demos" / "teacher_demos.npz"))
    parser.add_argument("--all-episodes", action="store_true", help="keep failed teacher episodes too")
    parser.add_argument("--config-dir", type=str, default=None)
    args = parser.parse_args()
    stats = collect_demonstrations(load_all_configs(args.config_dir), args.episodes, args.level, args.out,
                                   args.noise, only_successful=not args.all_episodes)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
