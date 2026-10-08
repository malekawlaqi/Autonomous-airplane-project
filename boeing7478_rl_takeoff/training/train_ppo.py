"""Train PPO as a COMPARISON algorithm (SAC is the primary controller).

Usage:
    python -m training.train_ppo --config configs/ppo.yaml
    python -m training.train_ppo --config configs/ppo.yaml --resume models/ppo/checkpoint_500000.zip
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO  # noqa: E402

from training import activation_from_name  # noqa: E402
from training.train_sac import parse_args, run_training  # noqa: E402


def build_ppo(cfg: dict[str, Any], env: Any, device: str, log_dir: Path) -> PPO:
    """Create a PPO model from the YAML configuration."""
    pol = cfg["policy"]
    policy_kwargs = {"net_arch": {"pi": list(pol["actor_layers"]), "vf": list(pol["critic_layers"])},
                     "activation_fn": activation_from_name(pol["activation"])}
    return PPO("MlpPolicy", env, learning_rate=float(cfg["learning_rate"]), n_steps=int(cfg["n_steps"]),
               batch_size=int(cfg["batch_size"]), n_epochs=int(cfg["n_epochs"]), gamma=float(cfg["gamma"]),
               gae_lambda=float(cfg["gae_lambda"]), clip_range=float(cfg["clip_range"]), ent_coef=float(cfg["ent_coef"]),
               vf_coef=float(cfg["vf_coef"]), max_grad_norm=float(cfg["max_grad_norm"]), policy_kwargs=policy_kwargs,
               tensorboard_log=str(log_dir), seed=int(cfg["seed"]), device=device, verbose=0)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m training.train_ppo``."""
    args = parse_args(__doc__ or "", "configs/ppo.yaml", argv)
    return run_training("PPO", PPO, build_ppo, args)


if __name__ == "__main__":
    raise SystemExit(main())
