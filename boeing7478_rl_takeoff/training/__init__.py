"""Training utilities shared by the SAC and PPO entry points."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from envs import PROJECT_ROOT


@dataclass(frozen=True)
class RunPaths:
    """Directories used by one training run."""

    models: Path
    logs: Path
    results: Path


def prepare_run_dirs(cfg: dict[str, Any]) -> RunPaths:
    """Create (and return) ``models/<run>``, ``logs`` and ``results/<run>`` directories."""
    run = str(cfg["run_name"])
    paths = cfg["paths"]
    models = PROJECT_ROOT / paths["models_dir"] / run
    logs = PROJECT_ROOT / paths["logs_dir"]
    results = PROJECT_ROOT / paths["results_dir"] / run
    for directory in (models, logs, results):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(models=models, logs=logs, results=results)


def select_device(requested: str = "auto") -> str:
    """Pick the torch device and print PyTorch/CUDA diagnostics.

    Args:
        requested: ``auto`` (CUDA when available), ``cuda`` or ``cpu``.

    Returns:
        The device string to give to Stable-Baselines3.
    """
    import torch

    cuda = torch.cuda.is_available()
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA available    : {cuda}")
    print(f"GPU name          : {torch.cuda.get_device_name(0) if cuda else 'n/a'}")
    if requested == "auto":
        device = "cuda" if cuda else "cpu"
    elif requested.startswith("cuda") and not cuda:
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
    else:
        device = requested
    print(f"Device selected   : {device}")
    return device


def activation_from_name(name: str) -> type:
    """Map an activation name from YAML to a torch.nn class."""
    from torch import nn

    table = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    if name.lower() not in table:
        raise ValueError(f"Unknown activation '{name}'; choose from {sorted(table)}")
    return table[name.lower()]


def make_env_factory(configs: dict[str, dict[str, Any]], level: int, rank: int, monitor_dir: Path | None) -> Callable[[], Any]:
    """Return a thunk creating one Monitor-wrapped environment."""

    def _init() -> Any:
        from stable_baselines3.common.monitor import Monitor

        from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv

        env = Boeing7478TakeoffEnv(configs=configs, level=level)
        filename = str(monitor_dir / f"monitor_{rank}.csv") if monitor_dir is not None else None
        return Monitor(env, filename=filename)

    return _init


def make_vec_env(configs: dict[str, dict[str, Any]], n_envs: int, seed: int, level: int,
                 monitor_dir: Path | None, subproc: bool = False) -> Any:
    """Create a (Dummy or Subproc) vectorised training environment and seed it."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = [make_env_factory(configs, level, i, monitor_dir) for i in range(n_envs)]
    vec = SubprocVecEnv(factories) if (subproc and n_envs > 1) else DummyVecEnv(factories)
    vec.seed(seed)
    return vec


def set_training_level(vec_env: Any, level: int) -> None:
    """Apply a curriculum level to every sub-environment."""
    vec_env.env_method("set_level", int(level))
