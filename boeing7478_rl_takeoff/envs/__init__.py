"""Gymnasium environment package for the Boeing 747-8 (research approximation) RL takeoff task."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
CONFIG_FILES: dict[str, str] = {
    "aircraft": "aircraft.yaml",
    "environment": "environment.yaml",
    "reward": "reward.yaml",
    "randomization": "domain_randomization.yaml",
}


ENV_ID: str = "Boeing7478Takeoff-v0"


def register_env() -> None:
    """Register the Gymnasium environment id (idempotent)."""
    import gymnasium as gym

    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="envs.boeing7478_takeoff_env:Boeing7478TakeoffEnv")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary (errors are not swallowed)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at top level")
    return data


def load_all_configs(config_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the aircraft, environment, reward and randomization configuration files.

    Args:
        config_dir: Directory containing the YAML files (defaults to ``<project>/configs``).

    Returns:
        Mapping with keys ``aircraft``, ``environment``, ``reward``, ``randomization``.
    """
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return {key: load_yaml(directory / filename) for key, filename in CONFIG_FILES.items()}
