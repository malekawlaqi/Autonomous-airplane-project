"""Curriculum manager: advance the randomization level when evaluation success is high enough."""
from __future__ import annotations

from typing import Any


class CurriculumManager:
    """Tracks the current curriculum level (1..max_level) and decides when to advance.

    Advancement depends on the **evaluation** success rate (deterministic policy, separate
    environment), never on noisy training returns.
    """

    def __init__(self, curriculum_cfg: dict[str, Any], start_level: int = 1) -> None:
        """Create the manager.

        Args:
            curriculum_cfg: The ``curriculum`` section of the training YAML.
            start_level: Level to start (or resume) from.
        """
        self.enabled = bool(curriculum_cfg["enabled"])
        self.threshold = float(curriculum_cfg["success_threshold"])
        self.min_eval_episodes = int(curriculum_cfg["min_eval_episodes"])
        self.max_level = int(curriculum_cfg["max_level"])
        self.level = int(start_level)
        self.history: list[dict[str, float]] = []

    def update(self, eval_success_rate: float, n_eval_episodes: int, num_timesteps: int) -> bool:
        """Record an evaluation and advance the level if the threshold is met.

        Returns:
            True if the level was increased.
        """
        advanced = bool(self.enabled and self.level < self.max_level
                        and n_eval_episodes >= self.min_eval_episodes and eval_success_rate > self.threshold)
        self.history.append({"timesteps": float(num_timesteps), "level": float(self.level),
                             "eval_success_rate": float(eval_success_rate), "advanced": float(advanced)})
        if advanced:
            self.level += 1
        return advanced

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state (saved next to checkpoints for exact resume)."""
        return {"level": self.level, "history": self.history}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state written by :meth:`state_dict`."""
        self.level = int(state["level"])
        self.history = list(state.get("history", []))
