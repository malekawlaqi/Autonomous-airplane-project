"""Action mapping: normalised policy actions -> rate-limited actuator commands -> JSBSim commands."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_NAMES: tuple[str, ...] = ("throttle", "elevator", "aileron", "rudder")
ACTION_DIM: int = len(ACTION_NAMES)


@dataclass(frozen=True)
class JSBSimCommands:
    """Commands already expressed in JSBSim conventions (signs applied)."""

    throttle: float
    elevator_cmd: float
    aileron: float
    rudder: float
    steer: float


class ActionMapper:
    """Converts policy actions into smooth actuator motion.

    Agent-space conventions (independent of JSBSim sign conventions):
        throttle : action [-1, 1] -> lever [0, 1] (all four engines share one lever)
        elevator : +1 = full nose-up,   aileron : +1 = roll right,   rudder : +1 = yaw right
    The per-axis rate limits (units/second, from ``aircraft.yaml``) prevent instantaneous
    control-surface switching: the actuator moves toward the commanded value by at most
    ``rate * dt`` each RL step.
    """

    def __init__(self, aircraft_cfg: dict[str, Any], policy_hz: float) -> None:
        """Create the mapper.

        Args:
            aircraft_cfg: Parsed ``aircraft.yaml`` (uses the ``actuators`` section).
            policy_hz: RL control frequency in Hz.
        """
        act = aircraft_cfg["actuators"]
        self.dt = 1.0 / float(policy_hz)
        self.signs = {k: float(v) for k, v in act["signs"].items()}
        self.steer_from_rudder = bool(act["nose_wheel_steering_from_rudder"])
        low, high = act["throttle_range"]
        self.throttle_low, self.throttle_high = float(low), float(high)
        rates = act["rate_limits_per_s"]
        self.max_step = np.array([float(rates[name]) * self.dt for name in ACTION_NAMES], dtype=np.float64)
        self.state = np.zeros(ACTION_DIM, dtype=np.float64)  # [throttle01, elevator, aileron, rudder]
        self.last_step_norm = np.zeros(ACTION_DIM, dtype=np.float64)

    def reset(self) -> None:
        """Return actuators to idle throttle and neutral surfaces."""
        self.state[:] = 0.0
        self.state[0] = self.throttle_low
        self.last_step_norm[:] = 0.0

    @staticmethod
    def validate(action: np.ndarray) -> np.ndarray:
        """Return a clipped float64 copy of ``action`` or raise on wrong shape / non-finite values."""
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.shape != (ACTION_DIM,):
            raise ValueError(f"Action must have shape ({ACTION_DIM},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Action contains NaN/inf: {arr}")
        return np.clip(arr, -1.0, 1.0)

    def target_from_action(self, action: np.ndarray) -> np.ndarray:
        """Map a validated action to the target actuator vector ``[throttle01, elev, ail, rud]``."""
        target = action.copy()
        target[0] = self.throttle_low + 0.5 * (action[0] + 1.0) * (self.throttle_high - self.throttle_low)
        return target

    def apply(self, action: np.ndarray) -> JSBSimCommands:
        """Advance the rate-limited actuators one RL step toward the commanded action.

        Args:
            action: Policy output, shape (4,), nominally in [-1, 1].

        Returns:
            The JSBSim command values to hold for the next RL transition.
        """
        target = self.target_from_action(self.validate(action))
        delta = np.clip(target - self.state, -self.max_step, self.max_step)
        self.state += delta
        self.last_step_norm = delta / self.max_step  # in [-1, 1]; used by the oscillation penalty
        return self.jsbsim_commands()

    def jsbsim_commands(self) -> JSBSimCommands:
        """Convert the current actuator state to JSBSim conventions."""
        throttle, elevator, aileron, rudder = (float(v) for v in self.state)
        steer = self.signs["steer"] * rudder if self.steer_from_rudder else 0.0
        return JSBSimCommands(throttle=throttle, elevator_cmd=self.signs["elevator"] * elevator,
                              aileron=self.signs["aileron"] * aileron, rudder=self.signs["rudder"] * rudder,
                              steer=steer)
