"""A conventional, SIMULATION-ONLY takeoff controller (the "teacher").

It is not the final neural policy.  It exists to (a) verify the environment can be solved,
(b) produce example trajectories / demonstrations, (c) help debug the reward and (d) optionally
initialise learning.  It acts only on the normalised observation vector, exactly like a policy.

Control concept: full thrust -> centerline tracking with nose-wheel/rudder + heading hold ->
rotate at the estimated VR toward a target pitch -> hold pitch / speed in the initial climb.
Gains are plain constants chosen for this simulation; they are NOT a flight-certified law.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.observations import OBS_NAMES
from envs.runway import FPS_PER_KT

_IDX = {name: i for i, name in enumerate(OBS_NAMES)}
RHO_SL_SLUGS_FT3: float = 0.0023769


@dataclass
class TeacherGains:
    """Tunable teacher gains (units noted per field)."""

    lateral_to_heading_deg_per_m: float = 0.25   # desired heading offset per metre of lateral error
    max_heading_command_deg: float = 6.0
    rudder_per_heading_deg: float = 0.12         # rudder action per degree of heading error
    rudder_per_yaw_rate_dps: float = 0.10        # rudder damping per deg/s
    aileron_per_roll_deg: float = 0.10
    aileron_per_roll_rate_dps: float = 0.06
    pitch_target_deg: float = 10.0
    elevator_per_pitch_deg: float = 0.07
    elevator_per_pitch_rate_dps: float = 0.12
    ground_pitch_rate_damping: float = 0.05
    rotation_speed_margin_kt: float = 0.0        # start rotation at VR_est - margin
    low_speed_pitch_deg: float = 7.0             # reduced pitch target when below V2
    high_speed_pitch_deg: float = 12.0           # increased pitch target when well above V2
    high_speed_margin_kt: float = 25.0


class SimulatedTeacher:
    """Observation-in / action-out takeoff controller."""

    def __init__(self, configs: dict[str, dict[str, Any]], gains: TeacherGains | None = None) -> None:
        """Create the teacher from the loaded configuration dictionaries."""
        self.gains = gains or TeacherGains()
        scales = configs["environment"]["observation"]["scales"]
        self.s = {k: float(v) for k, v in scales.items()}
        ac = configs["aircraft"]
        self.oew = float(ac["overrides"]["empty_weight_lbs"]["value"])
        self.mtow = float(ac["limits"]["mtow_lbs"])
        self.wing_area = float(ac["overrides"]["wingarea_ft2"]["value"])
        self.clmax = float(ac["envelope"]["clmax_est"])
        self.vr_factor = float(ac["envelope"]["vr_factor_of_vs"])
        self.v2_factor = float(ac["envelope"]["v2_factor_of_vs"])

    def speeds_from_obs(self, obs: np.ndarray) -> tuple[float, float]:
        """Return the estimated ``(VR, V2)`` in knots CAS from the normalised mass observation."""
        weight = self.oew + float(obs[_IDX["mass_norm"]]) * (self.mtow - self.oew)
        vs_kt = math.sqrt(2.0 * weight / (RHO_SL_SLUGS_FT3 * self.wing_area * self.clmax)) / FPS_PER_KT
        return vs_kt * self.vr_factor, vs_kt * self.v2_factor

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Compute the normalised ``[throttle, elevator, aileron, rudder]`` action."""
        g, s = self.gains, self.s
        airspeed = float(obs[_IDX["airspeed"]]) * s["airspeed_kt"]
        pitch_deg = math.degrees(float(obs[_IDX["pitch"]]) * s["pitch_rad"])
        roll_deg = math.degrees(float(obs[_IDX["roll"]]) * s["roll_rad"])
        heading_err_deg = math.degrees(float(obs[_IDX["relative_heading"]]) * s["relative_heading_rad"])
        lateral_m = float(obs[_IDX["lateral_error"]]) * s["lateral_error_m"]
        p_dps = math.degrees(float(obs[_IDX["p"]]) * s["p_rad_s"])
        q_dps = math.degrees(float(obs[_IDX["q"]]) * s["q_rad_s"])
        r_dps = math.degrees(float(obs[_IDX["r"]]) * s["r_rad_s"])
        vr, v2 = self.speeds_from_obs(obs)

        # Lateral: steer toward the centerline via a bounded heading command (positive lateral = right).
        desired_heading = -float(np.clip(g.lateral_to_heading_deg_per_m * lateral_m,
                                         -g.max_heading_command_deg, g.max_heading_command_deg))
        rudder = g.rudder_per_heading_deg * (desired_heading - heading_err_deg) - g.rudder_per_yaw_rate_dps * r_dps
        aileron = -(g.aileron_per_roll_deg * roll_deg + g.aileron_per_roll_rate_dps * p_dps)

        # Longitudinal: hold attitude until VR, then rotate to the climb pitch target.
        if airspeed < vr - g.rotation_speed_margin_kt:
            elevator = -g.ground_pitch_rate_damping * q_dps
        else:
            target = g.pitch_target_deg
            if airspeed < v2:
                target = g.low_speed_pitch_deg
            elif airspeed > v2 + g.high_speed_margin_kt:
                target = g.high_speed_pitch_deg
            elevator = g.elevator_per_pitch_deg * (target - pitch_deg) - g.elevator_per_pitch_rate_dps * q_dps
        action = np.array([1.0, elevator, aileron, rudder], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)
