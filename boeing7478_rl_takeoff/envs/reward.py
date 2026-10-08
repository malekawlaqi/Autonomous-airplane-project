"""Dense, component-wise reward.  Every numeric constant comes from ``configs/reward.yaml``.

Each component is a small method returning a float (penalties are returned as NEGATIVE numbers),
so the total reward is simply the sum of the component dictionary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.observations import FlightState
from envs.termination import SUCCESS, TIMEOUT, EpisodeTracker

COMPONENT_NAMES: tuple[str, ...] = (
    "progress", "centerline", "heading", "acceleration", "rotation", "climb", "altitude_progress",
    "stability", "success_bonus", "time_penalty", "lateral_penalty", "roll_penalty", "heading_penalty",
    "aoa_penalty", "sideslip_penalty", "control_oscillation_penalty", "premature_rotation_penalty",
    "clean_centerline", "lateral_rate_penalty", "lateral_accel_penalty", "yaw_rate_penalty", "roll_rate_penalty",
    "clean_takeoff_bonus", "failure_penalty",
)


@dataclass(frozen=True)
class RewardContext:
    """Everything a reward evaluation needs for one RL transition."""

    state: FlightState
    prev_state: FlightState
    tracker: EpisodeTracker
    actuator_step_norm: np.ndarray
    dt_s: float
    runway_half_width_m: float
    region_ok: bool
    reason: str | None
    terminated_or_truncated: bool
    motion: dict[str, float] | None = None   # whole-episode motion statistics up to now (see EpisodeStats.motion_summary)


class RewardFunction:
    """Computes the reward and its separate components."""

    def __init__(self, reward_cfg: dict[str, Any], env_cfg: dict[str, Any]) -> None:
        """Create the reward function.

        Args:
            reward_cfg: Parsed ``reward.yaml``.
            env_cfg: Parsed ``environment.yaml`` (target altitude and time limit).
        """
        self.w: dict[str, float] = {k: float(v) for k, v in reward_cfg["weights"].items()}
        self.s: dict[str, float] = {k: float(v) for k, v in reward_cfg["shaping"].items()}
        self.profile = str(reward_cfg.get("profile", "clean"))
        overrides = reward_cfg.get("profiles", {}).get(self.profile)
        if overrides is None:
            raise ValueError(f"reward profile '{self.profile}' not defined in reward.yaml 'profiles'")
        self.w.update({k: float(v) for k, v in overrides.get("weights", {}).items()})
        self.s.update({k: float(v) for k, v in overrides.get("shaping", {}).items()})
        self.failure_penalties: dict[str, float] = {k: float(v) for k, v in reward_cfg["failure_penalties"].items()}
        self.target_alt_ft = float(env_cfg["target_altitude_ft"])
        self.max_seconds = float(env_cfg["max_episode_seconds"])

    # ------------------------------------------------------------------ helpers
    def _gate(self, st: FlightState) -> float:
        """0..1 factor that is 0 when stationary: alignment rewards cannot be farmed by standing still."""
        return float(np.clip(st.ground_speed_mps / self.s["speed_gate_mps"], 0.0, 1.0))

    def _capped_sq(self, excess: float, scale: float) -> float:
        """Squared normalised excess, capped to avoid reward spikes."""
        return min((excess / scale) ** 2, self.s["penalty_term_cap"])

    # ------------------------------------------------------------------ positive components
    def progress_reward(self, ctx: RewardContext) -> float:
        """Reward forward progress along the runway (metres this step, clipped)."""
        delta = float(np.clip(ctx.state.along_m - ctx.prev_state.along_m, -5.0, 20.0))
        return self.w["progress"] * delta

    def centerline_reward(self, ctx: RewardContext) -> float:
        """Reward small lateral error (only while moving)."""
        return self.w["centerline"] * self._gate(ctx.state) * math.exp(-(ctx.state.lateral_m / self.s["centerline_scale_m"]) ** 2)

    def heading_reward(self, ctx: RewardContext) -> float:
        """Reward alignment with the runway heading (only while moving)."""
        return self.w["heading"] * self._gate(ctx.state) * math.exp(-(ctx.state.heading_error_deg / self.s["heading_scale_deg"]) ** 2)

    def acceleration_reward(self, ctx: RewardContext) -> float:
        """Reward controlled acceleration on the ground before rotation speed."""
        st = ctx.state
        if st.wow_any is False or st.airspeed_kt >= st.vr_est_kt:
            return 0.0
        accel = (st.ground_speed_mps - ctx.prev_state.ground_speed_mps) / ctx.dt_s
        return self.w["acceleration"] * float(np.clip(accel / self.s["accel_target_mps2"], 0.0, 1.0))

    def rotation_reward(self, ctx: RewardContext) -> float:
        """Reward a controlled pitch rate toward the initial-climb pitch near/after rotation speed."""
        st = ctx.state
        if st.airspeed_kt < st.vr_est_kt - self.s["rotation_speed_margin_kt"] or st.agl_ft > self.target_alt_ft:
            return 0.0
        desired_dps = float(np.clip(self.s["pitch_rate_gain_per_s"] * (self.s["pitch_target_deg"] - st.pitch_deg),
                                    self.s["pitch_rate_min_dps"], self.s["pitch_rate_max_dps"]))
        error = math.degrees(st.q_rad_s) - desired_dps
        return self.w["rotation"] * math.exp(-(error / self.s["pitch_rate_tolerance_dps"]) ** 2)

    def climb_reward(self, ctx: RewardContext) -> float:
        """Reward positive climb rate once airborne."""
        st = ctx.state
        if st.wow_any:
            return 0.0
        return self.w["climb"] * float(np.clip(st.vertical_speed_fpm / self.s["climb_target_fpm"], 0.0, 1.0))

    def altitude_progress_reward(self, ctx: RewardContext) -> float:
        """Reward AGL gained this step, up to the target altitude (potential-based)."""
        cap = self.target_alt_ft
        now = min(max(ctx.state.agl_ft, 0.0), cap)
        before = min(max(ctx.prev_state.agl_ft, 0.0), cap)
        return self.w["altitude_progress"] * (now - before)

    def stability_reward(self, ctx: RewardContext) -> float:
        """Reward each step spent inside the success region."""
        return self.w["stability"] if ctx.region_ok else 0.0

    def success_bonus(self, ctx: RewardContext) -> float:
        """Large bonus for a completed valid takeoff, plus a time bonus (faster = more, only on success)."""
        if ctx.reason != SUCCESS:
            return 0.0
        remaining_fraction = max(0.0, 1.0 - ctx.state.time_s / self.max_seconds)
        return self.w["success_bonus"] + self.w["success_time_bonus"] * remaining_fraction

    # ------------------------------------------------------------------ penalties (negative values)
    def time_penalty(self, ctx: RewardContext) -> float:
        """Small constant cost per RL step."""
        return -self.w["time_penalty"]

    def lateral_penalty(self, ctx: RewardContext) -> float:
        """Penalise centerline error relative to runway half width."""
        return -self.w["lateral_penalty"] * self._capped_sq(ctx.state.lateral_m, ctx.runway_half_width_m)

    def roll_penalty(self, ctx: RewardContext) -> float:
        """Penalise bank angle beyond a small deadband."""
        excess = max(abs(ctx.state.roll_deg) - self.s["roll_deadband_deg"], 0.0)
        return -self.w["roll_penalty"] * self._capped_sq(excess, self.s["roll_scale_deg"])

    def heading_penalty(self, ctx: RewardContext) -> float:
        """Penalise heading error beyond a small deadband."""
        excess = max(abs(ctx.state.heading_error_deg) - self.s["heading_deadband_deg"], 0.0)
        return -self.w["heading_penalty"] * self._capped_sq(excess, self.s["heading_penalty_scale_deg"])

    def excessive_aoa_penalty(self, ctx: RewardContext) -> float:
        """Penalise angle of attack above the soft limit."""
        if ctx.state.airspeed_kt < self.s["aero_gate_airspeed_kt"]:
            return 0.0
        excess = max(ctx.state.alpha_deg - self.s["aoa_soft_limit_deg"], 0.0)
        return -self.w["aoa_penalty"] * self._capped_sq(excess, self.s["aoa_scale_deg"])

    def sideslip_penalty(self, ctx: RewardContext) -> float:
        """Penalise sideslip beyond a small deadband."""
        if ctx.state.airspeed_kt < self.s["aero_gate_airspeed_kt"]:
            return 0.0
        excess = max(abs(ctx.state.beta_deg) - self.s["beta_deadband_deg"], 0.0)
        return -self.w["sideslip_penalty"] * self._capped_sq(excess, self.s["beta_scale_deg"])

    def control_oscillation_penalty(self, ctx: RewardContext) -> float:
        """Penalise large actuator moves (each channel normalised to its per-step rate limit)."""
        return -self.w["control_oscillation_penalty"] * float(np.sum(np.square(ctx.actuator_step_norm)))

    def premature_rotation_penalty(self, ctx: RewardContext) -> float:
        """Penalise nose-up attitude well before rotation speed."""
        st = ctx.state
        if not st.wow_any or st.airspeed_kt >= st.vr_est_kt - self.s["premature_speed_margin_kt"]:
            return 0.0
        excess = max(st.pitch_deg - self.s["premature_pitch_deg"], 0.0)
        return -self.w["premature_rotation_penalty"] * self._capped_sq(excess, self.s["premature_scale_deg"])

    # ------------------------------------------------------------------ passenger comfort / clean trajectory
    def _deadzone_sq(self, value: float, deadband: float, scale: float) -> float:
        """Squared, capped excess of ``|value|`` over a deadband (shared by the comfort penalties)."""
        return self._capped_sq(max(abs(value) - deadband, 0.0), scale)

    def clean_centerline_reward(self, ctx: RewardContext) -> float:
        """Extra reward ONLY when the aircraft is really on the centerline (tight gaussian, gated by speed)."""
        return self.w["clean_centerline"] * self._gate(ctx.state) * math.exp(-(ctx.state.lateral_m / self.s["clean_centerline_scale_m"]) ** 2)

    def lateral_rate_penalty(self, ctx: RewardContext) -> float:
        """Penalise sideways velocity: swaying left/right is exactly what a straight takeoff avoids."""
        return -self.w["lateral_rate_penalty"] * self._deadzone_sq(ctx.state.lateral_velocity_mps, self.s["lateral_rate_deadband_mps"],
                                                                   self.s["lateral_rate_scale_mps"])

    def lateral_accel_penalty(self, ctx: RewardContext) -> float:
        """Penalise sideways acceleration (the jolt passengers feel when the path bends)."""
        accel = (ctx.state.lateral_velocity_mps - ctx.prev_state.lateral_velocity_mps) / ctx.dt_s
        return -self.w["lateral_accel_penalty"] * self._deadzone_sq(accel, self.s["lateral_accel_deadband_mps2"],
                                                                    self.s["lateral_accel_scale_mps2"])

    def yaw_rate_penalty(self, ctx: RewardContext) -> float:
        """Penalise the nose swinging left/right."""
        return -self.w["yaw_rate_penalty"] * self._deadzone_sq(math.degrees(ctx.state.r_rad_s), self.s["yaw_rate_deadband_dps"],
                                                               self.s["yaw_rate_scale_dps"])

    def roll_rate_penalty(self, ctx: RewardContext) -> float:
        """Penalise wing rocking."""
        return -self.w["roll_rate_penalty"] * self._deadzone_sq(math.degrees(ctx.state.p_rad_s), self.s["roll_rate_deadband_dps"],
                                                                self.s["roll_rate_scale_dps"])

    def cleanliness(self, motion: dict[str, float]) -> float:
        """Trajectory cleanliness in [0, 1] = exp(-mean(x_i^2)), x_i = statistic_i / scale_i.

        The statistics are RMS and max lateral error, RMS lateral acceleration and max roll.  Because the exponent
        sums the squares, ANY bad axis (e.g. 10 m off the centerline) pulls the score down even if the others are perfect.
        """
        terms = (motion["rms_lateral_m"] / self.s["clean_rms_lateral_scale_m"],
                 motion["max_lateral_m"] / self.s["clean_max_lateral_scale_m"],
                 motion["rms_lateral_accel_mps2"] / self.s["clean_lateral_accel_scale_mps2"],
                 motion["max_roll_deg"] / self.s["clean_max_roll_scale_deg"])
        return float(math.exp(-sum(t ** 2 for t in terms) / len(terms)))

    def clean_takeoff_bonus(self, ctx: RewardContext) -> float:
        """Bonus for a valid takeoff proportional to how clean (straight, smooth, level) the whole run was."""
        if ctx.reason != SUCCESS or ctx.motion is None:
            return 0.0
        return self.w["clean_takeoff_bonus"] * self.cleanliness(ctx.motion)

    def failure_penalty(self, ctx: RewardContext) -> float:
        """Terminal penalty for the failure reason (0 for success / still running)."""
        if not ctx.terminated_or_truncated or ctx.reason in (None, SUCCESS):
            return 0.0
        key = TIMEOUT if ctx.reason == TIMEOUT else ctx.reason
        return -self.failure_penalties[key]

    # ------------------------------------------------------------------ total
    def compute(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """Return ``(total_reward, components)``; every component is a finite float."""
        components = {
            "progress": self.progress_reward(ctx), "centerline": self.centerline_reward(ctx),
            "heading": self.heading_reward(ctx), "acceleration": self.acceleration_reward(ctx),
            "rotation": self.rotation_reward(ctx), "climb": self.climb_reward(ctx),
            "altitude_progress": self.altitude_progress_reward(ctx), "stability": self.stability_reward(ctx),
            "success_bonus": self.success_bonus(ctx), "time_penalty": self.time_penalty(ctx),
            "lateral_penalty": self.lateral_penalty(ctx), "roll_penalty": self.roll_penalty(ctx),
            "heading_penalty": self.heading_penalty(ctx), "aoa_penalty": self.excessive_aoa_penalty(ctx),
            "sideslip_penalty": self.sideslip_penalty(ctx),
            "control_oscillation_penalty": self.control_oscillation_penalty(ctx),
            "premature_rotation_penalty": self.premature_rotation_penalty(ctx),
            "clean_centerline": self.clean_centerline_reward(ctx), "lateral_rate_penalty": self.lateral_rate_penalty(ctx),
            "lateral_accel_penalty": self.lateral_accel_penalty(ctx), "yaw_rate_penalty": self.yaw_rate_penalty(ctx),
            "roll_rate_penalty": self.roll_rate_penalty(ctx), "clean_takeoff_bonus": self.clean_takeoff_bonus(ctx),
            "failure_penalty": self.failure_penalty(ctx),
        }
        return float(sum(components.values())), components
