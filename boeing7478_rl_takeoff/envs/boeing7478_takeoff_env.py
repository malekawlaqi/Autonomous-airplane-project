"""Gymnasium environment: autonomous takeoff of the Boeing 747-8 research approximation in JSBSim.

One RL environment transition = one policy action followed by ``physics_hz / policy_hz`` JSBSim
physics steps (12 at 120 Hz / 10 Hz).  No rendering is performed.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs import PROJECT_ROOT, load_all_configs
from envs.actions import ACTION_DIM, ActionMapper
from envs.jsbsim_interface import InvalidSimulatorStateError, JSBSimInterface
from envs.observations import OBS_DIM, FlightState, ObservationBuilder, compute_flight_state
from envs.randomization import EpisodeConditions, DomainRandomizer
from envs.reward import COMPONENT_NAMES, RewardContext, RewardFunction
from envs.termination import (SIMULATOR_INVALID_STATE, SUCCESS, EpisodeTracker, TerminationResult,
                              check_termination, finish_step, update_tracker)


class EpisodeStats:
    """Accumulates per-episode evaluation metrics."""

    def __init__(self, aero_gate_kt: float = 50.0) -> None:
        """Start with empty accumulators (``aero_gate_kt``: airspeed above which AoA is recorded)."""
        self.aero_gate_kt = aero_gate_kt
        self.total_reward = 0.0
        self.component_sums = {name: 0.0 for name in COMPONENT_NAMES}
        self.lateral_sq_sum = 0.0
        self.lateral_count = 0
        self.max_abs_lateral = 0.0
        self.max_roll = 0.0
        self.max_pitch = -90.0
        self.max_alpha = -90.0
        self.max_abs_heading_error = 0.0
        self.action_step_abs_sum = 0.0
        self.steps_in_region_after_100ft = 0
        self.steps_after_100ft = 0
        self.max_agl = 0.0
        self.motion_n = 0              # whole-episode motion statistics (feed the 'clean trajectory' reward)
        self.motion_lat_sq = 0.0
        self.motion_alat_sq = 0.0
        self.motion_yaw_sq = 0.0
        self.motion_max_lat = 0.0
        self.motion_max_roll = 0.0

    def observe_motion(self, st: FlightState, prev: FlightState, dt_s: float) -> None:
        """Accumulate lateral error, lateral acceleration (from the true velocity vector), yaw rate and roll."""
        accel = (st.lateral_velocity_mps - prev.lateral_velocity_mps) / dt_s
        self.motion_n += 1
        self.motion_lat_sq += st.lateral_m ** 2
        self.motion_alat_sq += accel ** 2
        self.motion_yaw_sq += math.degrees(st.r_rad_s) ** 2
        self.motion_max_lat = max(self.motion_max_lat, abs(st.lateral_m))
        self.motion_max_roll = max(self.motion_max_roll, abs(st.roll_deg))

    def motion_summary(self) -> dict[str, float]:
        """Motion statistics of the episode so far (up to and including the latest step)."""
        n = max(self.motion_n, 1)
        return {"rms_lateral_m": math.sqrt(self.motion_lat_sq / n), "max_lateral_m": self.motion_max_lat,
                "rms_lateral_accel_mps2": math.sqrt(self.motion_alat_sq / n), "max_roll_deg": self.motion_max_roll,
                "rms_yaw_rate_dps": math.sqrt(self.motion_yaw_sq / n)}

    def update(self, st: FlightState, reward: float, components: dict[str, float], step_norm: np.ndarray,
               region_ok: bool, target_altitude_ft: float, reached_target_before: bool) -> None:
        """Fold one transition into the statistics."""
        self.total_reward += reward
        for name, value in components.items():
            self.component_sums[name] += value
        self.lateral_sq_sum += st.lateral_m ** 2
        self.lateral_count += 1
        self.max_abs_lateral = max(self.max_abs_lateral, abs(st.lateral_m))
        self.max_roll = max(self.max_roll, abs(st.roll_deg))
        self.max_pitch = max(self.max_pitch, st.pitch_deg)
        if st.airspeed_kt >= self.aero_gate_kt:  # AoA is meaningless below flying speed (parked in wind)
            self.max_alpha = max(self.max_alpha, st.alpha_deg)
        self.max_abs_heading_error = max(self.max_abs_heading_error, abs(st.heading_error_deg))
        self.action_step_abs_sum += float(np.mean(np.abs(step_norm)))
        self.max_agl = max(self.max_agl, st.agl_ft)
        if reached_target_before:
            self.steps_after_100ft += 1
            self.steps_in_region_after_100ft += int(region_ok)

    def summary(self, steps: int) -> dict[str, float]:
        """Return the metrics as plain floats."""
        n = max(self.lateral_count, 1)
        return {
            "total_reward": self.total_reward, "centerline_rms_m": math.sqrt(self.lateral_sq_sum / n),
            "centerline_max_m": self.max_abs_lateral, "max_roll_deg": self.max_roll, "max_pitch_deg": self.max_pitch,
            "max_alpha_deg": self.max_alpha, "max_heading_error_deg": self.max_abs_heading_error,
            "control_roughness": self.action_step_abs_sum / max(steps, 1), "max_agl_ft": self.max_agl,
            "stabilization_fraction_after_100ft": (self.steps_in_region_after_100ft / self.steps_after_100ft
                                                   if self.steps_after_100ft else 0.0),
        }


class Boeing7478TakeoffEnv(gym.Env):
    """Takeoff environment (see module docstring).

    Observation: ``Box(-clip, clip, (27,), float32)`` (see ``envs.observations.OBS_NAMES``).
    Action: ``Box(-1, 1, (4,), float32)`` = ``[throttle, elevator, aileron, rudder]``.
    Reward: dense, component-wise (see ``envs.reward``).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, configs: dict[str, dict[str, Any]] | None = None, config_dir: str | Path | None = None,
                 level: int | None = None, fixed_conditions: EpisodeConditions | None = None,
                 project_root: str | Path | None = None) -> None:
        """Create the environment.

        Args:
            configs: Pre-loaded configuration dictionaries (keys aircraft/environment/reward/randomization).
            config_dir: Directory with the YAML files when ``configs`` is not given.
            level: Curriculum level (defaults to ``initial_curriculum_level`` in the config).
            fixed_conditions: If given, every reset uses exactly these conditions (tests/teacher).
            project_root: Project root (defaults to the repository location).
        """
        super().__init__()
        self.cfg = configs if configs is not None else load_all_configs(config_dir)
        env = self.cfg["environment"]
        self.physics_hz = float(env["physics_hz"])
        self.policy_hz = float(env["policy_hz"])
        ratio = self.physics_hz / self.policy_hz
        if abs(ratio - round(ratio)) > 1e-9 or ratio < 1:
            raise ValueError(f"physics_hz ({self.physics_hz}) must be an integer multiple of policy_hz ({self.policy_hz})")
        self.physics_steps_per_action = int(round(ratio))
        self.dt_policy = 1.0 / self.policy_hz
        self.target_altitude_ft = float(env["target_altitude_ft"])
        self.stable_steps = int(round(float(env["stable_seconds"]) * self.policy_hz))
        self.max_seconds = float(env["max_episode_seconds"])
        self.curriculum_level = int(level if level is not None else env["initial_curriculum_level"])
        self.fixed_conditions = fixed_conditions

        self.sim = JSBSimInterface(self.cfg["aircraft"], self.physics_hz, Path(project_root or PROJECT_ROOT))
        self.mapper = ActionMapper(self.cfg["aircraft"], self.policy_hz)
        self.obs_builder = ObservationBuilder(env, self.cfg["aircraft"])
        self.reward_fn = RewardFunction(self.cfg["reward"], env)
        self.randomizer = DomainRandomizer(self.cfg["randomization"], self.cfg["aircraft"])

        clip = float(env["observation"]["clip"])
        self.observation_space = spaces.Box(low=-clip, high=clip, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self.conditions: EpisodeConditions | None = None
        self.tracker = EpisodeTracker()
        self.aero_gate_kt = float(env["failure"]["aero_check_min_airspeed_kt"])
        self.stats = EpisodeStats(self.aero_gate_kt)
        self._state: FlightState | None = None
        self._last_obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._gust_rng = np.random.default_rng(0)
        self._gust = np.zeros(2)  # [head, cross] gust components in knots
        self.total_transitions = 0  # across all episodes of this env instance

    # ------------------------------------------------------------------ curriculum
    def set_level(self, level: int) -> None:
        """Set the curriculum level used by subsequent resets (callable through ``VecEnv.env_method``)."""
        if level not in self.randomizer.levels:
            raise ValueError(f"Unknown level {level}")
        self.curriculum_level = int(level)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode on the runway: stationary, engines running, takeoff-configured.

        Args:
            seed: Seeds the environment RNG; the same seed reproduces the same conditions.
            options: Optional ``{"level": int, "conditions": EpisodeConditions}`` overrides.

        Returns:
            Normalised observation and an info dictionary describing the episode conditions.
        """
        super().reset(seed=seed)
        options = options or {}
        if "conditions" in options:
            cond = options["conditions"]
        elif self.fixed_conditions is not None:
            cond = self.fixed_conditions
        else:
            episode_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
            level = int(options["level"]) if "level" in options else self.randomizer.choose_level(
                self.curriculum_level, self.np_random)
            cond = self.randomizer.sample(level, episode_seed)
        self.conditions = cond
        meta = self.sim.start_episode(cond)
        self.mapper.reset()
        self.tracker = EpisodeTracker()
        self.stats = EpisodeStats(self.aero_gate_kt)
        self._gust_rng = np.random.default_rng(cond.seed + 1)
        self._gust[:] = 0.0
        sensors = self.sim.read_sensors()
        self._state = self._make_state(sensors)
        obs = self.obs_builder.build(self._state, self.mapper.state)
        self._last_obs = obs
        info = {"level": cond.level, "episode_seed": cond.seed, "conditions": cond.to_dict(),
                "ground_agl_ref_ft": meta["ground_agl_ref_ft"], "weight_lbs": meta["weight_lbs"],
                "cg_x_in": meta["cg_x_in"], "vs_est_kt": self._state.vs_est_kt, "vr_est_kt": self._state.vr_est_kt}
        return obs, info

    def _make_state(self, sensors: dict[str, float]) -> FlightState:
        """Derive the runway-relative flight state from raw sensors."""
        assert self.conditions is not None
        along, lateral = self.sim.runway_xy(sensors)
        st = compute_flight_state(sensors, self.conditions.runway, along, lateral, self.sim.ground_agl_ref_ft,
                                  self.cfg["aircraft"])
        return dataclasses.replace(st, time_s=sensors["sim_time_s"] - self.sim.sim_time_start_s)

    def _update_gust(self) -> None:
        """Advance the Gauss-Markov gust process and push steady+gust wind into JSBSim."""
        assert self.conditions is not None
        wx = self.conditions.weather
        if wx.gust_sigma_kt > 0.0:
            decay = math.exp(-self.dt_policy / wx.gust_time_constant_s)
            noise = self._gust_rng.standard_normal(2) * wx.gust_sigma_kt * math.sqrt(1.0 - decay ** 2)
            self._gust = self._gust * decay + noise
            self.sim.set_wind_from_components(self.conditions.runway, wx.headwind_kt, wx.crosswind_from_right_kt,
                                              float(self._gust[0]), float(self._gust[1]))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one policy action for ``physics_steps_per_action`` physics steps.

        Returns:
            ``(observation, reward, terminated, truncated, info)`` following Gymnasium conventions.
        """
        if self._state is None or self.conditions is None:
            raise RuntimeError("Call reset() before step()")
        cmds = self.mapper.apply(action)
        prev = self._state
        try:
            self.sim.set_flight_controls(cmds.throttle, cmds.elevator_cmd, cmds.aileron, cmds.rudder, cmds.steer)
            self._update_gust()
            self.sim.step(self.physics_steps_per_action)
            sensors = self.sim.read_sensors()
            st = self._make_state(sensors)
            obs = self.obs_builder.build(st, self.mapper.state)
        except (InvalidSimulatorStateError, ValueError) as exc:
            return self._invalid_state_result(str(exc))
        self.total_transitions += 1

        self.stats.observe_motion(st, prev, self.dt_policy)
        motion = self.stats.motion_summary()
        cleanliness = self.reward_fn.cleanliness(motion)
        region_ok = update_tracker(self.tracker, st, self.cfg["environment"], self.target_altitude_ft)
        term = check_termination(st, self.tracker, self.conditions.runway, self.cfg["environment"],
                                 self.cfg["aircraft"], self.stable_steps, self.max_seconds, cleanliness)
        done = term.terminated or term.truncated
        ctx = RewardContext(state=st, prev_state=prev, tracker=self.tracker,
                            actuator_step_norm=self.mapper.last_step_norm.copy(), dt_s=self.dt_policy,
                            runway_half_width_m=self.conditions.runway.half_width_m, region_ok=region_ok,
                            reason=term.reason, terminated_or_truncated=done, motion=self.stats.motion_summary())
        reward, components = self.reward_fn.compute(ctx)
        reached_before = self.tracker.time_to_100ft_s is not None
        self.stats.update(st, reward, components, self.mapper.last_step_norm, region_ok, self.target_altitude_ft,
                          reached_before)
        finish_step(self.tracker, st)
        self._state = st
        self._last_obs = obs
        info = self._build_info(st, term, components, cmds_state=self.mapper.state)
        return obs, float(reward), term.terminated, term.truncated, info

    def _invalid_state_result(self, message: str) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Terminate with SIMULATOR_INVALID_STATE (the error text is kept in ``info``)."""
        penalty = -float(self.cfg["reward"]["failure_penalties"][SIMULATOR_INVALID_STATE])
        term = TerminationResult(True, False, SIMULATOR_INVALID_STATE)
        components = {name: 0.0 for name in COMPONENT_NAMES}
        components["failure_penalty"] = penalty
        self.stats.total_reward += penalty
        self.stats.component_sums["failure_penalty"] += penalty
        assert self._state is not None
        info = self._build_info(self._state, term, components, cmds_state=self.mapper.state)
        info["simulator_error"] = message
        return self._last_obs.copy(), penalty, True, False, info

    def _build_info(self, st: FlightState, term: TerminationResult, components: dict[str, float],
                    cmds_state: np.ndarray) -> dict[str, Any]:
        """Assemble the info dict (with a full episode summary when the episode ends)."""
        assert self.conditions is not None
        info: dict[str, Any] = {
            "reason": term.reason, "reward_components": components, "time_s": st.time_s, "along_m": st.along_m,
            "lateral_m": st.lateral_m, "agl_ft": st.agl_ft, "airspeed_kt": st.airspeed_kt,
            "vertical_speed_fpm": st.vertical_speed_fpm, "pitch_deg": st.pitch_deg, "roll_deg": st.roll_deg,
            "heading_error_deg": st.heading_error_deg, "alpha_deg": st.alpha_deg, "beta_deg": st.beta_deg,
            "actuators": cmds_state.copy(), "success": term.reason == SUCCESS,
            "success_counter": self.tracker.success_counter,
        }
        if term.terminated or term.truncated:
            c = self.conditions
            tr = self.tracker
            summary = self.stats.summary(tr.steps)
            summary.update({
                "success": term.reason == SUCCESS, "reason": term.reason or "NONE", "transitions": tr.steps,
                "duration_s": st.time_s, "level": c.level, "mass_lbs": st.mass_lbs, "cg_x_in": st.cg_x_in,
                "runway_length_m": c.runway.length_m, "runway_elevation_ft": c.runway.elevation_ft,
                "runway_surface": c.runway.surface, "static_friction_factor": c.runway.static_friction_factor,
                "headwind_kt": c.weather.headwind_kt, "crosswind_kt": c.weather.crosswind_from_right_kt,
                "gust_sigma_kt": c.weather.gust_sigma_kt, "turbulence_severity": c.weather.turbulence_severity,
                "delta_isa_c": c.weather.delta_isa_c, "qnh_hpa": c.weather.qnh_hpa, "density_ratio": st.density_ratio,
                "liftoff_time_s": tr.time_to_liftoff_s if tr.time_to_liftoff_s is not None else float("nan"),
                "runway_distance_used_m": tr.liftoff_distance_m if tr.liftoff_distance_m is not None else float("nan"),
                "time_to_100ft_s": tr.time_to_100ft_s if tr.time_to_100ft_s is not None else float("nan"),
                "takeoff_time_s": st.time_s - self.stable_steps * self.dt_policy if term.reason == SUCCESS else float("nan"),
                "final_airspeed_kt": st.airspeed_kt, "final_agl_ft": st.agl_ft, "final_vs_fpm": st.vertical_speed_fpm,
                "vs_est_kt": st.vs_est_kt, "vr_est_kt": st.vr_est_kt, "seed": c.seed,
            })
            motion = self.stats.motion_summary()
            summary.update({"cleanliness": self.reward_fn.cleanliness(motion), "rms_lateral_accel_mps2": motion["rms_lateral_accel_mps2"],
                            "rms_yaw_rate_dps": motion["rms_yaw_rate_dps"], "reward_profile": self.reward_fn.profile})
            summary.update({f"rc_{k}": v for k, v in self.stats.component_sums.items()})
            info["episode_summary"] = summary
        return info

    def close(self) -> None:
        """Release JSBSim resources."""
        self.sim.close()
