"""Success region, liftoff tracking and failure/termination logic.

Gymnasium convention used here:
    terminated = a true terminal outcome (SUCCESS or a failure of the task itself)
    truncated  = the time limit was reached (reason TIMEOUT); the episode is cut off artificially
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from envs.observations import FlightState
from envs.runway import Runway

SUCCESS = "SUCCESS"
TIMEOUT = "TIMEOUT"
SIMULATOR_INVALID_STATE = "SIMULATOR_INVALID_STATE"


@dataclass
class EpisodeTracker:
    """Mutable per-episode bookkeeping used by success/termination logic and rewards."""

    steps: int = 0
    liftoff_counter: int = 0
    has_lifted_off: bool = False
    airborne_steps: int = 0
    airborne_steps_before_contact: int = 0
    was_airborne_prev: bool = False
    prev_vertical_speed_fpm: float = 0.0
    success_counter: int = 0
    time_to_liftoff_s: float | None = None
    liftoff_distance_m: float | None = None
    time_to_100ft_s: float | None = None


@dataclass(frozen=True)
class TerminationResult:
    """Outcome of one termination check."""

    terminated: bool
    truncated: bool
    reason: str | None


def in_success_region(st: FlightState, cfg: dict[str, Any], target_altitude_ft: float) -> bool:
    """Return True if every success criterion holds at this instant (excluding the 5 s hold)."""
    c = cfg["success"]
    low, high = c["pitch_range_deg"]
    return bool(
        (not st.wow_any) and st.agl_ft >= target_altitude_ft and st.vertical_speed_fpm >= c["min_climb_fpm"]
        and abs(st.roll_deg) <= c["max_abs_roll_deg"] and low <= st.pitch_deg <= high
        and abs(st.heading_error_deg) <= c["max_heading_error_deg"] and st.alpha_deg <= c["max_alpha_deg"]
        and abs(st.beta_deg) <= c["max_abs_beta_deg"] and abs(st.lateral_m) <= c["max_lateral_m"]
        and st.airspeed_kt >= c["min_speed_ratio_to_vs"] * st.vs_est_kt)


def update_tracker(tracker: EpisodeTracker, st: FlightState, cfg: dict[str, Any], target_altitude_ft: float) -> bool:
    """Advance the tracker with the newest state; returns whether the success region currently holds.

    Must be called exactly once per RL transition, before :func:`check_termination`.
    """
    tracker.steps += 1
    lift = cfg["liftoff"]
    clear = (not st.wow_any) and st.agl_ft >= lift["min_agl_margin_ft"]
    tracker.liftoff_counter = tracker.liftoff_counter + 1 if clear else 0
    if (not tracker.has_lifted_off) and tracker.liftoff_counter >= lift["confirm_steps"]:
        tracker.has_lifted_off = True
        tracker.time_to_liftoff_s = st.time_s
        tracker.liftoff_distance_m = st.along_m
    tracker.airborne_steps = tracker.airborne_steps + 1 if not st.wow_any else 0
    region = in_success_region(st, cfg, target_altitude_ft)
    tracker.success_counter = tracker.success_counter + 1 if region else 0
    if tracker.time_to_100ft_s is None and (not st.wow_any) and st.agl_ft >= target_altitude_ft:
        tracker.time_to_100ft_s = st.time_s
    return region


def _touchdown_reason(st: FlightState, tracker: EpisodeTracker, fail: dict[str, Any]) -> str | None:
    """Classify re-contact with the ground after lift-off (None if no re-contact this step)."""
    touched = st.wow_any and tracker.was_airborne_prev
    if not (touched and tracker.has_lifted_off):
        return None
    crash = fail["crash"]
    sink = min(tracker.prev_vertical_speed_fpm, st.vertical_speed_fpm)
    if tracker.airborne_steps_before_contact < fail["min_airborne_steps_for_return"]:
        return None  # a bounce right after lift-off: not classified as a return
    severe = (sink <= -crash["touchdown_sink_fpm"] or abs(st.roll_deg) > crash["touchdown_max_abs_roll_deg"]
              or st.pitch_deg > crash["touchdown_max_pitch_deg"])
    return "CRASH" if severe else "UNSTABLE_AFTER_LIFTOFF"


def check_termination(st: FlightState, tracker: EpisodeTracker, runway: Runway, cfg: dict[str, Any],
                      aircraft_cfg: dict[str, Any], stable_steps: int, max_seconds: float,
                      motion_cleanliness: float = 1.0) -> TerminationResult:
    """Evaluate success, failure and timeout conditions in priority order.

    Args:
        st: Current flight state (``st.time_s`` is time since episode start).
        tracker: Tracker already updated for this step via :func:`update_tracker`.
        runway: The episode's runway.
        cfg: ``environment.yaml``.
        aircraft_cfg: ``aircraft.yaml`` (for the ground pitch limit).
        stable_steps: RL steps the success region must hold.
        max_seconds: Episode time limit.
    """
    fail = cfg["failure"]
    loc = fail["loss_of_control"]
    if tracker.success_counter >= stable_steps:
        # The flight envelope held; additionally the whole takeoff must have been CLEAN
        # (centred, smooth, level) - otherwise the episode ends with a failed success.
        min_clean = float(cfg["success"].get("min_cleanliness", 0.0))
        if min_clean > 0.0 and motion_cleanliness < min_clean:
            return TerminationResult(True, False, "UNCLEAN_TAKEOFF")
        return TerminationResult(True, False, SUCCESS)

    def terminal(reason: str) -> TerminationResult:
        return TerminationResult(True, False, reason)

    if st.agl_ft < fail["crash"]["min_agl_ft"]:
        return terminal("CRASH")
    touchdown = _touchdown_reason(st, tracker, fail)
    if touchdown is not None:
        return terminal(touchdown)
    air_heading_limit = loc["max_air_heading_error_deg"] if not st.wow_any else loc["max_ground_heading_error_deg"]
    beta_valid = st.airspeed_kt >= fail["aero_check_min_airspeed_kt"]
    if (abs(st.roll_deg) > loc["max_abs_roll_deg"] or abs(st.pitch_deg) > loc["max_abs_pitch_deg"]
            or (beta_valid and abs(st.beta_deg) > loc["max_abs_beta_deg"])
            or abs(st.heading_error_deg) > air_heading_limit):
        return terminal("LOSS_OF_CONTROL")
    if (not st.wow_any) and st.alpha_deg > fail["unstable_aoa_deg"]:
        return terminal("UNSTABLE_AFTER_LIFTOFF")
    if (st.airspeed_kt > fail["max_airspeed_kt"] or abs(st.lateral_m) > fail["max_airborne_lateral_m"]
            or (st.wow_main and st.pitch_deg > float(aircraft_cfg["envelope"]["ground_pitch_limit_deg"]))):
        return terminal("ENVELOPE_VIOLATION")
    if st.wow_any and abs(st.lateral_m) > runway.half_width_m:
        return terminal("RUNWAY_EXCURSION")
    if (not tracker.has_lifted_off) and st.along_m >= runway.length_m:
        fast = st.airspeed_kt >= fail["failed_rotation_min_speed_ratio"] * st.vr_est_kt
        return terminal("FAILED_ROTATION" if fast else "INSUFFICIENT_RUNWAY")
    na = fail["no_acceleration"]
    if st.time_s > na["check_after_s"] and st.ground_speed_mps < na["min_ground_speed_mps"] and not tracker.has_lifted_off:
        return terminal("FAILED_TO_ACCELERATE")
    if st.time_s >= max_seconds:
        return TerminationResult(False, True, TIMEOUT)
    return TerminationResult(False, False, None)


def finish_step(tracker: EpisodeTracker, st: FlightState) -> None:
    """Store values needed by the *next* step's touchdown logic (call after check_termination)."""
    tracker.was_airborne_prev = not st.wow_any
    tracker.prev_vertical_speed_fpm = st.vertical_speed_fpm
    tracker.airborne_steps_before_contact = tracker.airborne_steps
