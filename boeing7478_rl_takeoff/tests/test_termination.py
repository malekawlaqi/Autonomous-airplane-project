"""Tests for the success region and every termination/truncation branch (synthetic states)."""
from __future__ import annotations

import dataclasses

import pytest

from envs import load_all_configs
from envs.observations import FlightState
from envs.runway import Runway
from envs.termination import (EpisodeTracker, check_termination, finish_step, in_success_region, update_tracker)

CFG = load_all_configs()
ENV, AC = CFG["environment"], CFG["aircraft"]
TARGET = float(ENV["target_altitude_ft"])
STABLE_STEPS = int(round(ENV["stable_seconds"] * ENV["policy_hz"]))
RUNWAY = Runway(length_m=4000.0, width_m=60.0, heading_deg=0.0, elevation_ft=0.0)


def ground(**kw: float) -> FlightState:
    v = dict(time_s=10.0, along_m=500.0, lateral_m=0.0, remaining_m=3500.0, agl_ft=0.0, airspeed_kt=100.0,
             ground_speed_kt=100.0, ground_speed_mps=51.0, vertical_speed_fpm=0.0, pitch_deg=0.0, roll_deg=0.0,
             heading_error_deg=0.0, course_error_deg=0.0, alpha_deg=2.0, beta_deg=0.0, p_rad_s=0.0, q_rad_s=0.0,
             r_rad_s=0.0, wow_nose=True, wow_main=True, wow_any=True, headwind_kt=0.0, crosswind_kt=0.0,
             density_ratio=1.0, mass_lbs=750_000.0, cg_x_in=1327.0, mean_n1=95.0, vs_est_kt=125.0, vr_est_kt=135.0,
             v2_est_kt=144.0)
    v.update(kw)
    return FlightState(**v)  # type: ignore[arg-type]


def flying(**kw: float) -> FlightState:
    v = dict(agl_ft=1100.0, airspeed_kt=165.0, vertical_speed_fpm=1500.0, pitch_deg=9.0, wow_nose=False, wow_main=False,
             wow_any=False, along_m=2000.0, remaining_m=2000.0)
    v.update(kw)
    return ground(**v)


def step(tracker: EpisodeTracker, st: FlightState, cleanliness: float = 1.0):
    update_tracker(tracker, st, ENV, TARGET)
    result = check_termination(st, tracker, RUNWAY, ENV, AC, STABLE_STEPS, ENV["max_episode_seconds"], cleanliness)
    finish_step(tracker, st)
    return result


def test_stable_seconds_is_50_steps() -> None:
    assert STABLE_STEPS == 50


def test_success_region_true_and_each_criterion_can_break_it() -> None:
    assert in_success_region(flying(), ENV, TARGET)
    breakers = dict(agl_ft=50.0, vertical_speed_fpm=0.0, roll_deg=20.0, pitch_deg=-2.0, heading_error_deg=40.0,
                    alpha_deg=14.0, beta_deg=10.0, lateral_m=500.0, airspeed_kt=100.0, wow_any=True)
    for field, value in breakers.items():
        assert not in_success_region(flying(**{field: value}), ENV, TARGET), field


def test_success_requires_five_continuous_seconds() -> None:
    tracker = EpisodeTracker()
    for i in range(STABLE_STEPS - 1):
        r = step(tracker, flying(time_s=40.0 + 0.1 * i))
        assert not r.terminated and r.reason is None
    r = step(tracker, flying(time_s=45.0))
    assert r.terminated and not r.truncated and r.reason == "SUCCESS"


def test_success_counter_resets_when_region_is_left() -> None:
    tracker = EpisodeTracker()
    for _ in range(30):
        step(tracker, flying())
    step(tracker, flying(roll_deg=20.0))
    assert tracker.success_counter == 0
    for _ in range(STABLE_STEPS - 1):
        assert not step(tracker, flying()).terminated


def test_liftoff_alone_is_not_success() -> None:
    tracker = EpisodeTracker()
    for _ in range(100):
        r = step(tracker, flying(agl_ft=20.0))  # airborne but far below 100 ft
    assert tracker.has_lifted_off and not r.terminated


def test_runway_excursion() -> None:
    r = step(EpisodeTracker(), ground(lateral_m=45.0))
    assert r.terminated and r.reason == "RUNWAY_EXCURSION"


def test_runway_exhausted_labels() -> None:
    slow = step(EpisodeTracker(), ground(along_m=4001.0, remaining_m=-1.0, airspeed_kt=90.0))
    fast = step(EpisodeTracker(), ground(along_m=4001.0, remaining_m=-1.0, airspeed_kt=150.0))
    assert slow.reason == "INSUFFICIENT_RUNWAY" and fast.reason == "FAILED_ROTATION"
    assert slow.terminated and fast.terminated


def test_loss_of_control() -> None:
    for kw in (dict(roll_deg=70.0), dict(pitch_deg=50.0), dict(heading_error_deg=60.0)):
        r = step(EpisodeTracker(), ground(**kw))
        assert r.terminated and r.reason == "LOSS_OF_CONTROL", kw


def test_sideslip_is_not_checked_below_flying_speed() -> None:
    r = step(EpisodeTracker(), ground(beta_deg=89.0, airspeed_kt=10.0))
    assert not r.terminated
    r2 = step(EpisodeTracker(), ground(beta_deg=45.0, airspeed_kt=120.0))
    assert r2.terminated and r2.reason == "LOSS_OF_CONTROL"


def test_unstable_aoa_after_liftoff_and_envelope_violations() -> None:
    assert step(EpisodeTracker(), flying(alpha_deg=18.0)).reason == "UNSTABLE_AFTER_LIFTOFF"
    assert step(EpisodeTracker(), flying(airspeed_kt=300.0)).reason == "ENVELOPE_VIOLATION"
    assert step(EpisodeTracker(), ground(pitch_deg=14.0)).reason == "ENVELOPE_VIOLATION"   # tail-strike proxy


def test_crash_on_hard_return_and_unstable_on_soft_return() -> None:
    for sink, expected in ((-2500.0, "CRASH"), (-100.0, "UNSTABLE_AFTER_LIFTOFF")):
        tracker = EpisodeTracker()
        for _ in range(12):
            assert not step(tracker, flying(agl_ft=30.0, vertical_speed_fpm=sink, alpha_deg=5.0, pitch_deg=5.0)).terminated
        r = step(tracker, ground(vertical_speed_fpm=sink, pitch_deg=5.0))
        assert r.terminated and r.reason == expected, (sink, r.reason)


def test_below_ground_is_crash() -> None:
    assert step(EpisodeTracker(), ground(agl_ft=-30.0)).reason == "CRASH"


def test_failed_to_accelerate() -> None:
    r = step(EpisodeTracker(), ground(time_s=35.0, ground_speed_mps=1.0, airspeed_kt=3.0, along_m=60.0, remaining_m=3940.0))
    assert r.terminated and r.reason == "FAILED_TO_ACCELERATE"


def test_unclean_trajectory_that_reaches_altitude_is_not_success() -> None:
    tracker = EpisodeTracker()
    for _ in range(STABLE_STEPS - 1):
        assert not step(tracker, flying(time_s=40.0 + 0.1 * _)).terminated
    # Envelope criteria held for the full 5 s, but the trajectory was wobbly:
    assert step(tracker, flying(time_s=46.0), cleanliness=0.3).reason == "UNCLEAN_TAKEOFF"
    # ...and with a clean trajectory the same envelope hold is a SUCCESS:
    tracker2 = EpisodeTracker()
    for i in range(STABLE_STEPS - 1):
        step(tracker2, flying(time_s=40.0 + 0.1 * i))
    assert step(tracker2, flying(time_s=46.0), cleanliness=0.9).reason == "SUCCESS"


def test_timeout_is_truncation_not_termination() -> None:
    r = step(EpisodeTracker(), flying(time_s=181.0, agl_ft=50.0))
    assert r.truncated and not r.terminated and r.reason == "TIMEOUT"


def test_running_state_does_not_terminate() -> None:
    r = step(EpisodeTracker(), ground())
    assert not r.terminated and not r.truncated and r.reason is None


def test_state_dataclass_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        ground().agl_ft = 1.0  # type: ignore[misc]
