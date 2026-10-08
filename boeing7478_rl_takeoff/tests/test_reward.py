"""Tests for the reward function (components, config-driven weights, no reward for standing still)."""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from envs import PROJECT_ROOT, load_all_configs
from envs.observations import FlightState
from envs.reward import COMPONENT_NAMES, RewardContext, RewardFunction
from envs.termination import EpisodeTracker


def base_state(**overrides: float) -> FlightState:
    values = dict(time_s=10.0, along_m=100.0, lateral_m=0.0, remaining_m=3900.0, agl_ft=0.0, airspeed_kt=60.0,
                  ground_speed_kt=60.0, ground_speed_mps=30.0, vertical_speed_fpm=0.0, pitch_deg=0.0, roll_deg=0.0,
                  heading_error_deg=0.0, course_error_deg=0.0, alpha_deg=2.0, beta_deg=0.0, p_rad_s=0.0, q_rad_s=0.0,
                  r_rad_s=0.0, wow_nose=True, wow_main=True, wow_any=True, headwind_kt=0.0, crosswind_kt=0.0,
                  density_ratio=1.0, mass_lbs=750_000.0, cg_x_in=1327.0, mean_n1=95.0, vs_est_kt=125.0,
                  vr_est_kt=135.0, v2_est_kt=144.0)
    values.update(overrides)
    return FlightState(**values)  # type: ignore[arg-type]


def make_ctx(state: FlightState, prev: FlightState | None = None, reason: str | None = None, done: bool = False,
             region_ok: bool = False, step_norm: float = 0.0) -> RewardContext:
    return RewardContext(state=state, prev_state=prev or state, tracker=EpisodeTracker(),
                         actuator_step_norm=np.full(4, step_norm), dt_s=0.1, runway_half_width_m=30.0,
                         region_ok=region_ok, reason=reason, terminated_or_truncated=done)


@pytest.fixture(scope="module")
def reward_fn() -> RewardFunction:
    cfg = load_all_configs()
    return RewardFunction(cfg["reward"], cfg["environment"])


def test_all_components_present_and_finite(reward_fn: RewardFunction) -> None:
    total, comps = reward_fn.compute(make_ctx(base_state()))
    assert tuple(comps) == COMPONENT_NAMES
    assert all(isinstance(v, float) and np.isfinite(v) for v in comps.values())
    assert total == pytest.approx(sum(comps.values()))


def test_forward_progress_is_rewarded(reward_fn: RewardFunction) -> None:
    prev = base_state(along_m=100.0)
    assert reward_fn.progress_reward(make_ctx(base_state(along_m=107.0), prev)) > 0
    assert reward_fn.progress_reward(make_ctx(base_state(along_m=100.0), prev)) == 0


def test_standing_still_cannot_farm_alignment_reward(reward_fn: RewardFunction) -> None:
    still = base_state(ground_speed_mps=0.0, airspeed_kt=0.0)
    total, comps = reward_fn.compute(make_ctx(still))
    assert comps["centerline"] == 0.0 and comps["heading"] == 0.0
    assert total < 0.0   # only the small time penalty


def test_time_penalty_is_small_relative_to_progress(reward_fn: RewardFunction) -> None:
    per_step_progress = reward_fn.progress_reward(make_ctx(base_state(along_m=107.0), base_state(along_m=100.0)))
    assert abs(reward_fn.time_penalty(make_ctx(base_state()))) < 0.2 * per_step_progress


def test_lateral_and_roll_penalties_are_negative(reward_fn: RewardFunction) -> None:
    ctx = make_ctx(base_state(lateral_m=20.0, roll_deg=25.0, heading_error_deg=20.0))
    assert reward_fn.lateral_penalty(ctx) < 0 and reward_fn.roll_penalty(ctx) < 0 and reward_fn.heading_penalty(ctx) < 0


def test_aoa_and_sideslip_penalties_only_when_flying_speed(reward_fn: RewardFunction) -> None:
    slow = make_ctx(base_state(alpha_deg=20.0, beta_deg=80.0, airspeed_kt=10.0))
    fast = make_ctx(base_state(alpha_deg=20.0, beta_deg=10.0, airspeed_kt=150.0))
    assert reward_fn.excessive_aoa_penalty(slow) == 0.0 and reward_fn.sideslip_penalty(slow) == 0.0
    assert reward_fn.excessive_aoa_penalty(fast) < 0.0 and reward_fn.sideslip_penalty(fast) < 0.0


def test_success_bonus_only_on_success_and_faster_is_better(reward_fn: RewardFunction) -> None:
    ok_fast = make_ctx(base_state(time_s=45.0), reason="SUCCESS", done=True)
    ok_slow = make_ctx(base_state(time_s=100.0), reason="SUCCESS", done=True)
    fail = make_ctx(base_state(time_s=45.0), reason="CRASH", done=True)
    assert reward_fn.success_bonus(ok_fast) > reward_fn.success_bonus(ok_slow) > 0
    assert reward_fn.success_bonus(fail) == 0.0
    # even the slowest valid takeoff must beat any failure
    assert reward_fn.success_bonus(ok_slow) > 0 > reward_fn.failure_penalty(fail)


def test_failure_penalties_apply_per_reason(reward_fn: RewardFunction) -> None:
    for reason in ("CRASH", "RUNWAY_EXCURSION", "LOSS_OF_CONTROL", "INSUFFICIENT_RUNWAY"):
        assert reward_fn.failure_penalty(make_ctx(base_state(), reason=reason, done=True)) < 0
    assert reward_fn.failure_penalty(make_ctx(base_state(), reason="SUCCESS", done=True)) == 0
    assert reward_fn.failure_penalty(make_ctx(base_state(), reason=None, done=False)) == 0


def test_airborne_climb_and_altitude_rewards(reward_fn: RewardFunction) -> None:
    prev = base_state(agl_ft=40.0, wow_any=False, wow_main=False, wow_nose=False)
    now = dataclasses.replace(prev, agl_ft=55.0, vertical_speed_fpm=1800.0)
    assert reward_fn.climb_reward(make_ctx(now, prev)) > 0
    assert reward_fn.altitude_progress_reward(make_ctx(now, prev)) > 0
    above = dataclasses.replace(prev, agl_ft=1_500.0)
    assert reward_fn.altitude_progress_reward(make_ctx(dataclasses.replace(above, agl_ft=1_600.0), above)) == 0  # capped at target (1000 ft)


def test_oscillation_penalty_scales_with_actuator_steps(reward_fn: RewardFunction) -> None:
    calm = reward_fn.control_oscillation_penalty(make_ctx(base_state(), step_norm=0.1))
    rough = reward_fn.control_oscillation_penalty(make_ctx(base_state(), step_norm=1.0))
    assert rough < calm < 0


def test_no_hidden_numeric_reward_constants_in_source() -> None:
    """reward.py may only contain structural constants; weights must come from reward.yaml."""
    tree = ast.parse(Path(PROJECT_ROOT / "envs" / "reward.py").read_text(encoding="utf-8"))
    allowed = {0, 0.0, 1, 1.0, -1.0, 2, 5.0, 20.0, 4}   # clipping bounds of the per-step progress delta, exponents, etc.
    found = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
             and not isinstance(n.value, bool)}
    assert found <= allowed, f"unexpected numeric literals in reward.py: {sorted(found - allowed)}"


def test_reward_weights_come_from_yaml() -> None:
    cfg = load_all_configs()
    cfg["reward"]["weights"]["time_penalty"] = 1.234
    fn = RewardFunction(cfg["reward"], cfg["environment"])
    assert fn.time_penalty(make_ctx(base_state())) == pytest.approx(-1.234)


# ---------------------------------------------------------------- passenger comfort / clean trajectory
CLEAN_MOTION = dict(rms_lateral_m=0.5, max_lateral_m=1.0, rms_lateral_accel_mps2=0.02, max_roll_deg=0.1, rms_yaw_rate_dps=0.1)
WOBBLY_MOTION = dict(rms_lateral_m=9.0, max_lateral_m=25.0, rms_lateral_accel_mps2=0.75, max_roll_deg=12.0, rms_yaw_rate_dps=3.0)


def ctx_with_motion(state, prev=None, motion=None, reason=None, done=False):
    ctx = make_ctx(state, prev, reason=reason, done=done)
    return dataclasses.replace(ctx, motion=motion)


def test_new_components_exist_and_baseline_profile_zeroes_them(reward_fn: RewardFunction) -> None:
    names = ("clean_centerline", "lateral_rate_penalty", "lateral_accel_penalty", "yaw_rate_penalty", "roll_rate_penalty",
             "clean_takeoff_bonus")
    assert all(n in COMPONENT_NAMES for n in names)
    cfg = load_all_configs()
    cfg["reward"]["profile"] = "baseline"
    base = RewardFunction(cfg["reward"], cfg["environment"])
    wobbly = base_state(lateral_m=8.0, course_error_deg=6.0, r_rad_s=0.1, p_rad_s=0.1)
    _total, comps = base.compute(ctx_with_motion(wobbly, base_state(), WOBBLY_MOTION, reason="SUCCESS", done=True))
    assert all(comps[n] == 0.0 for n in names)          # 'baseline' reproduces the ORIGINAL reward exactly
    with pytest.raises(ValueError):
        cfg["reward"]["profile"] = "does_not_exist"
        RewardFunction(cfg["reward"], cfg["environment"])


def test_straight_and_centered_is_rewarded_swaying_is_penalised(reward_fn: RewardFunction) -> None:
    straight = base_state(lateral_m=0.2, course_error_deg=0.0)
    swaying = base_state(lateral_m=0.2, course_error_deg=5.0)           # same position, but drifting sideways at 2.6 m/s
    assert reward_fn.clean_centerline_reward(make_ctx(straight)) > 0.09
    assert reward_fn.lateral_rate_penalty(make_ctx(straight)) == 0.0
    assert reward_fn.lateral_rate_penalty(make_ctx(swaying)) < 0.0
    jolt = reward_fn.lateral_accel_penalty(make_ctx(swaying, straight))   # velocity jumps from 0 to 2.6 m/s in 0.1 s
    assert jolt < 0.0 and reward_fn.lateral_accel_penalty(make_ctx(straight, straight)) == 0.0
    assert reward_fn.yaw_rate_penalty(make_ctx(base_state(r_rad_s=0.1))) < 0.0 == reward_fn.yaw_rate_penalty(make_ctx(base_state()))
    assert reward_fn.roll_rate_penalty(make_ctx(base_state(p_rad_s=0.1))) < 0.0 == reward_fn.roll_rate_penalty(make_ctx(base_state()))


def test_clean_centerline_reward_decays_with_offset(reward_fn: RewardFunction) -> None:
    values = [reward_fn.clean_centerline_reward(make_ctx(base_state(lateral_m=y))) for y in (0.0, 1.0, 2.0, 4.0, 10.0)]
    assert values == sorted(values, reverse=True) and values[0] > 10 * values[3] and values[-1] < 1e-4


def test_cleanliness_score_ranks_trajectories(reward_fn: RewardFunction) -> None:
    clean, wobbly = reward_fn.cleanliness(CLEAN_MOTION), reward_fn.cleanliness(WOBBLY_MOTION)
    assert 0.95 < clean <= 1.0 and 0.0 <= wobbly < 0.2
    smooth_but_off_centre = dict(CLEAN_MOTION, rms_lateral_m=10.0, max_lateral_m=24.0)
    assert reward_fn.cleanliness(smooth_but_off_centre) < 0.4      # one bad axis (centering) must drag the score down
    assert reward_fn.cleanliness(dict(CLEAN_MOTION, rms_lateral_m=2.0)) > reward_fn.cleanliness(dict(CLEAN_MOTION, rms_lateral_m=4.0))


def test_clean_takeoff_earns_much_more_than_wobbly_takeoff(reward_fn: RewardFunction) -> None:
    final = base_state(time_s=45.0)
    clean = reward_fn.clean_takeoff_bonus(ctx_with_motion(final, motion=CLEAN_MOTION, reason="SUCCESS", done=True))
    wobbly = reward_fn.clean_takeoff_bonus(ctx_with_motion(final, motion=WOBBLY_MOTION, reason="SUCCESS", done=True))
    assert clean > 100.0 and wobbly < 0.2 * clean
    assert reward_fn.clean_takeoff_bonus(ctx_with_motion(final, motion=CLEAN_MOTION, reason="CRASH", done=True)) == 0.0
    assert reward_fn.clean_takeoff_bonus(ctx_with_motion(final, motion=None, reason="SUCCESS", done=True)) == 0.0
    total_clean, _ = reward_fn.compute(ctx_with_motion(final, motion=CLEAN_MOTION, reason="SUCCESS", done=True))
    total_wobbly, _ = reward_fn.compute(ctx_with_motion(final, motion=WOBBLY_MOTION, reason="SUCCESS", done=True))
    assert total_clean > total_wobbly + 100.0


def test_standing_still_cannot_farm_comfort_reward(reward_fn: RewardFunction) -> None:
    still = base_state(ground_speed_mps=0.0, airspeed_kt=0.0, lateral_m=0.0)
    assert reward_fn.clean_centerline_reward(make_ctx(still)) == 0.0
    total, _ = reward_fn.compute(ctx_with_motion(still, motion=CLEAN_MOTION))
    assert total < 0.0
