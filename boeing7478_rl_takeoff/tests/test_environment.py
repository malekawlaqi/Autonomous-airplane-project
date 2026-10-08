"""Integration tests: the Gymnasium environment with real JSBSim."""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import pytest

from envs import ENV_ID, load_all_configs, register_env
from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv
from envs.randomization import DomainRandomizer
from teacher.simulated_teacher import SimulatedTeacher


@pytest.fixture(scope="module")
def configs() -> dict:
    return load_all_configs()


def run(env: Boeing7478TakeoffEnv, policy, seed: int, level: int = 1) -> dict:
    obs, _ = env.reset(seed=seed, options={"level": level})
    while True:
        obs, _r, te, tr, info = env.step(policy(obs))
        if te or tr:
            return info


def test_frequencies_and_physics_steps_per_action(configs: dict) -> None:
    env = Boeing7478TakeoffEnv(configs=configs)
    assert (env.physics_hz, env.policy_hz, env.physics_steps_per_action) == (120.0, 10.0, 12)


def test_frequencies_are_configurable(configs: dict) -> None:
    cfg = copy.deepcopy(configs)
    cfg["environment"]["physics_hz"], cfg["environment"]["policy_hz"] = 60, 20
    assert Boeing7478TakeoffEnv(configs=cfg).physics_steps_per_action == 3
    cfg["environment"]["policy_hz"] = 7
    with pytest.raises(ValueError):
        Boeing7478TakeoffEnv(configs=cfg)


def test_spaces_and_reset(configs: dict) -> None:
    env = Boeing7478TakeoffEnv(configs=configs)
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (4,) and env.action_space.dtype == np.float32
    assert env.action_space.low.tolist() == [-1.0] * 4 and env.action_space.high.tolist() == [1.0] * 4
    assert obs.shape == env.observation_space.shape and env.observation_space.contains(obs)
    assert info["level"] == 1 and info["vr_est_kt"] > info["vs_est_kt"]


def test_aircraft_starts_stationary_on_runway(configs: dict) -> None:
    env = Boeing7478TakeoffEnv(configs=configs)
    _obs, info = env.reset(seed=4)
    st = env._state
    assert st is not None and st.wow_any and st.ground_speed_mps < 2.0 and abs(st.agl_ft) < 1.0
    assert st.along_m == pytest.approx(info["conditions"]["start_along_m"], abs=1.0)
    assert st.mean_n1 > 20.0   # engines running (idle)


def test_step_returns_gym_api_types(configs: dict) -> None:
    env = Boeing7478TakeoffEnv(configs=configs)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, float) and isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "reward_components" in info and obs.dtype == np.float32


def test_step_before_reset_raises(configs: dict) -> None:
    with pytest.raises(RuntimeError):
        Boeing7478TakeoffEnv(configs=configs).step(np.zeros(4, dtype=np.float32))


def test_same_seed_same_trajectory(configs: dict) -> None:
    traces = []
    for _ in range(2):
        env = Boeing7478TakeoffEnv(configs=configs)
        env.reset(seed=21)
        obs_list = [env.step(np.array([1.0, 0.0, 0.1, -0.1], dtype=np.float32))[0] for _ in range(60)]
        traces.append(np.array(obs_list))
    assert np.array_equal(traces[0], traces[1])


def test_different_seeds_give_different_conditions(configs: dict) -> None:
    rz = DomainRandomizer(configs["randomization"], configs["aircraft"])
    a, b = rz.sample(2, 1), rz.sample(2, 2)
    assert a.runway.length_m != b.runway.length_m and a.loading.payload_lbs != b.loading.payload_lbs
    assert rz.sample(2, 1) == a   # reproducible


def test_randomization_levels_are_ordered_in_difficulty(configs: dict) -> None:
    levels = configs["randomization"]["levels"]
    for key in ("crosswind_abs_kt", "gust_sigma_kt", "cg_shift_in", "runway_elevation_ft", "delta_isa_c"):
        widths = [levels[i][key][1] - levels[i][key][0] for i in (1, 2, 3, 4)]
        assert widths == sorted(widths) and widths[0] < widths[-1], key


def test_loading_respects_published_limits(configs: dict) -> None:
    rz = DomainRandomizer(configs["randomization"], configs["aircraft"])
    for level in (1, 2, 3, 4):
        for seed in range(40):
            load = rz.sample(level, seed).loading
            total = rz.oew_lbs + load.fuel_lbs + load.payload_lbs
            assert total <= rz.mtow_lbs + 1.0 and load.payload_lbs <= rz.max_payload_lbs + 1e-6
            assert 0 <= load.fuel_lbs <= rz.fuel_capacity_lbs + 1e-6


def test_teacher_achieves_valid_takeoff_and_reports_summary(configs: dict) -> None:
    env = Boeing7478TakeoffEnv(configs=configs)
    info = run(env, SimulatedTeacher(configs).act, seed=0)
    summary = info["episode_summary"]
    assert info["reason"] == "SUCCESS" and summary["success"] is True
    assert summary["final_agl_ft"] >= 100.0 and summary["duration_s"] < configs["environment"]["max_episode_seconds"]
    for key in ("centerline_rms_m", "centerline_max_m", "max_roll_deg", "max_pitch_deg", "max_alpha_deg",
                "runway_distance_used_m", "liftoff_time_s", "takeoff_time_s", "control_roughness"):
        assert np.isfinite(summary[key]), key


def test_timeout_truncates(configs: dict) -> None:
    cfg = copy.deepcopy(configs)
    cfg["environment"]["max_episode_seconds"] = 3.0
    info = run(Boeing7478TakeoffEnv(configs=cfg), lambda o: np.array([1.0, 0, 0, 0], dtype=np.float32), seed=0)
    assert info["reason"] == "TIMEOUT"


def test_set_level_changes_sampling(configs: dict) -> None:
    env = Boeing7478TakeoffEnv(configs=configs, level=1)
    env.set_level(3)
    env.reset(seed=0, options={})
    assert env.conditions is not None and env.conditions.level <= 3
    with pytest.raises(ValueError):
        env.set_level(9)


def test_gym_registration(configs: dict) -> None:
    register_env()
    env = gym.make(ENV_ID, configs=configs)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (27,)


def test_invalid_simulator_state_terminates_with_reason(configs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from envs.jsbsim_interface import InvalidSimulatorStateError
    env = Boeing7478TakeoffEnv(configs=configs)
    env.reset(seed=0)

    def boom(_n: int) -> None:
        raise InvalidSimulatorStateError("injected NaN")

    monkeypatch.setattr(env.sim, "step", boom)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert terminated and not truncated and info["reason"] == "SIMULATOR_INVALID_STATE" and reward < 0
    assert np.all(np.isfinite(obs)) and info["episode_summary"]["reason"] == "SIMULATOR_INVALID_STATE"


def test_max_alpha_ignores_parked_in_wind_phase(configs: dict) -> None:
    """AoA is meaningless below flying speed; the episode statistic must not record it (regression)."""
    env = Boeing7478TakeoffEnv(configs=configs)
    info = run(env, SimulatedTeacher(configs).act, seed=0)
    assert info["episode_summary"]["max_alpha_deg"] < configs["environment"]["success"]["max_alpha_deg"] + 5.0


def test_episode_summary_reports_cleanliness_and_teacher_is_clean(configs: dict) -> None:
    env = Boeing7478TakeoffEnv(configs=configs)
    summary = run(env, SimulatedTeacher(configs).act, seed=3)["episode_summary"]
    assert summary["success"] and 0.9 < summary["cleanliness"] <= 1.0 and summary["reward_profile"] == "clean"
    assert summary["rc_clean_takeoff_bonus"] > 100.0 and summary["rms_lateral_accel_mps2"] < 0.2


def test_wobbly_policy_gets_low_cleanliness_even_if_it_succeeds(configs: dict) -> None:
    """Weaving the rudder while accelerating hurts the clean score and the return (same seed, same conditions)."""
    env = Boeing7478TakeoffEnv(configs=configs)
    teacher = SimulatedTeacher(configs)
    clean = run(env, teacher.act, seed=3)["episode_summary"]
    state = {"k": 0}

    def weaving(obs):      # full throttle + open-loop rudder weave (no feedback), period ~4 s: the plane snakes left/right
        state["k"] += 1
        return np.array([1.0, 0.0, 0.0, 0.6 * np.sin(state["k"] * 0.15)], dtype=np.float32)

    wobbly = run(env, weaving, seed=3)["episode_summary"]
    assert wobbly["cleanliness"] < clean["cleanliness"] - 0.1
    assert wobbly["total_reward"] < clean["total_reward"]
