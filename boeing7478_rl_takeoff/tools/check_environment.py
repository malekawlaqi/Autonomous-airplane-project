"""Pre-training validation gate.  Run this and fix every failure before training.

    python tools/check_environment.py            # full check
    python tools/check_environment.py --quick    # skip the slow checks (1000 random actions, gym check_env)

Checks: JSBSim loads, aircraft loads, reset works, observation/action shapes, observation bounds,
actions bounded, no NaN/inf, 1000 random actions don't crash, deterministic seeding, reward
components are numbers, termination (success / failure / truncation) works, gymnasium check_env.
"""
from __future__ import annotations

import argparse
import copy
import sys
import traceback
import warnings
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from envs import load_all_configs  # noqa: E402

Result = tuple[bool, str]


def run_episode(env, policy: Callable[[np.ndarray], np.ndarray], seed: int, level: int = 1, max_steps: int = 3000) -> dict:
    """Run one episode with ``policy`` and return the final info dict."""
    obs, _ = env.reset(seed=seed, options={"level": level})
    info: dict = {}
    for _ in range(max_steps):
        obs, _r, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            info["terminated"], info["truncated"] = terminated, truncated
            return info
    raise AssertionError(f"episode did not end within {max_steps} steps")


def check_jsbsim_loads(configs) -> Result:
    """JSBSim imports and reports a version."""
    import jsbsim
    return True, f"JSBSim {jsbsim.__version__}"


def check_aircraft_loads(configs) -> Result:
    """The derived 747-8 research-approximation FDM loads and every registered property exists."""
    from envs import PROJECT_ROOT
    from envs.jsbsim_interface import SENSOR_PROPERTIES, JSBSimInterface
    from envs.randomization import DomainRandomizer
    iface = JSBSimInterface(configs["aircraft"], configs["environment"]["physics_hz"], PROJECT_ROOT)
    cond = DomainRandomizer(configs["randomization"], configs["aircraft"]).sample(1, 0)
    meta = iface.start_episode(cond)  # raises if the aircraft fails to load or any property is missing
    return True, (f"{configs['aircraft']['model_status']}: {iface.built.model_name}, weight {meta['weight_lbs']:.0f} lb, "
                  f"{len(SENSOR_PROPERTIES)} sensor properties verified")


def make_env(configs):
    """Create a fresh environment."""
    from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv
    return Boeing7478TakeoffEnv(configs=configs)


def check_reset_shapes(configs) -> Result:
    """reset() works; observation/action shapes and dtypes are correct; physics-per-action is 12."""
    env = make_env(configs)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (27,), obs.shape
    assert obs.dtype == np.float32
    assert env.action_space.shape == (4,) and env.action_space.dtype == np.float32
    assert env.action_space.low.min() == -1.0 and env.action_space.high.max() == 1.0
    assert env.physics_steps_per_action == round(configs["environment"]["physics_hz"] / configs["environment"]["policy_hz"])
    return True, f"obs {obs.shape}, action {env.action_space.shape}, physics steps/action {env.physics_steps_per_action}"


def check_observation_bounds(configs) -> Result:
    """Observations lie inside the declared space and are finite over a teacher episode."""
    from teacher.simulated_teacher import SimulatedTeacher
    env = make_env(configs)
    teacher = SimulatedTeacher(configs)
    obs, _ = env.reset(seed=3)
    for _ in range(600):
        assert env.observation_space.contains(obs), "observation outside space"
        assert np.all(np.isfinite(obs)), "non-finite observation"
        obs, _r, te, tr, _i = env.step(teacher.act(obs))
        if te or tr:
            break
    return True, "observations finite and inside Box bounds"


def check_actions_bounded(configs) -> Result:
    """Out-of-range actions are clipped; NaN actions are rejected loudly; rate limits hold."""
    env = make_env(configs)
    env.reset(seed=0)
    env.step(np.array([50.0, -50.0, 50.0, -50.0], dtype=np.float32))
    act = env.mapper.state
    assert -1.0 <= act[1] <= 1.0 and 0.0 <= act[0] <= 1.0 and abs(act[2]) <= 1.0 and abs(act[3]) <= 1.0
    max_step = env.mapper.max_step
    assert np.all(np.abs(act[1:]) <= max_step[1:] + 1e-9), "rate limit exceeded on first step"
    try:
        env.step(np.array([np.nan, 0, 0, 0], dtype=np.float32))
    except ValueError:
        return True, "actions clipped to [-1,1], rate-limited, NaN rejected"
    raise AssertionError("NaN action was not rejected")


def check_random_actions(configs) -> Result:
    """1000 random actions run without a Python crash and without NaN/inf output."""
    env = make_env(configs)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=1)
    resets = 0
    for _ in range(1000):
        obs, reward, te, tr, info = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if te or tr:
            resets += 1
            obs, _ = env.reset()
    return True, f"1000 random actions OK ({resets} episode ends)"


def check_determinism(configs) -> Result:
    """Same seed + same actions -> identical observations, rewards and conditions."""
    runs = []
    for _ in range(2):
        env = make_env(configs)
        obs, info = env.reset(seed=11)
        trace = [obs.copy()]
        rewards = []
        rng = np.random.default_rng(5)
        for _i in range(150):
            obs, r, te, tr, _ = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
            trace.append(obs.copy())
            rewards.append(r)
            if te or tr:
                break
        runs.append((np.array(trace), np.array(rewards), info["episode_seed"]))
    assert runs[0][2] == runs[1][2], "episode seeds differ"
    assert np.array_equal(runs[0][0], runs[1][0]), "observation traces differ for identical seed/actions"
    assert np.array_equal(runs[0][1], runs[1][1]), "rewards differ for identical seed/actions"
    return True, "identical traces for identical seed and actions"


def check_reward_components(configs) -> Result:
    """Every reward component is a finite float and the total equals their sum."""
    from envs.reward import COMPONENT_NAMES
    env = make_env(configs)
    env.reset(seed=2)
    for _ in range(50):
        _o, reward, te, tr, info = env.step(np.array([1.0, 0, 0, 0], dtype=np.float32))
        comps = info["reward_components"]
        assert set(comps) == set(COMPONENT_NAMES), "component names mismatch"
        assert all(isinstance(v, float) and np.isfinite(v) for v in comps.values())
        assert abs(sum(comps.values()) - reward) < 1e-5
    return True, f"{len(COMPONENT_NAMES)} components, all finite, sum == reward"


def check_termination(configs) -> Result:
    """Success terminates; failures terminate; the time limit truncates (not terminates)."""
    from teacher.simulated_teacher import SimulatedTeacher
    messages = []
    env = make_env(configs)
    info = run_episode(env, SimulatedTeacher(configs).act, seed=0)
    assert info["terminated"] and not info["truncated"] and info["reason"] == "SUCCESS", info["reason"]
    assert info["episode_summary"]["success"] is True
    messages.append("teacher->SUCCESS(terminated)")

    info = run_episode(env, lambda o: np.array([-1.0, 0, 0, 0], dtype=np.float32), seed=0)
    assert info["terminated"] and info["reason"] == "FAILED_TO_ACCELERATE", info["reason"]
    messages.append("idle->FAILED_TO_ACCELERATE(terminated)")

    info = run_episode(env, lambda o: np.array([1.0, 0, 0, 1.0], dtype=np.float32), seed=0)
    assert info["terminated"] and info["reason"] not in ("SUCCESS", "TIMEOUT"), info["reason"]
    messages.append(f"full-right-rudder->{info['reason']}(terminated)")

    short = copy.deepcopy(configs)
    short["environment"]["max_episode_seconds"] = 5.0
    info = run_episode(make_env(short), lambda o: np.array([1.0, 0, 0, 0], dtype=np.float32), seed=0)
    assert info["truncated"] and not info["terminated"] and info["reason"] == "TIMEOUT", info["reason"]
    messages.append("5 s limit->TIMEOUT(truncated)")
    return True, "; ".join(messages)


def check_gym_env_checker(configs) -> Result:
    """gymnasium.utils.env_checker.check_env passes."""
    from gymnasium.utils.env_checker import check_env
    env = make_env(configs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    texts = sorted({str(w.message)[:90] for w in caught})
    return True, f"check_env passed ({len(texts)} distinct warnings: {texts})"


def main() -> int:
    """Run all checks, print a table and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip slow checks")
    args = parser.parse_args()
    configs = load_all_configs()
    checks: list[tuple[str, Callable]] = [
        ("JSBSim loads", check_jsbsim_loads), ("Aircraft loads", check_aircraft_loads),
        ("Reset / shapes / 12 physics steps per action", check_reset_shapes),
        ("Observation bounds, no NaN/inf", check_observation_bounds), ("Actions bounded / rate-limited", check_actions_bounded),
        ("Deterministic seed", check_determinism), ("Reward components are numbers", check_reward_components),
        ("Termination / truncation", check_termination),
    ]
    if not args.quick:
        checks += [("1000 random actions", check_random_actions), ("gymnasium check_env", check_gym_env_checker)]
    failed = 0
    for name, fn in checks:
        try:
            ok, detail = fn(configs)
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        except Exception:  # noqa: BLE001 - report with full traceback, then count as failure
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
