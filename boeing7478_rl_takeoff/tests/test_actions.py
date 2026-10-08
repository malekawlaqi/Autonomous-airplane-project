"""Tests for the action mapping (clipping, throttle conversion, rate limits, JSBSim sign conventions)."""
from __future__ import annotations

import numpy as np
import pytest

from envs import load_all_configs
from envs.actions import ACTION_DIM, ActionMapper


@pytest.fixture()
def mapper() -> ActionMapper:
    cfg = load_all_configs()
    m = ActionMapper(cfg["aircraft"], cfg["environment"]["policy_hz"])
    m.reset()
    return m


def test_action_dim_is_four() -> None:
    assert ACTION_DIM == 4


def test_throttle_conversion_range(mapper: ActionMapper) -> None:
    low = mapper.target_from_action(np.array([-1.0, 0, 0, 0]))
    high = mapper.target_from_action(np.array([1.0, 0, 0, 0]))
    mid = mapper.target_from_action(np.array([0.0, 0, 0, 0]))
    assert (low[0], mid[0], high[0]) == (0.0, 0.5, 1.0)


def test_out_of_range_actions_are_clipped(mapper: ActionMapper) -> None:
    clipped = mapper.validate(np.array([9.0, -9.0, 9.0, -9.0]))
    assert np.all(np.abs(clipped) <= 1.0)


def test_nan_and_wrong_shape_rejected(mapper: ActionMapper) -> None:
    with pytest.raises(ValueError):
        mapper.apply(np.array([np.nan, 0, 0, 0]))
    with pytest.raises(ValueError):
        mapper.apply(np.zeros(3))


def test_rate_limits_prevent_instant_switching(mapper: ActionMapper) -> None:
    mapper.apply(np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.all(np.abs(mapper.state - np.array([0.0, 0, 0, 0])) <= mapper.max_step + 1e-12)
    previous = mapper.state.copy()
    mapper.apply(np.array([-1.0, -1.0, -1.0, -1.0]))
    assert np.all(np.abs(mapper.state - previous) <= mapper.max_step + 1e-12)
    assert np.all(np.abs(mapper.last_step_norm) <= 1.0 + 1e-12)


def test_surfaces_reach_commanded_value_eventually(mapper: ActionMapper) -> None:
    for _ in range(100):
        mapper.apply(np.array([1.0, 0.5, -0.5, 0.25]))
    assert np.allclose(mapper.state, [1.0, 0.5, -0.5, 0.25])


def test_jsbsim_sign_conventions(mapper: ActionMapper) -> None:
    """Verified conventions: elevator and rudder signs are inverted relative to the agent convention."""
    for _ in range(100):
        mapper.apply(np.array([0.0, 1.0, 1.0, 1.0]))
    cmd = mapper.jsbsim_commands()
    assert cmd.elevator_cmd < 0.0   # agent +elevator (nose-up) = JSBSim negative command
    assert cmd.aileron > 0.0        # agent +aileron (roll right) = JSBSim positive command
    assert cmd.rudder < 0.0         # agent +rudder (yaw right) = JSBSim negative command
    assert cmd.steer > 0.0          # nose-wheel steering follows the rudder pedal direction


def test_reset_returns_to_idle(mapper: ActionMapper) -> None:
    for _ in range(50):
        mapper.apply(np.ones(4))
    mapper.reset()
    assert np.all(mapper.state == 0.0)
