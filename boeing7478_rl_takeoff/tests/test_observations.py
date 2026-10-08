"""Tests for runway geometry and the observation vector."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from envs import load_all_configs
from envs.observations import OBS_DIM, OBS_NAMES, ObservationBuilder, compute_flight_state, stall_speed_estimate_kcas
from envs.runway import Runway, wrap_angle_deg


def test_obs_names_match_dimension() -> None:
    assert OBS_DIM == len(OBS_NAMES) == 27
    assert len(set(OBS_NAMES)) == OBS_DIM
    for required in ("airspeed", "ground_speed", "agl", "vertical_speed", "pitch", "roll", "alpha", "beta", "p", "q", "r",
                     "lateral_error", "distance_travelled", "distance_remaining", "weight_on_wheels", "throttle",
                     "headwind", "crosswind", "density_ratio"):
        assert required in OBS_NAMES


def test_runway_frame_round_trip() -> None:
    rw = Runway(length_m=4000, width_m=60, heading_deg=123.0, elevation_ft=0)
    n, e = rw.runway_to_ned(250.0, -7.0)
    along, lateral = rw.ned_to_runway(n, e)
    assert along == pytest.approx(250.0) and lateral == pytest.approx(-7.0)


def test_runway_east_heading_geometry() -> None:
    rw = Runway(length_m=4000, width_m=60, heading_deg=90.0, elevation_ft=0)
    along, lateral = rw.ned_to_runway(north_m=10.0, east_m=100.0)   # heading east: north is to the LEFT
    assert along == pytest.approx(100.0) and lateral == pytest.approx(-10.0)


def test_wind_components_sign() -> None:
    rw = Runway(length_m=4000, width_m=60, heading_deg=0.0, elevation_ft=0)
    head, cross = rw.wind_components(wind_north_mps=-10.0, wind_east_mps=0.0)  # air moving south, runway heading north
    assert head == pytest.approx(10.0) and cross == pytest.approx(0.0)         # pure headwind
    n, e = rw.wind_ned_from_components(5.0, 3.0)
    assert rw.wind_components(n, e) == pytest.approx((5.0, 3.0))


def test_wrap_angle() -> None:
    assert wrap_angle_deg(190.0) == pytest.approx(-170.0)
    assert wrap_angle_deg(-190.0) == pytest.approx(170.0)


def test_stall_estimate_physical_scale() -> None:
    vs = stall_speed_estimate_kcas(750_000, 5960.0, 2.2, 0.0023769)
    assert 100.0 < vs < 160.0


def _sensors() -> dict[str, float]:
    keys = ["vc_kts", "vg_fps", "v_north_fps", "v_east_fps", "h_agl_ft", "hdot_fps", "phi_rad", "theta_rad", "psi_rad",
            "alpha_rad", "beta_rad", "p_rad_s", "q_rad_s", "r_rad_s", "wow_nose", "wow_left", "wow_right", "n1_0",
            "n1_1", "n1_2", "n1_3", "weight_lbs", "cg_x_in", "rho_slugs_ft3", "rho_sl_slugs_ft3", "wind_north_fps",
            "wind_east_fps"]
    s = {k: 0.0 for k in keys}
    s.update({"vc_kts": 100.0, "v_north_fps": 150.0, "h_agl_ft": 16.0, "weight_lbs": 750_000.0, "cg_x_in": 1327.0,
              "rho_slugs_ft3": 0.002, "rho_sl_slugs_ft3": 0.0023769, "wow_left": 1.0, "wow_right": 1.0,
              "wow_nose": 1.0, "n1_0": 90.0, "n1_1": 90.0, "n1_2": 90.0, "n1_3": 90.0})
    return s


def test_observation_vector_finite_clipped_and_float32() -> None:
    cfg = load_all_configs()
    rw = Runway(length_m=4000, width_m=60, heading_deg=0.0, elevation_ft=0)
    st = compute_flight_state(_sensors(), rw, 500.0, 2.0, 16.0, cfg["aircraft"])
    builder = ObservationBuilder(cfg["environment"], cfg["aircraft"])
    obs = builder.build(st, np.array([1.0, 0.0, 0.0, 0.0]))
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert np.all(np.abs(obs) <= cfg["environment"]["observation"]["clip"])
    assert st.remaining_m == pytest.approx(3500.0) and st.wow_any and st.lateral_m == 2.0


def test_extreme_values_are_clipped() -> None:
    cfg = load_all_configs()
    rw = Runway(length_m=4000, width_m=60, heading_deg=0.0, elevation_ft=0)
    s = _sensors()
    s["hdot_fps"] = 1e6
    st = compute_flight_state(s, rw, 500.0, 2.0, 16.0, cfg["aircraft"])
    obs = ObservationBuilder(cfg["environment"], cfg["aircraft"]).build(st, np.zeros(4))
    assert np.max(obs) == pytest.approx(cfg["environment"]["observation"]["clip"])


def test_non_finite_state_raises() -> None:
    cfg = load_all_configs()
    rw = Runway(length_m=4000, width_m=60, heading_deg=0.0, elevation_ft=0)
    st = compute_flight_state(_sensors(), rw, 500.0, 2.0, 16.0, cfg["aircraft"])
    builder = ObservationBuilder(cfg["environment"], cfg["aircraft"])
    for field, value in (("pitch_deg", math.nan), ("airspeed_kt", math.inf), ("agl_ft", -math.inf)):
        with pytest.raises(ValueError):
            builder.build(dataclasses.replace(st, **{field: value}), np.zeros(4))
