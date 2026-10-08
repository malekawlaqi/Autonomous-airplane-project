"""Derived flight state and the normalised observation vector."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from envs.runway import FPS_PER_KT, Runway, wrap_angle_rad

OBS_NAMES: tuple[str, ...] = (
    "airspeed", "ground_speed", "agl", "vertical_speed", "pitch", "roll", "relative_heading",
    "alpha", "beta", "p", "q", "r", "lateral_error", "course_error", "distance_travelled",
    "distance_remaining", "weight_on_wheels", "elevator_pos", "aileron_pos", "rudder_pos", "throttle",
    "engine_n1", "mass_norm", "cg_norm", "headwind", "crosswind", "density_ratio",
)
OBS_DIM: int = len(OBS_NAMES)
KT_PER_MPS: float = 1.0 / 0.514444


@dataclass(frozen=True)
class FlightState:
    """Quantities derived from raw JSBSim sensors, expressed relative to the runway.

    Units: knots (CAS) for ``airspeed_kt``/``ground_speed_kt``, feet for ``agl_ft`` (above the
    wheel-ground contact height), degrees for angles, metres for runway coordinates.
    """

    time_s: float
    along_m: float
    lateral_m: float
    remaining_m: float
    agl_ft: float
    airspeed_kt: float
    ground_speed_kt: float
    ground_speed_mps: float
    vertical_speed_fpm: float
    pitch_deg: float
    roll_deg: float
    heading_error_deg: float
    course_error_deg: float
    alpha_deg: float
    beta_deg: float
    p_rad_s: float
    q_rad_s: float
    r_rad_s: float
    wow_nose: bool
    wow_main: bool
    wow_any: bool
    headwind_kt: float
    crosswind_kt: float
    density_ratio: float
    mass_lbs: float
    cg_x_in: float
    mean_n1: float
    vs_est_kt: float
    vr_est_kt: float
    v2_est_kt: float

    @property
    def lateral_velocity_mps(self) -> float:
        """Sideways ground velocity in the runway frame (+ = drifting right), from the true velocity vector."""
        return self.ground_speed_mps * math.sin(math.radians(self.course_error_deg))


def stall_speed_estimate_kcas(weight_lbs: float, wing_area_ft2: float, clmax: float,
                              rho_sl_slugs_ft3: float) -> float:
    """Estimate 1-g stall speed in knots CAS from ``Vs = sqrt(2 W / (rho_sl S CLmax))``.

    This is a simple screening estimate used for success/reward logic. It is NOT Boeing
    performance data.
    """
    vs_fps = math.sqrt(2.0 * weight_lbs / (rho_sl_slugs_ft3 * wing_area_ft2 * clmax))
    return vs_fps / FPS_PER_KT


def compute_flight_state(sensors: dict[str, float], runway: Runway, along_m: float, lateral_m: float,
                         ground_agl_ref_ft: float, aircraft_cfg: dict[str, Any]) -> FlightState:
    """Combine raw sensors, runway geometry and aircraft constants into a :class:`FlightState`."""
    env = aircraft_cfg["envelope"]
    wing_area = float(aircraft_cfg["overrides"]["wingarea_ft2"]["value"])
    vn, ve = sensors["v_north_fps"] * 0.3048, sensors["v_east_fps"] * 0.3048
    v_along, v_lat = runway.ned_to_runway(vn, ve)
    ground_speed_mps = math.hypot(vn, ve)
    course_err = math.atan2(v_lat, v_along) if ground_speed_mps > 2.0 else 0.0
    head, cross = runway.wind_components(sensors["wind_north_fps"] * 0.3048, sensors["wind_east_fps"] * 0.3048)
    vs_est = stall_speed_estimate_kcas(sensors["weight_lbs"], wing_area, float(env["clmax_est"]),
                                       sensors["rho_sl_slugs_ft3"])
    wow_nose, wow_l, wow_r = (sensors["wow_nose"] > 0.5, sensors["wow_left"] > 0.5, sensors["wow_right"] > 0.5)
    return FlightState(
        time_s=0.0, along_m=along_m, lateral_m=lateral_m, remaining_m=runway.remaining_m(along_m),
        agl_ft=sensors["h_agl_ft"] - ground_agl_ref_ft, airspeed_kt=sensors["vc_kts"],
        ground_speed_kt=ground_speed_mps * KT_PER_MPS, ground_speed_mps=ground_speed_mps,
        vertical_speed_fpm=sensors["hdot_fps"] * 60.0, pitch_deg=math.degrees(sensors["theta_rad"]),
        roll_deg=math.degrees(sensors["phi_rad"]),
        heading_error_deg=math.degrees(wrap_angle_rad(sensors["psi_rad"] - runway.heading_rad)),
        course_error_deg=math.degrees(course_err), alpha_deg=math.degrees(sensors["alpha_rad"]),
        beta_deg=math.degrees(sensors["beta_rad"]), p_rad_s=sensors["p_rad_s"], q_rad_s=sensors["q_rad_s"],
        r_rad_s=sensors["r_rad_s"], wow_nose=wow_nose, wow_main=wow_l or wow_r, wow_any=wow_nose or wow_l or wow_r,
        headwind_kt=head * KT_PER_MPS, crosswind_kt=cross * KT_PER_MPS,
        density_ratio=sensors["rho_slugs_ft3"] / sensors["rho_sl_slugs_ft3"], mass_lbs=sensors["weight_lbs"],
        cg_x_in=sensors["cg_x_in"],
        mean_n1=0.25 * (sensors["n1_0"] + sensors["n1_1"] + sensors["n1_2"] + sensors["n1_3"]),
        vs_est_kt=vs_est, vr_est_kt=vs_est * float(env["vr_factor_of_vs"]),
        v2_est_kt=vs_est * float(env["v2_factor_of_vs"]))


class ObservationBuilder:
    """Builds the normalised, clipped, finite observation vector."""

    def __init__(self, env_cfg: dict[str, Any], aircraft_cfg: dict[str, Any]) -> None:
        """Create the builder from ``environment.yaml`` and ``aircraft.yaml``."""
        obs = env_cfg["observation"]
        self.scales: dict[str, float] = {k: float(v) for k, v in obs["scales"].items()}
        self.clip = float(obs["clip"])
        self.oew = float(aircraft_cfg["overrides"]["empty_weight_lbs"]["value"])
        self.mtow = float(aircraft_cfg["limits"]["mtow_lbs"])
        self.nominal_cg = float(aircraft_cfg["limits"]["nominal_cg_x_in"])

    def build(self, st: FlightState, actuators: np.ndarray) -> np.ndarray:
        """Return the observation for a flight state and actuator vector ``[thr01, elev, ail, rud]``.

        Raises:
            ValueError: if the resulting vector is not finite (the environment maps this to
                SIMULATOR_INVALID_STATE).
        """
        s = self.scales
        values = (
            st.airspeed_kt / s["airspeed_kt"], st.ground_speed_kt / s["ground_speed_kt"], st.agl_ft / s["agl_ft"],
            st.vertical_speed_fpm / s["vertical_speed_fpm"], math.radians(st.pitch_deg) / s["pitch_rad"],
            math.radians(st.roll_deg) / s["roll_rad"], math.radians(st.heading_error_deg) / s["relative_heading_rad"],
            math.radians(st.alpha_deg) / s["alpha_rad"], math.radians(st.beta_deg) / s["beta_rad"],
            st.p_rad_s / s["p_rad_s"], st.q_rad_s / s["q_rad_s"], st.r_rad_s / s["r_rad_s"],
            st.lateral_m / s["lateral_error_m"], math.radians(st.course_error_deg) / s["course_error_rad"],
            st.along_m / s["distance_travelled_m"], st.remaining_m / s["distance_remaining_m"],
            1.0 if st.wow_any else 0.0, actuators[1], actuators[2], actuators[3], actuators[0],
            st.mean_n1 / 100.0, (st.mass_lbs - self.oew) / (self.mtow - self.oew),
            (st.cg_x_in - self.nominal_cg) / s["cg_shift_in"], st.headwind_kt / s["wind_kt"],
            st.crosswind_kt / s["wind_kt"],
            (st.density_ratio - s["density_ratio_center"]) / s["density_ratio_scale"],
        )
        vec = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(vec)):
            bad = [OBS_NAMES[i] for i in np.flatnonzero(~np.isfinite(vec))]
            raise ValueError(f"Non-finite observation entries: {bad}")
        return np.clip(vec, -self.clip, self.clip).astype(np.float32)
