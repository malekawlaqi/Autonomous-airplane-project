"""Thin, strict wrapper around JSBSim plus the generator of the 747-8 research-approximation FDM.

Design rules (see PHASE 23 of the project brief):
* every JSBSim property used by the project is listed in ``SENSOR_PROPERTIES`` /
  ``COMMAND_PROPERTIES`` / ``SETUP_PROPERTIES`` and verified at load time;
* a missing property raises :class:`PropertyNotFoundError` with similar names - no guessing;
* non-finite simulator output raises :class:`InvalidSimulatorStateError`.
"""
from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsbsim

from envs.randomization import EpisodeConditions
from envs.runway import FPS_PER_KT

# Silence the startup banner / info chatter (a new executive is created every episode) but keep
# warnings-as-errors visible: ERROR and FATAL messages are still printed.
jsbsim.get_logger().set_min_level(jsbsim.LogLevel.ERROR)

# --------------------------------------------------------------------------------------
# Property registry.  logical-name -> JSBSim property.  Verified in verify_properties().
# --------------------------------------------------------------------------------------
SENSOR_PROPERTIES: dict[str, str] = {
    "sim_time_s": "simulation/sim-time-sec",
    "vc_kts": "velocities/vc-kts",
    "vtrue_kts": "velocities/vtrue-kts",
    "vg_fps": "velocities/vg-fps",
    "v_north_fps": "velocities/v-north-fps",
    "v_east_fps": "velocities/v-east-fps",
    "h_agl_ft": "position/h-agl-ft",
    "h_sl_ft": "position/h-sl-ft",
    "hdot_fps": "velocities/h-dot-fps",
    "phi_rad": "attitude/phi-rad",
    "theta_rad": "attitude/theta-rad",
    "psi_rad": "attitude/psi-rad",
    "alpha_rad": "aero/alpha-rad",
    "beta_rad": "aero/beta-rad",
    "p_rad_s": "velocities/p-rad_sec",
    "q_rad_s": "velocities/q-rad_sec",
    "r_rad_s": "velocities/r-rad_sec",
    "lat_deg": "position/lat-gc-deg",
    "lon_deg": "position/long-gc-deg",
    "dist_north_m": "position/distance-from-start-lat-mt",
    "dist_east_m": "position/distance-from-start-lon-mt",
    "wow_nose": "gear/unit[0]/WOW",
    "wow_left": "gear/unit[1]/WOW",
    "wow_right": "gear/unit[2]/WOW",
    "elevator_pos_rad": "fcs/elevator-pos-rad",
    "aileron_pos_rad": "fcs/left-aileron-pos-rad",
    "rudder_pos_rad": "fcs/rudder-pos-rad",
    "steer_pos_deg": "fcs/steer-pos-deg",
    "flap_pos_deg": "fcs/flap-pos-deg",
    "n1_0": "propulsion/engine[0]/n1",
    "n1_1": "propulsion/engine[1]/n1",
    "n1_2": "propulsion/engine[2]/n1",
    "n1_3": "propulsion/engine[3]/n1",
    "thrust_0_lbs": "propulsion/engine[0]/thrust-lbs",
    "weight_lbs": "inertia/weight-lbs",
    "cg_x_in": "inertia/cg-x-in",
    "rho_slugs_ft3": "atmosphere/rho-slugs_ft3",
    "rho_sl_slugs_ft3": "atmosphere/rho-sl-slugs_ft3",
    "temperature_r": "atmosphere/T-R",
    "pressure_psf": "atmosphere/P-psf",
    "wind_north_fps": "atmosphere/total-wind-north-fps",
    "wind_east_fps": "atmosphere/total-wind-east-fps",
    "wind_down_fps": "atmosphere/total-wind-down-fps",
    "nz_g": "accelerations/Nz",
}

COMMAND_PROPERTIES: dict[str, str] = {
    "elevator_cmd": "fcs/elevator-cmd-norm",
    "aileron_cmd": "fcs/aileron-cmd-norm",
    "rudder_cmd": "fcs/rudder-cmd-norm",
    "steer_cmd": "fcs/steer-cmd-norm",
    "throttle_cmd_0": "fcs/throttle-cmd-norm[0]",
    "throttle_cmd_1": "fcs/throttle-cmd-norm[1]",
    "throttle_cmd_2": "fcs/throttle-cmd-norm[2]",
    "throttle_cmd_3": "fcs/throttle-cmd-norm[3]",
}

SETUP_PROPERTIES: dict[str, str] = {
    "pitch_trim_cmd": "fcs/pitch-trim-cmd-norm",
    "flap_cmd": "fcs/flap-cmd-norm",
    "gear_cmd": "gear/gear-cmd-norm",
    "speedbrake_cmd": "fcs/speedbrake-cmd-norm",
    "brake_left": "fcs/left-brake-cmd-norm",
    "brake_right": "fcs/right-brake-cmd-norm",
    "brake_center": "fcs/center-brake-cmd-norm",
    "set_running": "propulsion/set-running",
    "delta_t": "atmosphere/delta-T",
    "p_sl_psf": "atmosphere/P-sl-psf",
    "wind_north": "atmosphere/wind-north-fps",
    "wind_east": "atmosphere/wind-east-fps",
    "wind_down": "atmosphere/wind-down-fps",
    "turb_type": "atmosphere/turb-type",
    "turb_severity": "atmosphere/turbulence/milspec/severity",
    "turb_wind20": "atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps",
    "random_seed": "simulation/randomseed",
    "ground_static_factor": "ground/static-friction-factor",
    "ground_rolling_factor": "ground/rolling_friction-factor",
    "ic_terrain_ft": "ic/terrain-elevation-ft",
    "ic_h_agl_ft": "ic/h-agl-ft",
    "ic_lat": "ic/lat-gc-deg",
    "ic_lon": "ic/long-gc-deg",
    "ic_psi": "ic/psi-true-deg",
    "ic_theta": "ic/theta-deg",
    "ic_phi": "ic/phi-deg",
    "ic_u": "ic/u-fps",
    "ic_v": "ic/v-fps",
    "ic_w": "ic/w-fps",
    "payload_weight": "inertia/pointmass-weight-lbs[0]",
    "payload_x": "inertia/pointmass-location-X-inches[0]",
    "tank_0": "propulsion/tank[0]/contents-lbs",
    "tank_1": "propulsion/tank[1]/contents-lbs",
    "tank_2": "propulsion/tank[2]/contents-lbs",
    "tank_3": "propulsion/tank[3]/contents-lbs",
    "tank_4": "propulsion/tank[4]/contents-lbs",
}

NUM_TANKS: int = 5
NUM_ENGINES: int = 4
PAYLOAD_POINTMASS_NAME: str = "PAYLOAD"


class JSBSimError(RuntimeError):
    """Base class for errors raised by the JSBSim interface."""


class PropertyNotFoundError(JSBSimError):
    """A required JSBSim property does not exist."""


class InvalidSimulatorStateError(JSBSimError):
    """The simulator returned a non-finite value or stopped running."""


@dataclass(frozen=True)
class BuiltAircraft:
    """Paths to the generated research-approximation aircraft."""

    model_name: str
    aircraft_path: Path
    engine_path: Path
    systems_path: Path
    xml_path: Path
    engine_name: str


# --------------------------------------------------------------------------------------
# Derived aircraft generation
# --------------------------------------------------------------------------------------
def find_jsbsim_root() -> Path:
    """Locate the JSBSim data root (the folder holding ``aircraft/``, ``engine/``, ``systems/``).

    Raises:
        FileNotFoundError: if the installed package carries no aircraft data.
    """
    root = Path(jsbsim.get_default_root_dir())
    if not (root / "aircraft").is_dir():
        raise FileNotFoundError(f"JSBSim aircraft data not found under {root}")
    return root


def _set_text(root: ET.Element, path: str, value: Any) -> None:
    """Set the text of a required XML child; raise if the element does not exist."""
    element = root.find(path)
    if element is None:
        raise KeyError(f"Expected element '{path}' not found in base FDM XML")
    element.text = f" {value} "


def _write_xml(root: ET.Element, path: Path, comment: str) -> None:
    """Write an XML tree with a leading comment."""
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ET.tostring(root, encoding="unicode")
    path.write_text(f'<?xml version="1.0"?>\n<!--\n{comment}\n-->\n{body}\n', encoding="utf-8")


def build_derived_aircraft(aircraft_cfg: dict[str, Any], project_root: Path) -> BuiltAircraft:
    """Generate the 747-8 research-approximation FDM from the bundled JSBSim 747-400 FDM.

    Only the overrides listed in ``aircraft.yaml`` are applied; everything else is inherited
    unchanged from the 747-400 model (see AIRCRAFT_MODEL_NOTES.md).

    Args:
        aircraft_cfg: Parsed ``aircraft.yaml``.
        project_root: Project root directory.

    Returns:
        Paths of the generated aircraft/engine files.
    """
    jsb_root = find_jsbsim_root()
    base_dir = aircraft_cfg["base_model"]["jsbsim_aircraft_dir"]
    base_xml = jsb_root / "aircraft" / base_dir / f"{base_dir}.xml"
    if not base_xml.is_file():
        raise FileNotFoundError(f"Base FDM not found: {base_xml}")
    ov = aircraft_cfg["overrides"]
    name = aircraft_cfg["derived_model_name"]
    engine_name = "GEnx-2B67_RA"
    build_dir = (project_root / aircraft_cfg["build_dir"]).resolve()
    ac_dir = build_dir / "aircraft" / name
    eng_dir = build_dir / "engine"
    eng_dir.mkdir(parents=True, exist_ok=True)
    ac_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- aircraft FDM ----------------
    tree = ET.parse(base_xml)
    root = tree.getroot()
    root.set("name", name)
    header = root.find("fileheader")
    if header is None:
        raise KeyError("Base FDM has no <fileheader>")
    for child in list(header):
        if child.tag in ("description", "note", "author"):
            header.remove(child)
    ET.SubElement(header, "author").text = "Derived from JSBSim B747 (747-400) by boeing7478_rl_takeoff"
    ET.SubElement(header, "description").text = "7478_RESEARCH_APPROXIMATION - NOT a validated Boeing 747-8 model"
    ET.SubElement(header, "note").text = (
        "Aerodynamics, inertia, gear and engine lapse tables are inherited from the JSBSim 747-400 FDM. "
        "Only wing span/area, empty weight, fuel capacity and engine thrust/bypass ratio are set to published "
        "747-8 values. Educational/research use only; not endorsed by the manufacturer.")
    _set_text(root, "metrics/wingspan", ov["wingspan_ft"]["value"])
    _set_text(root, "metrics/wingarea", ov["wingarea_ft2"]["value"])
    _set_text(root, "mass_balance/emptywt", ov["empty_weight_lbs"]["value"])

    mass_balance = root.find("mass_balance")
    cg_loc = mass_balance.find("location[@name='CG']") if mass_balance is not None else None
    if mass_balance is None or cg_loc is None:
        raise KeyError("Base FDM has no mass_balance/CG location")
    pointmass = ET.SubElement(mass_balance, "pointmass", {"name": PAYLOAD_POINTMASS_NAME})
    ET.SubElement(pointmass, "weight", {"unit": "LBS"}).text = " 1.0 "
    location = ET.SubElement(pointmass, "location", {"name": PAYLOAD_POINTMASS_NAME, "unit": "IN"})
    for axis in ("x", "y", "z"):
        ET.SubElement(location, axis).text = f" {cg_loc.find(axis).text.strip()} "

    propulsion = root.find("propulsion")
    if propulsion is None:
        raise KeyError("Base FDM has no <propulsion>")
    engines = propulsion.findall("engine")
    tanks = propulsion.findall("tank")
    if len(engines) != NUM_ENGINES or len(tanks) != NUM_TANKS:
        raise ValueError(f"Base FDM has {len(engines)} engines/{len(tanks)} tanks; expected {NUM_ENGINES}/{NUM_TANKS}")
    for engine in engines:
        engine.set("file", engine_name)
    per_tank = float(ov["fuel_capacity_total_lbs"]["value"]) / NUM_TANKS
    for tank in tanks:
        tank.find("capacity").text = f" {per_tank:.3f} "
        tank.find("contents").text = " 0.0 "

    notes = (
        f"Generated by envs/jsbsim_interface.py from {base_xml.name} (FDM name "
        f"'{aircraft_cfg['base_model']['fdm_name']}').\n"
        "STATUS: 7478_RESEARCH_APPROXIMATION. Overrides: wingspan, wing area, empty weight, fuel capacity, "
        "engine thrust/bypass ratio (published 747-8 values, see configs/aircraft.yaml).\n"
        "Added: PAYLOAD point mass (weight/location set per episode). Tank contents set per episode.")
    xml_path = ac_dir / f"{name}.xml"
    _write_xml(root, xml_path, notes)

    # ---------------- engine ----------------
    base_engine = jsb_root / "engine" / "GE-CF6-80C2-B1F.xml"
    etree = ET.parse(base_engine)
    eroot = etree.getroot()
    eroot.set("name", engine_name)
    _set_text(eroot, "milthrust", ov["engine_takeoff_thrust_lbf"]["value"])
    _set_text(eroot, "bypassratio", ov["engine_bypass_ratio"]["value"])
    _write_xml(eroot, eng_dir / f"{engine_name}.xml",
               "GEnx-2B67 research approximation: takeoff thrust and bypass ratio are published values;\n"
               "thrust-lapse tables, TSFC and idle/max N1/N2 are inherited from the GE CF6-80C2-B1F file.")
    shutil.copyfile(jsb_root / "engine" / "direct.xml", eng_dir / "direct.xml")

    return BuiltAircraft(model_name=name, aircraft_path=build_dir / "aircraft", engine_path=eng_dir,
                         systems_path=jsb_root / "systems", xml_path=xml_path, engine_name=engine_name)


# --------------------------------------------------------------------------------------
# JSBSim wrapper
# --------------------------------------------------------------------------------------
class JSBSimInterface:
    """Owns one JSBSim executive and exposes strict, named access to its properties."""

    def __init__(self, aircraft_cfg: dict[str, Any], physics_hz: float, project_root: Path) -> None:
        """Build the derived aircraft files and prepare the interface.

        Args:
            aircraft_cfg: Parsed ``aircraft.yaml``.
            physics_hz: JSBSim integration rate in Hz.
            project_root: Project root directory.
        """
        self.cfg = aircraft_cfg
        self.physics_hz = float(physics_hz)
        self.dt = 1.0 / self.physics_hz
        self.built = build_derived_aircraft(aircraft_cfg, project_root)
        self.fdm: jsbsim.FGFDMExec | None = None
        self.ground_agl_ref_ft: float = 0.0
        self.sim_time_start_s: float = 0.0
        self._runway: Any = None
        self._dist_origin: tuple[float, float] = (0.0, 0.0)
        self._start_xy: tuple[float, float] = (0.0, 0.0)
        self._ic_latlon: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------ properties
    def _require_fdm(self) -> jsbsim.FGFDMExec:
        """Return the live FDM or raise."""
        if self.fdm is None:
            raise JSBSimError("JSBSim executive not created yet: call start_episode() first")
        return self.fdm

    def get(self, name: str) -> float:
        """Read a property by its JSBSim name (raises if it does not exist or is non-finite)."""
        fdm = self._require_fdm()
        try:
            value = fdm.get_property_value(name)
        except KeyError as exc:
            raise PropertyNotFoundError(self._missing_message(name)) from exc
        if not math.isfinite(value):
            raise InvalidSimulatorStateError(f"Non-finite value for '{name}': {value}")
        return float(value)

    def set(self, name: str, value: float) -> None:
        """Write a property by its JSBSim name (raises if it does not exist)."""
        fdm = self._require_fdm()
        if not math.isfinite(value):
            raise InvalidSimulatorStateError(f"Refusing to set '{name}' to non-finite value {value}")
        try:
            fdm.set_property_value(name, float(value))
        except KeyError as exc:
            raise PropertyNotFoundError(self._missing_message(name)) from exc

    def _missing_message(self, name: str) -> str:
        """Build a helpful message that lists similarly named properties."""
        fdm = self._require_fdm()
        stem = name.split("[")[0].split("/")[-1]
        catalog = fdm.query_property_catalog(stem)
        hints = ", ".join(line.split(" (")[0].strip() for line in catalog.splitlines()[:8] if line.strip())
        return f"JSBSim property '{name}' does not exist. Similar: {hints or 'none'}"

    def verify_properties(self) -> None:
        """Check that every registered property exists (raises :class:`PropertyNotFoundError`)."""
        fdm = self._require_fdm()
        missing: list[str] = []
        for registry in (SENSOR_PROPERTIES, COMMAND_PROPERTIES, SETUP_PROPERTIES):
            for logical, prop in registry.items():
                try:
                    fdm.get_property_value(prop)
                except KeyError:
                    missing.append(f"{logical} -> {prop}: {self._missing_message(prop)}")
        if missing:
            raise PropertyNotFoundError("Missing JSBSim properties:\n  " + "\n  ".join(missing))

    # ------------------------------------------------------------------ lifecycle
    def _create_fdm(self) -> jsbsim.FGFDMExec:
        """Create and configure a fresh JSBSim executive with the derived aircraft loaded."""
        fdm = jsbsim.FGFDMExec(str(find_jsbsim_root()))
        fdm.set_debug_level(0)
        fdm.disable_output()
        b = self.built
        for setter, path in ((fdm.set_aircraft_path, b.aircraft_path), (fdm.set_engine_path, b.engine_path),
                             (fdm.set_systems_path, b.systems_path)):
            if not setter(str(path)):
                raise JSBSimError(f"JSBSim rejected path {path}")
        loaded = fdm.load_model(b.model_name, True)
        if not loaded:
            raise JSBSimError(f"JSBSim failed to load aircraft '{b.model_name}' from {b.aircraft_path}")
        fdm.set_dt(self.dt)
        return fdm

    def start_episode(self, cond: EpisodeConditions) -> dict[str, float]:
        """Build a fresh simulation for an episode, configure takeoff state and settle on the runway.

        Args:
            cond: Fully specified episode conditions.

        Returns:
            Metadata about the settled initial state (ground reference height, achieved CG, ...).
        """
        self.fdm = self._create_fdm()
        runway, load, wx = cond.runway, cond.loading, cond.weather
        take = self.cfg["takeoff_config"]
        self.verify_properties()
        S = SETUP_PROPERTIES

        # --- aircraft loading --------------------------------------------------------
        for i in range(NUM_TANKS):
            self.set(S[f"tank_{i}"], load.fuel_lbs / NUM_TANKS)
        self.set(S["payload_weight"], max(load.payload_lbs, 1.0))
        self.set(S["payload_x"], load.payload_station_in)

        # --- atmosphere and wind -----------------------------------------------------
        self.set(S["delta_t"], wx.delta_isa_c * 1.8)  # Kelvin deviation -> Rankine deviation
        self.set(S["p_sl_psf"], wx.qnh_hpa * 100.0 * 0.0208854342)  # hPa -> Pa -> psf
        self._runway = runway
        self.set_wind_from_components(runway, wx.headwind_kt, wx.crosswind_from_right_kt, 0.0, 0.0)
        self.set(S["random_seed"], float(cond.seed % 2_000_000_000))
        if wx.turbulence_severity > 0.0:
            self.set(S["turb_type"], 3.0)
            self.set(S["turb_severity"], wx.turbulence_severity)
            self.set(S["turb_wind20"], 15.0 * FPS_PER_KT)
        else:
            self.set(S["turb_type"], 0.0)
        self.set(S["ground_static_factor"], runway.static_friction_factor)
        self.set(S["ground_rolling_factor"], runway.rolling_friction_factor)

        # --- initial conditions: on the runway, at rest -------------------------------
        x0 = cond.start_along_m
        y0 = cond.initial_lateral_offset_m
        lat, lon = runway.runway_to_latlon(x0, y0)
        self.set(S["ic_terrain_ft"], runway.elevation_ft)
        self.set(S["ic_h_agl_ft"], 17.0)
        self.set(S["ic_lat"], lat)
        self.set(S["ic_lon"], lon)
        self.set(S["ic_psi"], (runway.heading_deg + cond.initial_heading_offset_deg) % 360.0)
        self.set(S["ic_theta"], 0.0)
        self.set(S["ic_phi"], 0.0)
        for key in ("ic_u", "ic_v", "ic_w"):
            self.set(S[key], 0.0)
        fdm = self._require_fdm()
        if not fdm.run_ic():
            raise JSBSimError("JSBSim run_ic() failed")

        # --- takeoff configuration (never left to the agent) --------------------------
        self.set(S["set_running"], -1.0)
        self.set(S["gear_cmd"], 1.0 if take["gear_down"] else 0.0)
        self.set(S["flap_cmd"], float(take["flaps_deg"]))
        self.set(S["speedbrake_cmd"], float(take["speedbrake"]))
        self.set(S["pitch_trim_cmd"], float(take["pitch_trim_norm"]))
        self.set_flight_controls(throttle=0.0, elevator_cmd=0.0, aileron=0.0, rudder=0.0, steer=0.0)
        for key in ("brake_left", "brake_right", "brake_center"):
            self.set(S[key], 1.0)  # held during pre-roll settle only

        self.step(int(round(float(take["preroll_settle_s"]) * self.physics_hz)))

        for key in ("brake_left", "brake_right", "brake_center"):
            self.set(S[key], 0.0 if not take["parking_brake"] else 1.0)
        self.ground_agl_ref_ft = self.get(SENSOR_PROPERTIES["h_agl_ft"])
        self.sim_time_start_s = self.get(SENSOR_PROPERTIES["sim_time_s"])
        self._ic_latlon = (lat, lon)
        self._start_xy = (x0, y0)
        self._dist_origin = self._signed_offset_ned(self.read_sensors())
        return {"ground_agl_ref_ft": self.ground_agl_ref_ft, "cg_x_in": self.get(SENSOR_PROPERTIES["cg_x_in"]),
                "weight_lbs": self.get(SENSOR_PROPERTIES["weight_lbs"])}

    # ------------------------------------------------------------------ control
    def set_wind_from_components(self, runway: Any, headwind_kt: float, crosswind_from_right_kt: float,
                                 gust_head_kt: float, gust_cross_kt: float) -> None:
        """Set the steady+gust wind (runway-relative components, knots) in the JSBSim NED wind properties."""
        head = (headwind_kt + gust_head_kt) * 0.514444
        cross = (crosswind_from_right_kt + gust_cross_kt) * 0.514444
        north_mps, east_mps = runway.wind_ned_from_components(head, cross)
        self.set(SETUP_PROPERTIES["wind_north"], north_mps / 0.3048)
        self.set(SETUP_PROPERTIES["wind_east"], east_mps / 0.3048)
        self.set(SETUP_PROPERTIES["wind_down"], 0.0)

    def set_flight_controls(self, throttle: float, elevator_cmd: float, aileron: float, rudder: float,
                            steer: float) -> None:
        """Write the JSBSim command properties (already converted to JSBSim conventions)."""
        for i in range(NUM_ENGINES):
            self.set(COMMAND_PROPERTIES[f"throttle_cmd_{i}"], throttle)
        self.set(COMMAND_PROPERTIES["elevator_cmd"], elevator_cmd)
        self.set(COMMAND_PROPERTIES["aileron_cmd"], aileron)
        self.set(COMMAND_PROPERTIES["rudder_cmd"], rudder)
        self.set(COMMAND_PROPERTIES["steer_cmd"], steer)

    def step(self, n_physics_steps: int) -> None:
        """Advance JSBSim by ``n_physics_steps`` physics steps (one ``fdm.run()`` each)."""
        fdm = self._require_fdm()
        for _ in range(n_physics_steps):
            if not fdm.run():
                raise InvalidSimulatorStateError("JSBSim run() returned False (simulation ended)")

    # ------------------------------------------------------------------ sensing
    def read_sensors(self) -> dict[str, float]:
        """Read every registered sensor property once; all values are finite or an error is raised."""
        return {logical: self.get(prop) for logical, prop in SENSOR_PROPERTIES.items()}

    def _signed_offset_ned(self, sensors: dict[str, float]) -> tuple[float, float]:
        """North/east displacement (m) from the IC point.

        JSBSim's ``position/distance-from-start-*`` properties are UNSIGNED magnitudes (verified
        against velocity integration), so the sign is taken from the lat/lon difference.
        """
        north = math.copysign(sensors["dist_north_m"], sensors["lat_deg"] - self._ic_latlon[0])
        east = math.copysign(sensors["dist_east_m"], sensors["lon_deg"] - self._ic_latlon[1])
        return north, east

    def runway_xy(self, sensors: dict[str, float]) -> tuple[float, float]:
        """Return the aircraft CG position in runway coordinates ``(along_m, lateral_right_m)``."""
        north, east = self._signed_offset_ned(sensors)
        d_along, d_lateral = self._runway.ned_to_runway(north - self._dist_origin[0], east - self._dist_origin[1])
        return self._start_xy[0] + d_along, self._start_xy[1] + d_lateral

    def close(self) -> None:
        """Release the executive."""
        self.fdm = None
