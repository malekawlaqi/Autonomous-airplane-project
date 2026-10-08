"""Inspect the aircraft model, verify control signs and run a minimal JSBSim smoke test.

Usage:
    python tools/inspect_aircraft.py                 # model identity / status
    python tools/inspect_aircraft.py --smoke         # open-loop takeoff-roll smoke test
    python tools/inspect_aircraft.py --signs         # verify control-sign conventions
    python tools/inspect_aircraft.py --catalog n1    # list JSBSim properties matching a pattern
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jsbsim  # noqa: E402

from envs import PROJECT_ROOT, load_all_configs  # noqa: E402
from envs.jsbsim_interface import (COMMAND_PROPERTIES, SENSOR_PROPERTIES, JSBSimInterface,  # noqa: E402
                                   find_jsbsim_root)
from envs.randomization import DomainRandomizer  # noqa: E402


def report_identity(cfg: dict) -> None:
    """Print which JSBSim aircraft exist and whether a real 747-8 FDM is present."""
    root = find_jsbsim_root()
    aircraft = sorted(p.name for p in (root / "aircraft").iterdir() if p.is_dir())
    matches = [a for a in aircraft if "747-8" in a or "748" in a]
    print(f"JSBSim version        : {jsbsim.__version__}")
    print(f"JSBSim data root      : {root}")
    print(f"Bundled aircraft ({len(aircraft)}) : {', '.join(aircraft)}")
    print(f"Genuine 747-8 FDM dir : {matches if matches else 'NONE FOUND'}")
    base = cfg["base_model"]
    print(f"Base model used       : aircraft/{base['jsbsim_aircraft_dir']} (FDM name '{base['fdm_name']}')")
    print(f"MODEL STATUS          : {cfg['model_status']}")
    iface = JSBSimInterface(cfg, 120.0, PROJECT_ROOT)
    print(f"Derived FDM written to: {iface.built.xml_path}")


def nominal_conditions(cfg_all: dict, seed: int = 0, level: int = 1):
    """Sample a reproducible set of episode conditions."""
    return DomainRandomizer(cfg_all["randomization"], cfg_all["aircraft"]).sample(level, seed)


def run_smoke(cfg_all: dict, seconds: float) -> None:
    """Open-loop smoke test: full throttle, neutral controls, print the takeoff-roll timeline."""
    cond = nominal_conditions(cfg_all)
    iface = JSBSimInterface(cfg_all["aircraft"], 120.0, PROJECT_ROOT)
    meta = iface.start_episode(cond)
    print(f"Conditions: runway {cond.runway.length_m:.0f} m, hdg {cond.runway.heading_deg:.0f}, elev "
          f"{cond.runway.elevation_ft:.0f} ft, weight {meta['weight_lbs']:.0f} lb, CG x {meta['cg_x_in']:.1f} in")
    print(f"Ground reference h-agl (CG on gear): {meta['ground_agl_ref_ft']:.2f} ft")
    iface.set_flight_controls(throttle=1.0, elevator_cmd=0.0, aileron=0.0, rudder=0.0, steer=0.0)
    print(f"{'t[s]':>6} {'KCAS':>7} {'along[m]':>9} {'lat[m]':>7} {'N1':>6} {'thrust0':>9} {'pitch':>6} {'WOW':>4}")
    for second in range(int(seconds) + 1):
        s = iface.read_sensors()
        along, lat = iface.runway_xy(s)
        if second % 5 == 0:
            print(f"{second:6d} {s['vc_kts']:7.1f} {along:9.1f} {lat:7.2f} {s['n1_0']:6.1f} "
                  f"{s['thrust_0_lbs']:9.0f} {s['theta_rad'] * 57.3:6.2f} {int(s['wow_left'])}")
        iface.step(120)
    print("SMOKE TEST COMPLETED (no exceptions raised)")


def _hold(iface: JSBSimInterface, seconds: float, **controls: float) -> dict[str, float]:
    """Apply controls for ``seconds`` and return the sensors afterwards."""
    iface.set_flight_controls(**controls)
    iface.step(int(seconds * iface.physics_hz))
    return iface.read_sensors()


def run_signs(cfg_all: dict) -> None:
    """Empirically determine how JSBSim command signs map to aircraft response."""
    cond = nominal_conditions(cfg_all)
    iface = JSBSimInterface(cfg_all["aircraft"], 120.0, PROJECT_ROOT)
    iface.start_episode(cond)
    base = dict(throttle=1.0, elevator_cmd=0.0, aileron=0.0, rudder=0.0, steer=0.0)
    # Accelerate to ~100 kt so the surfaces have authority, still on the ground.
    for _ in range(60):
        s = _hold(iface, 1.0, **base)
        if s["vc_kts"] > 100.0:
            break
    print(f"Test speed {s['vc_kts']:.1f} KCAS, on ground WOW={int(s['wow_left'])}")
    for label, key, amplitude in (("elevator-cmd +1", "elevator_cmd", 1.0), ("elevator-cmd -1", "elevator_cmd", -1.0),
                                  ("aileron-cmd +0.5", "aileron", 0.5), ("rudder-cmd +0.5", "rudder", 0.5),
                                  ("steer-cmd +0.5", "steer", 0.5)):
        ctrl = dict(base)
        ctrl[key] = amplitude
        before = iface.read_sensors()
        after = _hold(iface, 1.0, **ctrl)
        print(f"{label:>18}: dq={after['q_rad_s'] - before['q_rad_s']:+.4f} rad/s  "
              f"dp={after['p_rad_s'] - before['p_rad_s']:+.4f}  dr={after['r_rad_s'] - before['r_rad_s']:+.4f}  "
              f"d_pitch={(after['theta_rad'] - before['theta_rad']) * 57.3:+.2f} deg  "
              f"elev_pos={after['elevator_pos_rad']:+.3f} ail_pos={after['aileron_pos_rad']:+.3f} "
              f"rud_pos={after['rudder_pos_rad']:+.3f} steer_deg={after['steer_pos_deg']:+.2f}  "
              f"d_psi={(after['psi_rad'] - before['psi_rad']) * 57.3:+.2f} deg")
        _hold(iface, 1.0, **base)
    print("Interpretation: +q = nose up, +p = roll right, +r = yaw right, +d_psi = heading increases (turn right).")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="run the open-loop smoke test")
    parser.add_argument("--seconds", type=float, default=60.0, help="smoke-test duration")
    parser.add_argument("--signs", action="store_true", help="verify control sign conventions")
    parser.add_argument("--catalog", type=str, default=None, help="list JSBSim properties matching a pattern")
    args = parser.parse_args()
    cfg_all = load_all_configs()
    if args.catalog is not None:
        iface = JSBSimInterface(cfg_all["aircraft"], 120.0, PROJECT_ROOT)
        iface.start_episode(nominal_conditions(cfg_all))
        print(iface.fdm.query_property_catalog(args.catalog))  # type: ignore[union-attr]
    elif args.smoke:
        run_smoke(cfg_all, args.seconds)
    elif args.signs:
        run_signs(cfg_all)
    else:
        report_identity(cfg_all["aircraft"])
        print(f"Registered sensor properties: {len(SENSOR_PROPERTIES)}, command properties: {len(COMMAND_PROPERTIES)}")


if __name__ == "__main__":
    main()
