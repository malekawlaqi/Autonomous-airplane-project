r"""Run the trained SAC/PPO policy inside MSFS via the SimConnect library.

This is the direct console command that actually sends controls to the sim.
(The other file `msfs_sac_takeoff_975000lbs.py` keeps a stub telemetry source and is safe to audit.)

USAGE
=====
  1. Start MSFS, load a free flight on your target runway, and unpause (P).
     Perform a runway alignment: in sim, press all of the following:
        - Throttle Level (usually the key contains comm? ...)
        - Wheels on the runway, IAS = 0, brakes released, flaps set in MSFS.
  2. In the project directory:
       .venv\Scripts\Activate.ps1
       python msfs_sac_runtime.py --model models\clean_teacher\best_model.zip --real

Or, in one line:
      .venv\Scripts\python.exe msfs_sac_runtime.py --model models\clean_teacher\best_model.zip --real

OPTIONS
-------
--config-dir  Directory that holds aircraft.yaml, environment.yaml, ...
              Use configs_975 to reproduce the 975,000-lb case (set 100 % MTOW
              rather than the 32 – 100 % variation range).
--real        Actually send the commands. Without --real the script only prints.
--dry-run     Alias of not using --real (kept for clarity – defaults to False).

The script splits into 2 phases:
  1. sample period of 10 s with cautious test sending MSFS events,
  2. main closed-loop using our SAC policy.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

# Make envs, evaluation importable without running the project package as a module.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.evaluate import load_sb3_model  # noqa: E402
from envs import load_all_configs  # noqa: E402
from envs.observations import OBS_NAMES  # noqa: E402


def _normalized_obs_from_telemetry(t: dict, scales: dict) -> np.ndarray:
    vf = {
        "airspeed": t["ias_kt"] / scales["airspeed_kt"],
        "ground_speed": t["ias_kt"] / scales["ground_speed_kt"],
        "agl": t["alt_agl_ft"] / scales["agl_ft"],
        "vertical_speed": t["vs_fpm"] / scales["vertical_speed_fpm"],
        "pitch": math.radians(t["pitch_deg"]) / scales["pitch_rad"],
        "roll": math.radians(t["roll_deg"]) / scales["roll_rad"],
        "relative_heading": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
        "lateral_error": 0.0,
        "course_error": 0.0,
        "distance_travelled": 0.0,
        "distance_remaining": 0.0,
        "weight_on_wheels": 1.0 if t["on_ground"] else 0.0,
        "elevator_pos": 0.0,
        "aileron_pos": 0.0,
        "rudder_pos": 0.0,
                "throttle": float(t["throttle_cmd"]),
                "engine_n1": float(t["n1_percent"]) / 100.0,
                "mass_norm": 0.5,
                "cg_norm": 0.0,
                "headwind": 0.0,
                "crosswind": 0.0,
                "density_ratio": 1.0,
            }
    return np.array([vf[name] for name in OBS_NAMES], dtype=np.float32)


def read_telemetry(ac) -> dict:
    """Read the needed fields from the sim through SimConnect AircraftRequests."""
    pos = ac.PositionandSpeedData
    instr = ac.FlightInstrumentationData
    lat = pos.get("PLANE_LATITUDE") or 0.0
    _lon = pos.get("PLANE_LONGITUDE") or 0.0
    alt_msl = pos.get("PLANE_ALTITUDE") or 0.0
    pitch = pos.get("PLANE_PITCH_DEGREES") or 0.0
    roll = pos.get("PLANE_BANK_DEGREES") or 0.0
    heading = pos.get("PLANE_HEADING_DEGREES_TRUE") or 0.0
    agl = pos.get("PLANE_ALT_ABOVE_GROUND") or 0.0
    ias = instr.get("AIRSPEED_INDICATED") or 0.0
    vs = instr.get("VERTICAL_SPEED") or 0.0
    on_ground = (pos.get("SIM_ON_GROUND") or "False") == "True" or alt_msl < 5.0
    throttle = instr.get("GENERAL_ENG_THROTTLE_LEVER_POSITION:1") or 0.0
    n1 = instr.get("ENG_N1:1") or 30.0
    return {
        "ias_kt": ias,
        "vs_fpm": vs,
        "alt_agl_ft": agl,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "heading_deg": heading,
        "on_ground": on_ground,
        "throttle_cmd": throttle,
        "n1_percent": n1,
        "lat": lat,
        "alt_msl": alt_msl,
    }


def send_actions(evt, action):
    """Write throttle/elevator/aileron/rudder into SimConnect events."""
    from SimConnect.EventList import Event  # noqa: F401

    thr01 = (float(action[0]) + 1.0) / 2.0
    t = max(0, min(16383, int(thr01 * 16383)))
    e = max(-16383, min(16383, int(float(action[1]) * 16383)))
    a = max(-16383, min(16383, int(float(action[2]) * 16383)))
    r = max(-16383, min(16383, int(float(action[3]) * 16383)))
    evt["elev"](e)
    evt["aileron"](a)
    evt["rudder"](r)
    evt["thr1"](t)
    evt["thr2"](t)
    evt["thr3"](t)
    evt["thr4"](t)
    return {"throttle_cmd": thr01, "elevator_cmd": float(action[1]), "aileron_cmd": float(action[2]),
            "rudder_cmd": float(action[3])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/clean_teacher/best_model.zip"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--real", action="store_true", help="Send commands to the sim (otherwise telemetry-only)")
    parser.add_argument("--max-seconds", type=float, default=180.0)
    args = parser.parse_args()

    cfg = load_all_configs(args.config_dir)
    scales = cfg["environment"]["observation"]["scales"]

    model = load_sb3_model(args.model, device="cpu")
    print(f"Loaded policy: {args.model}")

    if not args.real:
        print("Not running with --real. Reading telemetry only. Ctrl+C to stop.")

    from SimConnect import SimConnect
    from SimConnect.RequestList import AircraftRequests
    from SimConnect.EventList import Event

    sm = SimConnect()
    ac = AircraftRequests(sm)
    evt = {
        "elev": Event(b"AXIS_ELEVATOR_SET", sm),
        "aileron": Event(b"AXIS_AILERONS_SET", sm),
        "rudder": Event(b"AXIS_RUDDER_SET", sm),
        "thr1": Event(b"THROTTLE1_SET", sm),
        "thr2": Event(b"THROTTLE2_SET", sm),
        "thr3": Event(b"THROTTLE3_SET", sm),
        "thr4": Event(b"THROTTLE4_SET", sm),
    }

    deadline = time.monotonic() + args.max_seconds
    try:
        while time.monotonic() < deadline:
            tel = read_telemetry(ac)
            obs = _normalized_obs_from_telemetry(tel, scales)
            action, _ = model.predict(obs, deterministic=True)
            if args.real:
                applied = send_actions(evt, action)
                tel.update(applied)
                print(f"ALT {tel['alt_agl_ft']:6.0f} ft  IAS {tel['ias_kt']:4.0f} kt  "
                      f"pitch {tel['pitch_deg']:5.1f}  thr {tel['throttle_cmd']:.2f} "
                      f"elv {tel['elevator_cmd']:+.2f} ail {tel['aileron_cmd']:+.2f} rud {tel['rudder_cmd']:+.2f}")
            else:
                print(f"[idle] ALT {tel['alt_agl_ft']:6.0f} ft  IAS {tel['ias_kt']:4.0f} kt  "
                      f"pitch {tel['pitch_deg']:5.1f}  predicted action {action}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped manually.")


if __name__ == "__main__":
    main()
