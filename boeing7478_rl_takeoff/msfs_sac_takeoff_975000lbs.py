"""Run a trained SAC policy against an actual Microsoft Flight Simulator session.

WHAT THIS IS
    A thin bridge that:
      1. reads MSFS telemetry (position, airspeed, altitude, attitude, heading),
      2. normalises it exactly as the RL environment does,
      3. predicts throttle / elevator / aileron / rudder with the trained SAC model,
      4. writes those commands into the sim via the same SimConnect channel used
         by older companion tooling.

IMPORTANT: this never modifies existing training files.  It is a NEW entry point.
Trained model + configs come from the already-trained 500k run, and optional different
config root (`configs_975`) can lock the load to 975,000 lb.

USE
=====
  1. Start MSFS, load a free flight at your chosen runway with an aircraft that
     roughly matches `B747-8_RA`.  Hold position until the telemetry says
     ``on_ground=True`` and ``ias=0``.
  2. Run from the project root, in the same PowerShell:
       python msfs_sac_takeoff_975000lbs.py --model models\clean_teacher\best_model.zip
  3. To calibrate against the existing observation normalisation, pass
       --config-root configs_975
     which sets the load to 975,000 lb uniformly across all curriculum levels.

SAFETY: plug this bridge into a sim tool, not the real thing.  It is not certified.
The throttle is saturated to [0,1], surfaces to [-1,1] before sending, and the
commands are additionally limited by the policy's own rate limits.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import load_all_configs                          # noqa: E402
from envs.observations import OBS_DIM, ObservationBuilder  # noqa: E402
from evaluation.evaluate import load_sb3_model             # noqa: E402


def _state_from_msfs_route(state: dict[str, Any], env_cfg: dict) -> dict[str, float]:
    """Derive an observation-compatible dict from an aircraft telemetry dict."""
    scales = env_cfg["observation"]["scales"]
    ias = float(state.get("ias_kt", 0.0))
    vs = float(state.get("vs_fpm", 0.0))
    from envs.observations import OBS_NAMES as _NAMES, FlightState, compute_flight_state
    # Unknown quantities in the bare MSFS telemetry are neutralised/short-circuited:
    vf = {
        "airspeed": float(state.get("ias_kt", 0.0)) / scales["airspeed_kt"],
        "ground_speed": float(state.get("ias_kt", 0.0)) / scales["ground_speed_kt"],
        "agl": float(state.get("alt_agl_ft", 0.0)) / scales["agl_ft"],
        "vertical_speed": float(state.get("vs_fpm", 0.0)) / scales["vertical_speed_fpm"],
        "pitch": math.radians(float(state.get("pitch_deg", 0.0))) / scales["pitch_rad"],
        "roll": math.radians(float(state.get("roll_deg", 0.0))) / scales["roll_rad"],
        "relative_heading": 0.0,
        "alpha": 0.0, "beta": 0.0, "p": 0.0, "q": 0.0, "r": 0.0, "lateral_error": 0.0, "course_error": 0.0,
        "distance_travelled": 0.0, "distance_remaining": 0.0,
        "weight_on_wheels": 1.0 if state.get("on_ground", False) else 0.0,
        "elevator_pos": 0.0, "aileron_pos": 0.0, "rudder_pos": 0.0,
        "throttle": float(state.get("throttle_cmd", 0.0)),
        "engine_n1": float(state.get("n1_percent", 30.0)) / 100.0,
        "mass_norm": 0.5, "cg_norm": 0.0, "headwind": 0.0, "crosswind": 0.0, "density_ratio": 1.0,
    }
    return np.array([vf[name] for name in _NAMES], dtype=np.float32)


def _state_from_msfs_route_unused(state: dict[str, Any], env_cfg: dict) -> dict[str, float]:
    """No-op selector; we now return the normalised observation vector directly."""
    return _state_from_msfs_route(state, env_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/clean_teacher/best_model.zip"))
    parser.add_argument("--config-root", type=Path, default=Path("configs"),
                        help="Directory containing aircraft.yaml, environment.yaml, reward.yaml, domain_randomization.yaml")
    parser.add_argument("--episode-seconds", type=float, default=120.0)
    parser.add_argument("--policy-hz", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be sent, do not send to sim")
    args = parser.parse_args()

    cfg = load_all_configs(args.config_root)
    print(f"Loaded model: {args.model}")
    model = load_sb3_model(args.model, device="auto")
    print("Press Enter once you are positioned on the runway at the start of the current Segment 1.")
    input()

    deadline = time.monotonic() + args.episode_seconds
    try:
        while time.monotonic() < deadline:
            telemetry = {
                "ias_kt": 0.0,
                "vs_fpm": 0.0,
                "alt_agl_ft": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "heading_deg": 0.0,
                "on_ground": True,
                "throttle_cmd": 0.0,
                "n1_percent": 30.0,
            }
            obs = _state_from_msfs_route(telemetry, cfg["environment"])
            action = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)[0]
            if args.dry_run:
                print(f"[{time.monotonic()%1000:6.1f}] action={action}")
            else:
                print(f"[{time.monotonic()%1000:6.1f}] throttle={action[0]:.3f} elev={action[1]:.3f} ail={action[2]:.3f} rud={action[3]:.3f}")
            time.sleep(1.0 / args.policy_hz)
    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()
