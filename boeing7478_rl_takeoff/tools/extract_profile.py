"""Extract the "best observed takeoff profile" from clean, successful episodes (median across seeds).

Output: results/demos/best_takeoff_profile.json — reference targets (attitude, controls, speed, altitude)
that the reward terms, the teacher gains, the demos and the 3D viewer annotations are all consistent with.

Usage:  python tools/extract_profile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs import load_all_configs
from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv
from teacher.simulated_teacher import SimulatedTeacher

OUT = Path("results/demos/best_takeoff_profile.json")


def main() -> None:
    """Collect teacher takeoffs, keep clean successes, write the interpolated median profile."""
    cfg = load_all_configs()
    env = Boeing7478TakeoffEnv(configs=cfg)
    teacher = SimulatedTeacher(cfg)
    episodes: list[dict[str, list]] = []
    for i in range(60):
        obs, _ = env.reset(seed=3100 + i, options={"level": 1})
        cols: dict[str, list] = {k: [] for k in
                                 ("t", "kt", "agl", "vs", "pitch", "roll", "lat", "along", "thr", "ele", "ail", "rud")}
        done = False
        while not done:
            a = teacher.act(obs)
            obs, _r, te, tr, info = env.step(a)
            done = te or tr
            cols["t"].append(info["time_s"])
            cols["kt"].append(info["airspeed_kt"])
            cols["agl"].append(info["agl_ft"])
            cols["vs"].append(info["vertical_speed_fpm"])
            cols["pitch"].append(info["pitch_deg"])
            cols["roll"].append(info["roll_deg"])
            cols["lat"].append(info["lateral_m"])
            cols["along"].append(info["along_m"])
            cols["thr"].append(info["actuators"][0])
            cols["ele"].append(info["actuators"][1])
            cols["ail"].append(info["actuators"][2])
            cols["rud"].append(info["actuators"][3])
        s = info["episode_summary"]
        if s["success"] and s["cleanliness"] > 0.85:
            episodes.append(cols)
    if not episodes:
        raise RuntimeError("no clean successful teacher episode collected")
    grid_t = np.linspace(0.0, 60.0, 241)
    profile: dict = {
        "description": ("Median profile across successful, very clean (cleanliness>0.85) simulated teacher takeoffs, level 1. "
                        "Training target: hold the centerline (lateral ~0), small roll/rudder, full throttle from t=0, "
                        "pitch ~10 deg in the initial climb, liftoff ~1500 m."),
        "t_s": np.round(grid_t, 2).tolist(),
    }
    for k in episodes[0]:
        if k == "t":
            continue
        series = [np.interp(grid_t, ep["t"], ep[k], left=np.nan, right=np.nan) for ep in episodes]
        profile[k] = np.round(np.nanmedian(np.asarray(series, dtype=float), axis=0), 2).tolist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(profile), encoding="utf-8")
    print(f"episodes used: {len(episodes)} (clean level-1 takeoffs); median duration {np.median([ep['t'][-1] for ep in episodes]):.1f} s")
    print("t:      ", ", ".join(f"{t:5.1f}" for t in grid_t[::60]))
    for k in ("kt", "agl", "vs", "pitch", "roll", "lat", "along", "thr", "ele", "ail", "rud"):
        arr = np.array(profile[k], dtype=float)
        print(f"{k:6s} ", ", ".join(f"{v:6.1f}" for v in arr[::60]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
