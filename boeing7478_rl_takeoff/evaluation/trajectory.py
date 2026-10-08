"""Trajectory recording for the live 3D showcase (kept out of the training/physics code paths).

An *event* is one replayable episode plus a headline, e.g. "Training success #3 at 41,000 transitions".
Events are JSON files under ``results/<run>/showcase/events/`` that ``tools/live_viewer.py`` serves.

Frame row layout (matches the 3D viewer): ``[t, along_m, lateral_m, height_m, roll_deg, pitch_deg,
heading_error_deg, airspeed_kt, vertical_speed_fpm, throttle, elevator, aileron, rudder]``.

Backfill existing checkpoints into events (so the viewer has history immediately):
    python -m evaluation.trajectory --run sac_pilot --level 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PROJECT_ROOT, load_all_configs  # noqa: E402

FT_PER_M: float = 3.280839895
CG_HEIGHT_M: float = 12.49 / FT_PER_M   # wheel contact -> CG height of the settled FDM (see AIRCRAFT_MODEL_NOTES.md)


def frame_row(time_s: float, info: dict[str, Any]) -> list[float]:
    """Pack one pose/instrument frame from an ``info`` dict (fields as produced by the environment)."""
    a = info["actuators"]
    return [round(float(time_s), 2), round(info["along_m"], 2), round(info["lateral_m"], 2),
            round(info["agl_ft"] / FT_PER_M, 2), round(info["roll_deg"], 2), round(info["pitch_deg"], 2),
            round(info["heading_error_deg"], 2), round(info["airspeed_kt"], 1), round(info["vertical_speed_fpm"], 0),
            round(float(a[0]), 3), round(float(a[1]), 3), round(float(a[2]), 3), round(float(a[3]), 3)]


def with_initial_frame(frames: list[list[float]]) -> list[list[float]]:
    """Prepend a t=0 frame (copy of the first recorded pose) so the viewer's index math starts at t=0."""
    if not frames or frames[0][0] <= 1e-6:
        return frames
    first = list(frames[0])
    first[0] = 0.0
    return [first] + frames


def run_description(summary: dict[str, Any], frames: list[list[float]], label: str, runway_width_m: float) -> dict[str, Any]:
    """Build the viewer's run dictionary from an environment ``episode_summary`` and recorded frames."""
    return {"label": label, "reason": summary["reason"], "success": bool(summary["success"]),
            "duration_s": round(float(summary["duration_s"]), 1), "runway_length_m": round(float(summary["runway_length_m"]), 1),
            "runway_width_m": runway_width_m, "vr_kt": round(float(summary["vr_est_kt"]), 1),
            "info": (f"{summary['mass_lbs'] / 1000:.0f}k lb | xwind {summary['crosswind_kt']:+.0f} kt, "
                     f"headwind {summary['headwind_kt']:+.0f} kt | mu x{summary['static_friction_factor']:.2f} | "
                     f"elev {summary['runway_elevation_ft']:.0f} ft | gust {summary['gust_sigma_kt']:.1f} kt | L{int(summary['level'])}"
                     + (f" | clean {summary['cleanliness']:.2f}" if "cleanliness" in summary else "")),
            "cg_height_m": round(CG_HEIGHT_M, 2), "frames": with_initial_frame(frames)}


def record_episode(env: Any, policy: Callable[[np.ndarray], np.ndarray], seed: int, level: int, label: str,
                   runway_width_m: float = 60.0, conditions: Any = None) -> dict[str, Any]:
    """Roll out one episode with ``policy`` on ``env`` and return the viewer run dictionary.

    Args:
        conditions: Optional exact ``EpisodeConditions`` to fly (used to reconstruct past episodes).
    """
    options = {"conditions": conditions} if conditions is not None else {"level": level}
    obs, _ = env.reset(seed=seed, options=options)
    frames: list[list[float]] = []
    while True:
        obs, _r, terminated, truncated, info = env.step(policy(obs))
        frames.append(frame_row(info["time_s"], info))
        if terminated or truncated:
            return run_description(info["episode_summary"], frames, label, runway_width_m)


class ShowcaseRecorder:
    """Writes showcase events and a small status file for one training run."""

    def __init__(self, run_name: str, results_dir: Path, cfg: dict[str, Any], runway_width_m: float = 60.0) -> None:
        """Create the recorder (directories are created lazily; existing events are counted for resume)."""
        self.run_name = run_name
        self.enabled = bool(cfg.get("enabled", True))
        self.always_until = int(cfg.get("training_success_always_until", 25))
        self.min_gap = int(cfg.get("training_success_min_gap_steps", 10_000))
        self.eval_episodes = int(cfg.get("eval_episodes_recorded", 3))
        self.progress_delta = float(cfg.get("progress_min_delta", 0.10))
        self.max_events = int(cfg.get("max_events", 400))
        self.save_snapshots = bool(cfg.get("save_model_snapshots", True))
        self.update_every = int(cfg.get("update_every_steps", 50_000))   # periodic viewer update (0 = off)
        self.update_episodes = int(cfg.get("update_episodes", 2))        # deterministic replays recorded per update
        self.runway_width_m = runway_width_m
        self.dir = results_dir / "showcase"
        self.events_dir = self.dir / "events"
        self.counter = len(list(self.events_dir.glob("*.json"))) if self.events_dir.exists() else 0
        existing = self.events_dir.glob("*.json") if self.events_dir.exists() else []
        names = [p.name for p in existing]
        self.training_successes_recorded = sum(("_FIRST_SUCCESS_" in n or "_TRAINING_SUCCESS_" in n) for n in names)
        self.seen_first_liftoff = any("_FIRST_LIFTOFF_" in n for n in names)
        self.last_success_event_step = -10 ** 12
        self.shared: dict[str, Any] = {}   # filled by the callbacks (e.g. latest evaluation) for the status file

    def should_record_training_success(self, num_timesteps: int) -> bool:
        """Rate-limit success events: all of the first N, then at most one per ``min_gap`` transitions."""
        if not self.enabled or self.counter >= self.max_events:
            return False
        if self.training_successes_recorded < self.always_until:
            return True
        return num_timesteps - self.last_success_event_step >= self.min_gap

    def write_event(self, event_type: str, headline: str, num_timesteps: int, level: int, run: dict[str, Any],
                    extra: dict[str, Any] | None = None) -> Path | None:
        """Write one event JSON (returns its path, or None when disabled / at the event cap)."""
        if not self.enabled or self.counter >= self.max_events:
            return None
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        event = dict(run)
        event.update({"event_type": event_type, "headline": headline, "run_name": self.run_name,
                      "num_timesteps": int(num_timesteps), "level": int(level), "wall_time": time.time(),
                      "event_id": f"{self.run_name}/{self.counter:05d}"})
        if extra:
            event.update(extra)
        event["label"] = f"{self.run_name} @ {num_timesteps:,} - {event_type}"
        path = self.events_dir / f"e{self.counter:05d}_{event_type}_{int(num_timesteps)}.json"
        path.write_text(json.dumps(event, separators=(",", ":")), encoding="utf-8")
        return path

    def write_status(self, status: dict[str, Any]) -> None:
        """Atomically replace ``status.json`` (read by the live viewer's training panel)."""
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"status.json.tmp.{os.getpid()}"   # unique per process; the viewer server reads this file
        status = dict(status, run_name=self.run_name, wall_time=time.time())
        try:
            tmp.write_text(json.dumps(status), encoding="utf-8")
            tmp.replace(self.dir / "status.json")
        except OSError as exc:   # viewer is reading status.json right now: skip this write (next one will land)
            import logging; logging.getLogger(__name__).warning("status write delayed: %s", exc)
            try: tmp.unlink(missing_ok=True)
            except OSError: pass


def backfill_checkpoints(run: str, level: int, seeds: list[int], device: str = "cpu") -> int:
    """Create events from existing ``checkpoint_<N>.zip`` files of ``results/<run>`` (deterministic replays)."""
    from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv
    from evaluation.evaluate import load_sb3_model, sb3_policy

    configs = load_all_configs()
    models_dir = PROJECT_ROOT / "models" / run
    checkpoints = sorted((p for p in models_dir.glob("checkpoint_*.zip")), key=lambda p: int(p.stem.split("_")[1]))
    recorder = ShowcaseRecorder(run, PROJECT_ROOT / "results" / run, {"enabled": True})
    env = Boeing7478TakeoffEnv(configs=configs)
    width = float(configs["randomization"]["sampling"]["runway_width_m"])
    written = 0
    for ckpt in checkpoints:
        steps = int(ckpt.stem.split("_")[1])
        policy = sb3_policy(load_sb3_model(ckpt, device))
        for seed in seeds:
            episode = record_episode(env, policy, seed, level, "", width)
            verdict = "SUCCESS" if episode["success"] else episode["reason"]
            recorder.write_event("CHECKPOINT", f"Checkpoint {steps:,} transitions: {verdict} (deterministic replay, seed {seed})",
                                 steps, level, episode)
            written += 1
    return written


def reconstruct_past_successes(run: str, models_run: str | None = None, max_tries: int = 40,
                               device: str = "cpu", before_steps: int | None = None) -> dict[str, int]:
    """Re-create viewer events for successes logged BEFORE trajectory recording existed.

    The original trajectories were not saved (they used stochastic exploration noise that cannot be recovered).
    What *is* in ``results/<run>/training_episodes.csv`` is each success' seed and level, hence its exact
    conditions (runway, mass, CG, weather).  Each past success is re-flown under those conditions with the nearest
    available checkpoint policy: first deterministically, then with up to ``max_tries`` stochastic samples, keeping
    the first run that succeeds.  Events are labelled ``PAST_SUCCESS`` and say they are reconstructions.

    Returns:
        Counts: ``found`` (successes in the log), ``reconstructed`` and ``skipped`` (no checkpoint / not reproduced).
    """
    import pandas as pd
    import torch

    from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv
    from envs.randomization import DomainRandomizer
    from evaluation.evaluate import load_sb3_model

    configs = load_all_configs()
    results_dir = PROJECT_ROOT / "results" / run
    models_dir = PROJECT_ROOT / "models" / (models_run or run)
    log = pd.read_csv(results_dir / "training_episodes.csv")
    successes = log[log["success"].astype(bool)]
    if before_steps is not None:   # later successes were already recorded live (exact trajectories)
        successes = successes[successes["num_timesteps"] <= before_steps]
    successes = successes.reset_index(drop=True)
    checkpoints = sorted(int(p.stem.split("_")[1]) for p in models_dir.glob("checkpoint_*.zip")
                         if p.stem.split("_")[1].isdigit())
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {models_dir}")
    randomizer = DomainRandomizer(configs["randomization"], configs["aircraft"])
    env = Boeing7478TakeoffEnv(configs=configs)
    recorder = ShowcaseRecorder(run, results_dir, {"enabled": True, "max_events": 10_000},
                                float(configs["randomization"]["sampling"]["runway_width_m"]))
    base_time = time.time() - 1_000_000.0   # sorts before every live event in the viewer's playlist
    models: dict[int, Any] = {}
    counts = {"found": len(successes), "reconstructed": 0, "skipped": 0, "already_done": 0}
    already = set()   # idempotent: successes reconstructed by an earlier call are not duplicated
    for path in (recorder.events_dir.glob("*_PAST_SUCCESS_*.json") if recorder.events_dir.exists() else []):
        already.add(json.loads(path.read_text(encoding="utf-8")).get("original_steps"))
    for k, row in successes.iterrows():
        steps, seed, level = int(row["num_timesteps"]), int(row["seed"]), int(row["level"])
        if steps in already:
            counts["already_done"] += 1
            continue
        earlier = [c for c in checkpoints if c <= steps]
        ckpt, relation = (earlier[-1], "earlier checkpoint") if earlier else (checkpoints[0], "FIRST checkpoint, which is later than the original episode")
        if ckpt not in models:
            models[ckpt] = load_sb3_model(models_dir / f"checkpoint_{ckpt}.zip", device)
        model = models[ckpt]
        cond = randomizer.sample(level, seed)
        found = None
        for attempt in range(max_tries + 1):
            deterministic = attempt == 0
            if not deterministic:
                model.set_random_seed(seed % 1_000_000 + attempt)
                torch.manual_seed(seed % 1_000_000 + attempt)
            policy = (lambda obs, d=deterministic: model.predict(obs, deterministic=d)[0])
            episode = record_episode(env, policy, seed, level, "", recorder.runway_width_m, conditions=cond)
            if episode["success"]:
                found = (episode, "deterministic" if deterministic else f"stochastic sample #{attempt}")
                break
        if found is None:
            counts["skipped"] += 1
            print(f"[skip] success #{k + 1} (orig. {steps:,}, seed {seed}): not reproduced in {max_tries + 1} tries with checkpoint {ckpt:,}")
            continue
        episode, how = found
        recorder.write_event(
            "PAST_SUCCESS", f"PAST success #{k + 1} (original at {steps:,} transitions, seed {seed}) - RECONSTRUCTED: "
            f"same conditions, checkpoint {ckpt // 1000}k ({relation}), {how}", steps, level, episode,
            extra={"wall_time": base_time + k * 10.0, "reconstructed": True, "original_steps": steps})
        counts["reconstructed"] += 1
        print(f"[ok]   success #{k + 1} (orig. {steps:,}) <- checkpoint {ckpt:,} ({how})")
    return counts


def main() -> None:
    """Backfill / reconstruction command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=str, required=True, help="run name (models/<run>/checkpoint_*.zip)")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1_000_000])
    parser.add_argument("--reconstruct-past", action="store_true",
                        help="re-create events for successes logged before recording existed (see docstring)")
    parser.add_argument("--models-run", type=str, default=None, help="models/<name> to use (default: same as --run)")
    parser.add_argument("--max-tries", type=int, default=40)
    parser.add_argument("--before-steps", type=int, default=None,
                        help="only reconstruct successes up to this many transitions (later ones were recorded live)")
    args = parser.parse_args()
    if args.reconstruct_past:
        print(reconstruct_past_successes(args.run, args.models_run, args.max_tries, before_steps=args.before_steps))
    else:
        print(f"wrote {backfill_checkpoints(args.run, args.level, args.seeds)} checkpoint events for run '{args.run}'")


if __name__ == "__main__":
    main()
