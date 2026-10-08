"""Training callbacks: TensorBoard logging, milestones, evaluation, checkpoints, curriculum, best model.

Terminology: ``num_timesteps`` is the number of RL ENVIRONMENT TRANSITIONS (one policy action
followed by ~12 JSBSim physics steps), never physics steps.
"""
from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv
from evaluation.evaluate import aggregate, evaluate_policy, sb3_policy
from evaluation.trajectory import ShowcaseRecorder, frame_row, record_episode, run_description
from training import RunPaths, set_training_level
from training.curriculum import CurriculumManager

SUCCESS_THRESHOLDS: tuple[float, ...] = (0.10, 0.50, 0.90)
STEP_SCALARS: tuple[str, ...] = ("agl_ft", "airspeed_kt", "vertical_speed_fpm", "pitch_deg", "roll_deg",
                                 "heading_error_deg", "lateral_m", "alpha_deg", "beta_deg")
ACTUATOR_NAMES: tuple[str, ...] = ("throttle", "elevator", "aileron", "rudder")
FAILURE_REASONS: tuple[str, ...] = ("SUCCESS", "RUNWAY_EXCURSION", "FAILED_ROTATION", "INSUFFICIENT_RUNWAY",
                                    "LOSS_OF_CONTROL", "UNSTABLE_AFTER_LIFTOFF", "CRASH", "SIMULATOR_INVALID_STATE",
                                    "ENVELOPE_VIOLATION", "FAILED_TO_ACCELERATE", "UNCLEAN_TAKEOFF", "TIMEOUT")


class MilestoneTracker:
    """Records how many transitions were needed to reach success milestones.

    Training-based milestones use a rolling window of the last ``window`` training episodes
    (a threshold counts only once the window is full).  Evaluation-based milestones use the
    deterministic evaluation success rate.  Unreached milestones stay ``None`` (NOT YET MEASURED).
    """

    def __init__(self, window: int) -> None:
        """Create an empty tracker."""
        self.window = int(window)
        self.recent: deque[bool] = deque(maxlen=self.window)
        self.steps_to_first_success: int | None = None
        self.steps_to_first_success_eval: int | None = None
        self.train_steps_to_pct: dict[str, int | None] = {f"{int(t * 100)}": None for t in SUCCESS_THRESHOLDS}
        self.eval_steps_to_pct: dict[str, int | None] = {f"{int(t * 100)}": None for t in SUCCESS_THRESHOLDS}
        self.first_success_level: int | None = None

    def on_train_episode(self, steps: int, success: bool, level: int) -> bool:
        """Record a finished training episode; returns True if this was the first ever success."""
        first = success and self.steps_to_first_success is None
        if first:
            self.steps_to_first_success = steps
            self.first_success_level = level
        self.recent.append(bool(success))
        if len(self.recent) == self.window:
            rate = sum(self.recent) / self.window
            for threshold in SUCCESS_THRESHOLDS:
                key = f"{int(threshold * 100)}"
                if self.train_steps_to_pct[key] is None and rate >= threshold:
                    self.train_steps_to_pct[key] = steps
        return first

    def on_eval(self, steps: int, success_rate: float) -> None:
        """Record an evaluation result."""
        if success_rate > 0.0 and self.steps_to_first_success_eval is None:
            self.steps_to_first_success_eval = steps
        for threshold in SUCCESS_THRESHOLDS:
            key = f"{int(threshold * 100)}"
            if self.eval_steps_to_pct[key] is None and success_rate >= threshold:
                self.eval_steps_to_pct[key] = steps

    def any_success(self) -> bool:
        """True if any training or evaluation episode has succeeded."""
        return self.steps_to_first_success is not None or self.steps_to_first_success_eval is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-friendly types (``None`` = NOT YET MEASURED)."""
        return {"steps_to_first_success": self.steps_to_first_success,
                "steps_to_first_success_eval": self.steps_to_first_success_eval,
                "first_success_level": self.first_success_level,
                "steps_to_10_percent_success": self.train_steps_to_pct["10"],
                "steps_to_50_percent_success": self.train_steps_to_pct["50"],
                "steps_to_90_percent_success": self.train_steps_to_pct["90"],
                "eval_steps_to_10_percent_success": self.eval_steps_to_pct["10"],
                "eval_steps_to_50_percent_success": self.eval_steps_to_pct["50"],
                "eval_steps_to_90_percent_success": self.eval_steps_to_pct["90"],
                "success_window_episodes": self.window, "unit": "RL environment transitions"}

    def load_dict(self, data: dict[str, Any]) -> None:
        """Restore from :meth:`to_dict` output (resume)."""
        self.steps_to_first_success = data.get("steps_to_first_success")
        self.steps_to_first_success_eval = data.get("steps_to_first_success_eval")
        self.first_success_level = data.get("first_success_level")
        for pct in ("10", "50", "90"):
            self.train_steps_to_pct[pct] = data.get(f"steps_to_{pct}_percent_success")
            self.eval_steps_to_pct[pct] = data.get(f"eval_steps_to_{pct}_percent_success")


class TrainingMonitorCallback(BaseCallback):
    """Logs episode summaries and per-step signals to TensorBoard and a CSV; updates milestones."""

    def __init__(self, paths: RunPaths, milestones: MilestoneTracker, curriculum: CurriculumManager,
                 log_every_steps: int = 2000, showcase: ShowcaseRecorder | None = None) -> None:
        """Create the callback (``showcase``: optional recorder feeding the live 3D viewer)."""
        super().__init__()
        self.showcase = showcase
        self._frames: dict[int, list[list[float]]] = {}
        self.episode_count = 0
        self.success_count = 0
        self.recent_reasons: deque[str] = deque(maxlen=100)
        self.paths = paths
        self.milestones = milestones
        self.curriculum = curriculum
        self.log_every = max(int(log_every_steps), 1)
        self._csv_path = paths.results / "training_episodes.csv"
        self._csv_fields: list[str] | None = None
        self._step_sums: dict[str, float] = {}
        self._step_count = 0
        self._ep_actuator_sums: dict[int, list[float]] = {}
        self._ep_actuator_counts: dict[int, int] = {}

    def _write_row(self, row: dict[str, Any]) -> None:
        """Append one episode row to the CSV (header written once; extra keys are rejected loudly)."""
        write_header = self._csv_fields is None
        if write_header:
            if self._csv_path.exists() and self._csv_path.stat().st_size > 0:
                self._csv_fields = list(pd.read_csv(self._csv_path, nrows=0).columns)
                write_header = False
            else:
                self._csv_fields = list(row.keys())
        extras = [k for k in row if k not in self._csv_fields]
        if extras:   # resuming a run whose CSV was created before these columns existed: keep its layout, say so once
            if not getattr(self, "_schema_warned", False):
                print(f"[warning] {self._csv_path.name} has an older column layout; not writing new columns: {extras[:6]}...")
                self._schema_warned = True
            row = {k: v for k, v in row.items() if k in self._csv_fields}
        with open(self._csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fields, extrasaction="raise")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _on_step(self) -> bool:
        """Process per-env infos from the latest vectorised step."""
        for env_index, info in enumerate(self.locals["infos"]):
            for key in STEP_SCALARS:
                self._step_sums[key] = self._step_sums.get(key, 0.0) + float(info[key])
            acts = info["actuators"]
            sums = self._ep_actuator_sums.setdefault(env_index, [0.0] * 4)
            for i in range(4):
                sums[i] += float(acts[i])
            self._ep_actuator_counts[env_index] = self._ep_actuator_counts.get(env_index, 0) + 1
            self._step_count += 1
            if self.showcase is not None and self.showcase.enabled:
                self._frames.setdefault(env_index, []).append(frame_row(info["time_s"], info))
            summary = info.get("episode_summary")
            if summary is not None:
                self._on_episode_end(env_index, summary)
        if self._step_count >= self.log_every:
            for key, total in self._step_sums.items():
                self.logger.record(f"step/{key}", total / self._step_count)
            self._step_sums.clear()
            self._step_count = 0
            self._write_status()
        return True

    def _write_status(self) -> None:
        """Publish a small live-status file for the 3D viewer's training panel."""
        if self.showcase is None:
            return
        window = len(self.milestones.recent)
        self.showcase.write_status({
            "num_timesteps": int(self.num_timesteps), "level": self.curriculum.level, "episodes": self.episode_count,
            "training_successes": self.success_count,
            "rolling_success_rate": (sum(self.milestones.recent) / window) if window else None, "rolling_window": window,
            "recent_reasons": {r: self.recent_reasons.count(r) for r in set(self.recent_reasons)},
            "steps_to_first_success": self.milestones.steps_to_first_success,
            "eval": self.showcase.shared.get("eval"), "events": self.showcase.counter})

    def _record_training_events(self, env_index: int, summary: dict[str, Any], first_success: bool) -> None:
        """Turn a finished training episode into showcase events (exact trajectory of that episode)."""
        sc = self.showcase
        frames = self._frames.pop(env_index, [])
        if sc is None or not sc.enabled or not frames:
            return
        steps, level = int(self.num_timesteps), int(summary["level"])
        success = bool(summary["success"])
        lifted = not math.isnan(float(summary["liftoff_time_s"]))
        run = run_description(summary, frames, "", sc.runway_width_m)
        if success:
            if first_success:
                sc.write_event("FIRST_SUCCESS", f"FIRST successful takeoff (training) at {steps:,} transitions, level {level}",
                               steps, level, run)
                sc.training_successes_recorded += 1
                sc.last_success_event_step = steps
            elif sc.should_record_training_success(steps):
                sc.write_event("TRAINING_SUCCESS", f"Training success #{self.success_count} at {steps:,} transitions "
                                                   f"(level {level}, takeoff {summary['takeoff_time_s']:.0f} s)", steps, level, run)
                sc.training_successes_recorded += 1
                sc.last_success_event_step = steps
        elif lifted and not sc.seen_first_liftoff:
            sc.write_event("FIRST_LIFTOFF", f"First lift-off at {steps:,} transitions (ended: {summary['reason']})",
                           steps, level, run)
        if lifted:
            sc.seen_first_liftoff = True

    def _on_episode_end(self, env_index: int, summary: dict[str, Any]) -> None:
        """Log one finished training episode."""
        steps = int(self.num_timesteps)
        count = max(self._ep_actuator_counts.pop(env_index, 1), 1)
        act_means = [v / count for v in self._ep_actuator_sums.pop(env_index, [0.0] * 4)]
        success = bool(summary["success"])
        first = self.milestones.on_train_episode(steps, success, int(summary["level"]))
        self.episode_count += 1
        self.success_count += int(success)
        self.recent_reasons.append(str(summary["reason"]))
        self._record_training_events(env_index, summary, first)
        rec = self.logger.record_mean
        rec("episode/total_reward", summary["total_reward"])
        rec("episode/duration_s", summary["duration_s"])
        rec("episode/transitions", summary["transitions"])
        rec("episode/success", float(success))
        rec("episode/final_agl_ft", summary["final_agl_ft"])
        rec("episode/final_airspeed_kt", summary["final_airspeed_kt"])
        rec("episode/final_vertical_speed_fpm", summary["final_vs_fpm"])
        rec("episode/max_pitch_deg", summary["max_pitch_deg"])
        rec("episode/max_roll_deg", summary["max_roll_deg"])
        rec("episode/max_alpha_deg", summary["max_alpha_deg"])
        rec("episode/max_heading_error_deg", summary["max_heading_error_deg"])
        rec("episode/centerline_rms_m", summary["centerline_rms_m"])
        rec("episode/centerline_max_m", summary["centerline_max_m"])
        rec("episode/cleanliness", summary["cleanliness"])
        rec("episode/rms_lateral_accel_mps2", summary["rms_lateral_accel_mps2"])
        rec("episode/rms_yaw_rate_dps", summary["rms_yaw_rate_dps"])
        if success:
            rec("episode/cleanliness_of_successes", summary["cleanliness"])
        rec("episode/mass_lbs", summary["mass_lbs"])
        rec("episode/cg_x_in", summary["cg_x_in"])
        rec("episode/headwind_kt", summary["headwind_kt"])
        rec("episode/crosswind_kt", summary["crosswind_kt"])
        rec("episode/runway_length_m", summary["runway_length_m"])
        rec("episode/runway_friction_factor", summary["static_friction_factor"])
        rec("episode/curriculum_stage", summary["level"])
        for name, value in zip(ACTUATOR_NAMES, act_means):
            rec(f"episode/mean_{name}", value)
        if success:
            rec("episode/takeoff_time_s", summary["takeoff_time_s"])
            rec("episode/runway_distance_used_m", summary["runway_distance_used_m"])
        for key, value in summary.items():
            if key.startswith("rc_"):
                rec(f"reward_components/{key[3:]}", value)
        for reason in FAILURE_REASONS:
            rec(f"outcome/{reason}", float(summary["reason"] == reason))
        row = dict(summary)
        row["num_timesteps"] = steps
        self._write_row(row)
        if first:
            print(f"[milestone] FIRST SUCCESSFUL TAKEOFF (training) at {steps} transitions, level {summary['level']}")
        self.logger.record("milestones/steps_to_first_success",
                           self.milestones.steps_to_first_success if self.milestones.steps_to_first_success is not None else -1)
        (self.paths.results / "milestones.json").write_text(json.dumps(self.milestones.to_dict(), indent=2), encoding="utf-8")


class EvalCheckpointCallback(BaseCallback):
    """Deterministic evaluation, curriculum advancement, checkpoints, best model and no-success guard."""

    def __init__(self, configs: dict[str, dict[str, Any]], train_cfg: dict[str, Any], paths: RunPaths,
                 curriculum: CurriculumManager, milestones: MilestoneTracker,
                 showcase: ShowcaseRecorder | None = None) -> None:
        """Create the callback from the training YAML (``evaluation`` / ``curriculum`` sections)."""
        super().__init__()
        self.showcase = showcase
        self.best_rate_by_level: dict[int, float] = {}
        self.eval_success_seen = False
        ev = train_cfg["evaluation"]
        self.configs = configs
        self.paths = paths
        self.curriculum = curriculum
        self.milestones = milestones
        self.eval_freq = int(ev["eval_freq"])
        self.n_eval = int(ev["n_eval_episodes"])
        self.seed_base = int(ev["eval_seed_base"])
        self.ckpt_steps = sorted(int(s) for s in ev["checkpoint_steps"])
        self.ckpt_eval_episodes = int(ev["checkpoint_eval_episodes"])
        self.periodic_freq = int(ev["periodic_checkpoint_freq"])
        self.save_buffer = bool(ev["save_replay_buffer"])
        guard = ev["stop_if_no_success"]
        self.guard_enabled = bool(guard["enabled"])
        self.guard_steps = int(guard["after_steps"])
        self.next_eval = self.eval_freq
        self.next_periodic = self.periodic_freq
        self.pending_ckpts = list(self.ckpt_steps)
        self.best_key: tuple[float, float, float] = (-1.0, -1.0, -1e18)
        self.eval_env: Boeing7478TakeoffEnv | None = None
        self.stop_reason: str | None = None
        self.last_eval: dict[str, float] = {}

    # ------------------------------------------------------------------ lifecycle
    def _on_training_start(self) -> None:
        """Create the separate evaluation environment and align the schedule with a resumed run."""
        self.eval_env = Boeing7478TakeoffEnv(configs=self.configs)
        self.eval_success_seen = self.milestones.steps_to_first_success_eval is not None
        history = self.paths.results / "eval_history.csv"
        if history.exists():   # resume: do not re-announce progress that was already recorded
            past = pd.read_csv(history)
            past = past[(past["tag"] == "periodic") & (past["num_timesteps"] <= int(self.num_timesteps))]
            for level, group in past.groupby("level"):
                self.best_rate_by_level[int(level)] = float(group["success_rate"].max())
        n = int(self.num_timesteps)
        self.next_eval = (n // self.eval_freq + 1) * self.eval_freq
        self.next_periodic = (n // self.periodic_freq + 1) * self.periodic_freq
        every = self.showcase.update_every if self.showcase is not None else 0
        self.next_update = ((n // every + 1) * every) if every > 0 else None
        self.pending_ckpts = [s for s in self.ckpt_steps if s > n]

    def _on_training_end(self) -> None:
        """Close the evaluation environment."""
        if self.eval_env is not None:
            self.eval_env.close()

    # ------------------------------------------------------------------ actions
    def save_checkpoint(self, steps: int, name: str | None = None) -> Path:
        """Save model (+ optional replay buffer + curriculum/milestone state) and return the model path."""
        path = self.paths.models / (name or f"checkpoint_{steps}")
        self.model.save(str(path))
        if self.save_buffer and hasattr(self.model, "save_replay_buffer"):
            self.model.save_replay_buffer(str(path) + "_replay_buffer.pkl")
        state = {"num_timesteps": steps, "curriculum": self.curriculum.state_dict(),
                 "milestones": self.milestones.to_dict(), "reward_profile": self.configs["reward"].get("profile", "clean")}
        (Path(str(path) + "_state.json")).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return Path(str(path) + ".zip")

    def evaluate(self, episodes: int, tag: str) -> tuple[dict[str, float], pd.DataFrame]:
        """Run a deterministic evaluation at the CURRENT curriculum level."""
        assert self.eval_env is not None
        df = evaluate_policy(sb3_policy(self.model), self.configs, episodes, self.curriculum.level, self.seed_base,
                             env=self.eval_env)
        agg = aggregate(df)
        agg.update({"num_timesteps": float(self.num_timesteps), "level": float(self.curriculum.level)})
        self.last_eval = agg
        for key in ("success_rate", "mean_takeoff_time_s", "mean_runway_distance_m", "mean_centerline_rms_m",
                    "mean_max_roll_deg", "mean_max_pitch_deg", "mean_max_alpha_deg", "mean_return"):
            if not math.isnan(agg[key]):
                self.logger.record(f"eval/{key}", agg[key])
        self.logger.record("eval/curriculum_level", self.curriculum.level)
        return agg, df

    def _log_eval_csv(self, agg: dict[str, float], df: pd.DataFrame, tag: str) -> None:
        """Append aggregate row to ``eval_history.csv`` and (for checkpoints) write the per-episode CSV."""
        row = {"tag": tag, **agg}
        for reason, count in df["reason"].value_counts().items():
            row[f"n_{reason}"] = int(count)
        hist = self.paths.results / "eval_history.csv"
        frame = pd.DataFrame([row])
        if hist.exists():
            frame = pd.concat([pd.read_csv(hist), frame], ignore_index=True)
        frame.to_csv(hist, index=False)
        if tag.startswith("checkpoint"):
            df.to_csv(self.paths.results / f"{tag}_episodes.csv", index=False)

    def _record_showcase(self, event_type: str, headline: str, level: int, snapshot: bool = True) -> None:
        """Record deterministic replays of the CURRENT policy as showcase events (and a model snapshot)."""
        sc = self.showcase
        if sc is None or not sc.enabled or self.eval_env is None:
            return
        steps = int(self.num_timesteps)
        policy = sb3_policy(self.model)
        n = max(sc.eval_episodes, 1)
        for i in range(n):
            episode = record_episode(self.eval_env, policy, self.seed_base + i, level, "", sc.runway_width_m)
            sc.write_event(event_type, f"{headline} - deterministic replay {i + 1}/{n}", steps, level, episode)
        if snapshot and sc.save_snapshots:
            self.model.save(str(self.paths.models / f"showcase_{steps}_{event_type}"))
        print(f"[showcase] {event_type}: {headline}")

    def _step_update(self, steps: int) -> None:
        """Periodic showcase update (every ``showcase.update_every_steps``): replays of the CURRENT policy + a headline."""
        sc = self.showcase
        if sc is None or not sc.enabled or self.eval_env is None:
            return
        level, policy = int(self.curriculum.level), sb3_policy(self.model)
        recent = self.milestones.recent
        train_rate = f"{sum(recent) / len(recent):.0%}" if recent else "n/a"
        ev = sc.shared.get("eval")
        eval_txt = f"{ev['success_rate']:.0%} @{ev['num_timesteps'] // 1000}k" if ev else "n/a"
        n = max(sc.update_episodes, 1)
        for i in range(n):
            episode = record_episode(self.eval_env, policy, self.seed_base + i, level, "", sc.runway_width_m)
            verdict = "SUCCESS" if episode["success"] else episode["reason"]
            clean = f", clean {episode['cleanliness']:.2f}" if "cleanliness" in episode else ""
            sc.write_event("STEP_UPDATE", f"Update {steps:,} transitions: replay {i + 1}/{n} {verdict}{clean} | "
                                          f"train success (last {len(recent)}) {train_rate} | eval {eval_txt} | level {level}",
                           steps, level, episode)
        print(f"[showcase] STEP_UPDATE at {steps:,} transitions")

    def _detect_progress(self, steps: int, agg: dict[str, float]) -> None:
        """Flag 'big progress' on a periodic evaluation (before the curriculum may change the level)."""
        sc = self.showcase
        level, rate = int(self.curriculum.level), float(agg["success_rate"])
        previous = self.best_rate_by_level.get(level, 0.0)
        if sc is not None:
            sc.shared["eval"] = {"num_timesteps": steps, "level": level, "success_rate": rate,
                                 "mean_return": agg["mean_return"], "episodes": int(agg["episodes"])}
        if rate > 0.0 and not self.eval_success_seen:
            self._record_showcase("EVAL_FIRST_SUCCESS", f"FIRST evaluation success: {rate:.0%} at level {level}, {steps:,} transitions", level)
        elif rate >= previous + (sc.progress_delta if sc is not None else 0.1) and rate > 0.0:
            self._record_showcase("BIG_PROGRESS", f"Big progress: evaluation success {rate:.0%} (previous best {previous:.0%}) "
                                                  f"at level {level}, {steps:,} transitions", level)
        self.eval_success_seen = self.eval_success_seen or rate > 0.0
        self.best_rate_by_level[level] = max(previous, rate)

    def _periodic_eval(self, steps: int) -> None:
        """Evaluate, update milestones/curriculum, save best model."""
        agg, df = self.evaluate(self.n_eval, "periodic")
        self._log_eval_csv(agg, df, "periodic")
        self._detect_progress(steps, agg)
        self.milestones.on_eval(steps, agg["success_rate"])
        key = (float(self.curriculum.level), agg["success_rate"], agg["mean_return"])
        if key > self.best_key:
            self.best_key = key
            self.model.save(str(self.paths.models / "best_model"))
        print(f"[eval @ {steps}] level {self.curriculum.level} success {agg['success_rate']:.2f} "
              f"return {agg['mean_return']:.1f}")
        if self.curriculum.update(agg["success_rate"], self.n_eval, steps):
            set_training_level(self.training_env, self.curriculum.level)
            self.best_key = (-1.0, -1.0, -1e18)  # a harder level: best-model comparison restarts
            print(f"[curriculum] advanced to LEVEL {self.curriculum.level} at {steps} transitions")
            self._record_showcase("LEVEL_UP", f"Curriculum advanced to level {self.curriculum.level} at {steps:,} transitions",
                                  self.curriculum.level)
        (self.paths.results / "milestones.json").write_text(json.dumps(self.milestones.to_dict(), indent=2), encoding="utf-8")
        (self.paths.results / "curriculum_state.json").write_text(json.dumps(self.curriculum.state_dict(), indent=2),
                                                                  encoding="utf-8")

    def _on_step(self) -> bool:
        """Run scheduled evaluation/checkpointing; return False to stop training."""
        n = int(self.num_timesteps)
        if n >= self.next_eval:
            self.next_eval = (n // self.eval_freq + 1) * self.eval_freq
            self._periodic_eval(n)
        if n >= self.next_periodic:
            self.next_periodic = (n // self.periodic_freq + 1) * self.periodic_freq
            self.save_checkpoint(n)
        if self.next_update is not None and n >= self.next_update:   # every 50k transitions (configurable)
            self.next_update = (n // self.showcase.update_every + 1) * self.showcase.update_every
            self._step_update(n)
        while self.pending_ckpts and n >= self.pending_ckpts[0]:
            milestone = self.pending_ckpts.pop(0)
            path = self.save_checkpoint(n, f"checkpoint_{milestone}")
            agg, df = self.evaluate(self.ckpt_eval_episodes, f"checkpoint_{milestone}")
            self._log_eval_csv(agg, df, f"checkpoint_{milestone}")
            self.milestones.on_eval(n, agg["success_rate"])
            covered = self.showcase is not None and self.showcase.update_every > 0 and milestone % self.showcase.update_every == 0
            if self.showcase is not None and self.showcase.enabled and self.eval_env is not None and not covered:
                episode = record_episode(self.eval_env, sb3_policy(self.model), self.seed_base, self.curriculum.level, "",
                                         self.showcase.runway_width_m)
                self.showcase.write_event("CHECKPOINT", f"Checkpoint {milestone:,} transitions: eval success {agg['success_rate']:.0%}"
                                          f" (replay: {episode['reason']})", n, self.curriculum.level, episode)
            print(f"[checkpoint {milestone}] saved {path.name} | eval success {agg['success_rate']:.2f} "
                  f"({self.ckpt_eval_episodes} episodes, level {self.curriculum.level})")
        if self.guard_enabled and n >= self.guard_steps and not self.milestones.any_success():
            self.stop_reason = (f"No successful takeoff (training or evaluation) after {n} transitions. Stopping instead of "
                                "running millions of steps blindly. Investigate: reward, action mapping, observation "
                                "normalisation, aircraft configuration, runway configuration, JSBSim properties, "
                                "termination conditions, rotation behaviour, environment bugs. "
                                "(Disable via evaluation.stop_if_no_success.enabled.)")
            print("[WARNING] " + self.stop_reason)
            return False
        return True
