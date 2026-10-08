"""Tests for the live 3D showcase pipeline (event recording, rate limiting, callbacks hooks, HTTP server)."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from envs import load_all_configs
from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv
from evaluation.trajectory import ShowcaseRecorder, frame_row, record_episode, run_description
from teacher.simulated_teacher import SimulatedTeacher

CFG = load_all_configs()


@pytest.fixture(scope="module")
def teacher_episode() -> dict:
    env = Boeing7478TakeoffEnv(configs=CFG)
    return record_episode(env, SimulatedTeacher(CFG).act, 1_000_000, 1, "teacher test")


def test_record_episode_schema(teacher_episode: dict) -> None:
    ep = teacher_episode
    assert ep["success"] is True and ep["reason"] == "SUCCESS"
    frames = ep["frames"]
    assert frames[0][0] == 0.0 and all(len(f) == 13 for f in frames)
    assert all(b[0] > a[0] for a, b in zip(frames, frames[1:]))
    assert max(f[3] for f in frames) > 30.0   # climbed (height in metres)
    for key in ("runway_length_m", "runway_width_m", "vr_kt", "info", "cg_height_m", "duration_s"):
        assert key in ep


def test_recorder_writes_events_and_counts_on_resume(tmp_path: Path, teacher_episode: dict) -> None:
    rec = ShowcaseRecorder("unit", tmp_path, {"enabled": True})
    path = rec.write_event("FIRST_SUCCESS", "first!", 12_345, 1, teacher_episode)
    assert path is not None and path.exists() and path.parent == tmp_path / "showcase" / "events"
    data = json.loads(path.read_text())
    assert data["event_type"] == "FIRST_SUCCESS" and data["headline"] == "first!" and data["num_timesteps"] == 12_345
    assert data["frames"] == teacher_episode["frames"]
    rec.write_status({"num_timesteps": 5})
    assert json.loads((tmp_path / "showcase" / "status.json").read_text())["run_name"] == "unit"
    again = ShowcaseRecorder("unit", tmp_path, {"enabled": True})   # simulated resume
    assert again.counter == 1 and again.training_successes_recorded == 1


def test_success_events_are_rate_limited_after_the_first_n(tmp_path: Path) -> None:
    rec = ShowcaseRecorder("rl", tmp_path, {"training_success_always_until": 3, "training_success_min_gap_steps": 1000})
    for step in (10, 20, 30):
        assert rec.should_record_training_success(step)
        rec.training_successes_recorded += 1
        rec.last_success_event_step = step
    assert not rec.should_record_training_success(500)    # too soon after the last one
    assert rec.should_record_training_success(2000)


def test_disabled_and_event_cap(tmp_path: Path, teacher_episode: dict) -> None:
    assert ShowcaseRecorder("off", tmp_path, {"enabled": False}).write_event("X", "h", 1, 1, teacher_episode) is None
    capped = ShowcaseRecorder("cap", tmp_path, {"enabled": True, "max_events": 2})
    assert capped.write_event("A", "h", 1, 1, teacher_episode) and capped.write_event("B", "h", 2, 1, teacher_episode)
    assert capped.write_event("C", "h", 3, 1, teacher_episode) is None


def _callback(tmp_path: Path, steps: int):
    from training import RunPaths
    from training.callbacks import MilestoneTracker, TrainingMonitorCallback
    from training.curriculum import CurriculumManager
    paths = RunPaths(models=tmp_path / "m", logs=tmp_path / "l", results=tmp_path)
    rec = ShowcaseRecorder("cb", tmp_path, {"enabled": True})
    cb = TrainingMonitorCallback(paths, MilestoneTracker(100), CurriculumManager(
        {"enabled": True, "success_threshold": 0.8, "min_eval_episodes": 5, "max_level": 4}), 2000, rec)
    cb.model = SimpleNamespace(num_timesteps=steps)
    cb.num_timesteps = steps   # SB3 refreshes this attribute inside on_step(); unit tests call the hooks directly
    return cb, rec


def _training_episode(policy):
    env = Boeing7478TakeoffEnv(configs=CFG)
    obs, _ = env.reset(seed=77, options={"level": 1})
    frames = []
    while True:
        obs, _r, te, tr, info = env.step(policy(obs))
        frames.append(frame_row(info["time_s"], info))
        if te or tr:
            return info["episode_summary"], frames


def test_training_success_creates_first_success_event_with_exact_trajectory(tmp_path: Path) -> None:
    cb, rec = _callback(tmp_path, 41_000)
    summary, frames = _training_episode(SimulatedTeacher(CFG).act)
    assert summary["success"]
    cb._frames[0] = frames
    cb._record_training_events(0, summary, first_success=True)
    files = sorted((tmp_path / "showcase" / "events").glob("*FIRST_SUCCESS*.json"))
    assert len(files) == 1
    event = json.loads(files[0].read_text())
    assert event["success"] is True and "41,000" in event["headline"]
    assert event["frames"][1:] == frames    # exact recorded frames (plus a prepended t=0 frame)
    cb._frames[0] = list(frames)
    cb.success_count = 2
    cb._record_training_events(0, summary, first_success=False)
    assert len(list((tmp_path / "showcase" / "events").glob("*TRAINING_SUCCESS*.json"))) == 1


def test_first_liftoff_is_announced_once(tmp_path: Path) -> None:
    cb, rec = _callback(tmp_path, 9_000)
    summary, frames = _training_episode(lambda o: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    summary = dict(summary, success=False, reason="UNSTABLE_AFTER_LIFTOFF", liftoff_time_s=30.0)
    for _ in range(2):
        cb._frames[0] = list(frames)
        cb._record_training_events(0, summary, first_success=False)
    assert len(list((tmp_path / "showcase" / "events").glob("*FIRST_LIFTOFF*.json"))) == 1


@pytest.fixture()
def live_server(tmp_path: Path, teacher_episode: dict):
    from tools.live_viewer import EventIndex, Handler
    from tools.replay_3d import render_page
    results = tmp_path / "res"
    recorder = ShowcaseRecorder("runA", results / "runA", {"enabled": True})
    recorder.write_event("FIRST_SUCCESS", "hello", 100, 1, teacher_episode)
    recorder.write_status({"num_timesteps": 100, "level": 1})
    (tmp_path / "secret.txt").write_text("x")   # outside the served directory
    Handler.index = EventIndex(results)
    Handler.page = render_page([], "", 28.0, live=True).encode()
    Handler.glb_path = tmp_path / "missing.glb"
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(results)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", results
    server.shutdown()


def test_live_server_api_and_security(live_server) -> None:
    base, results = live_server
    get = lambda p: urllib.request.urlopen(base + p, timeout=10)   # noqa: E731
    page = get("/").read().decode()
    assert "const LIVE = true" in page and "api/events" in page
    events = json.loads(get("/api/events").read())
    assert len(events) == 1 and events[0]["event_type"] == "FIRST_SUCCESS" and events[0]["run_name"] == "runA"
    event = json.loads(get(events[0]["url"]).read())
    assert event["headline"] == "hello" and len(event["frames"]) > 100
    assert json.loads(get("/api/status").read())["runA"]["num_timesteps"] == 100
    with pytest.raises(urllib.error.HTTPError) as err:
        get("/model.glb")
    assert err.value.code == 404
    for evil in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%5csecret.txt"):
        try:
            body = get(evil).read()
        except urllib.error.HTTPError:
            continue
        assert body != b"x", f"directory traversal succeeded via {evil}"


def test_static_page_still_renders_without_live() -> None:
    from tools.replay_3d import render_page
    page = render_page([{"label": "x"}], "", 28.0, live=False)
    assert "const LIVE = false" in page and "__RUNS__" not in page and "__GLB__" not in page and "__LIVE__" not in page


def test_run_description_uses_summary_fields() -> None:
    summary = {"reason": "CRASH", "success": False, "duration_s": 12.34, "runway_length_m": 4000.0, "vr_est_kt": 140.0,
               "mass_lbs": 700000.0, "crosswind_kt": 3.0, "headwind_kt": 1.0, "static_friction_factor": 0.9,
               "runway_elevation_ft": 100.0, "gust_sigma_kt": 0.0, "level": 2}
    run = run_description(summary, [[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "lbl", 60.0)
    assert run["frames"][0][0] == 0.0 and len(run["frames"]) == 2 and run["success"] is False and "L2" in run["info"]
