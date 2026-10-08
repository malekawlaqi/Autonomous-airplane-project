"""Live 3D training showcase: serves the 3D viewer and auto-updates it as training produces events.

    python tools/live_viewer.py --open            # then start/continue training in another terminal

While a training run is active (``python -m training.train_sac ...``) it writes *events* under
``results/<run>/showcase/events/``: every training success (the exact trajectory of that episode), first lift-off,
first evaluation success, big evaluation progress, curriculum level-ups and periodic checkpoint replays.
This server publishes them; the page polls every 3 s, adds new events to its playlist, shows a banner and (with
"Auto-follow newest" ticked) plays them immediately.  It also shows a live training-status panel.

Backfill history from existing checkpoints:   python -m evaluation.trajectory --run sac_pilot
Only binds to localhost.  Three.js is loaded from the unpkg CDN (internet needed in the browser).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PROJECT_ROOT  # noqa: E402
from tools.replay_3d import render_page  # noqa: E402

DEFAULT_GLB = PROJECT_ROOT / "results" / "replay3d" / "boeing747erf.glb"


class EventIndex:
    """Scans ``results/*/showcase/events/*.json`` and caches lightweight metadata (frames are not loaded)."""

    def __init__(self, results_dir: Path) -> None:
        """Create an index rooted at ``results_dir``."""
        self.results_dir = results_dir
        self._cache: dict[Path, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def list_events(self) -> list[dict]:
        """Return event metadata sorted by creation time (oldest first)."""
        events: list[dict] = []
        with self._lock:
            for path in self.results_dir.glob("*/showcase/events/*.json"):
                mtime = path.stat().st_mtime
                cached = self._cache.get(path)
                if cached is None or cached[0] != mtime:
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        continue   # file is still being written; it will be picked up on the next poll
                    run = path.parents[2].name
                    meta = {"event_id": f"{run}/{path.name}", "url": "/" + path.relative_to(self.results_dir).as_posix(),
                            "event_type": data.get("event_type"), "headline": data.get("headline"), "run_name": run,
                            "num_timesteps": data.get("num_timesteps"), "wall_time": data.get("wall_time", mtime),
                            "success": data.get("success")}
                    self._cache[path] = (mtime, meta)
                events.append(self._cache[path][1])
        return sorted(events, key=lambda e: (e["wall_time"], e["event_id"]))

    def statuses(self) -> dict[str, dict]:
        """Return the latest ``status.json`` of every run that has one."""
        out: dict[str, dict] = {}
        for path in self.results_dir.glob("*/showcase/status.json"):
            try:
                out[path.parents[1].name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue   # being replaced atomically; the next poll will succeed
        return out


class Handler(SimpleHTTPRequestHandler):
    """Serves the viewer page, JSON APIs, the GLB model and the event files (read-only, results/ only)."""

    index: EventIndex
    page: bytes
    glb_path: Path

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """Route requests."""
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(self.page, "text/html; charset=utf-8")
        elif path == "/api/events":
            self._send(json.dumps(self.index.list_events()).encode(), "application/json")
        elif path == "/api/status":
            self._send(json.dumps(self.index.statuses()).encode(), "application/json")
        elif path == "/model.glb":
            if self.glb_path.is_file():
                self._send(self.glb_path.read_bytes(), "model/gltf-binary")
            else:
                self._send(b"no model", "text/plain", HTTPStatus.NOT_FOUND)
        else:
            super().do_GET()   # static files, restricted to the results/ directory

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence per-request logging (the page polls every few seconds)."""


def main() -> None:
    """Start the server."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--glb", type=str, default=str(DEFAULT_GLB), help="aircraft model served at /model.glb")
    parser.add_argument("--glb-cg-from-nose", type=float, default=28.0)
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    Handler.index = EventIndex(results_dir)
    Handler.page = render_page([], "", args.glb_cg_from_nose, live=True).encode("utf-8")
    Handler.glb_path = Path(args.glb)
    handler = functools.partial(Handler, directory=str(results_dir))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Live 3D showcase on {url}  (events from {results_dir}/*/showcase/events)  - Ctrl+C to stop")
    print(f"aircraft model: {args.glb if Path(args.glb).is_file() else 'NOT FOUND - using the simple built-in model'}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
