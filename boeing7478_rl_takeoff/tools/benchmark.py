"""Benchmark simulation and training throughput (JSBSim is CPU-bound; networks may use the GPU).

    python tools/benchmark.py                 # physics + env + SAC update benchmark
    python tools/benchmark.py --no-train      # physics + env only (no torch needed)

Measures: physics steps/s, RL environment transitions/s, SAC training updates/s per device,
full learn-loop transitions/s, GPU utilisation (via nvidia-smi when accessible) and a simple
CPU-vs-GPU bottleneck estimate.  Results are written to results/benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from envs import PROJECT_ROOT, load_all_configs, load_yaml  # noqa: E402
from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv  # noqa: E402
from envs.jsbsim_interface import JSBSimInterface  # noqa: E402
from envs.randomization import DomainRandomizer  # noqa: E402


def bench_physics(configs: dict, seconds: float = 3.0) -> dict[str, float]:
    """Raw JSBSim physics steps per second (no Python env logic between steps)."""
    iface = JSBSimInterface(configs["aircraft"], configs["environment"]["physics_hz"], PROJECT_ROOT)
    iface.start_episode(DomainRandomizer(configs["randomization"], configs["aircraft"]).sample(1, 0))
    iface.set_flight_controls(1.0, 0.0, 0.0, 0.0, 0.0)
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        iface.step(120)
        steps += 120
        if iface.read_sensors()["h_agl_ft"] > 3000:
            iface.start_episode(DomainRandomizer(configs["randomization"], configs["aircraft"]).sample(1, 0))
            iface.set_flight_controls(1.0, 0.0, 0.0, 0.0, 0.0)
    elapsed = time.perf_counter() - start
    return {"physics_steps_per_s": steps / elapsed, "simulated_seconds_per_wall_second": steps / elapsed / iface.physics_hz}


def bench_env(configs: dict, seconds: float = 5.0) -> dict[str, float]:
    """RL environment transitions per second with random actions (includes resets)."""
    env = Boeing7478TakeoffEnv(configs=configs)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    n, resets, start = 0, 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        _o, _r, te, tr, _i = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
        n += 1
        if te or tr:
            env.reset()
            resets += 1
    elapsed = time.perf_counter() - start
    return {"env_transitions_per_s": n / elapsed, "episodes_reset": float(resets),
            "ms_per_transition": 1000.0 * elapsed / n}


class GpuSampler:
    """Samples ``nvidia-smi`` utilisation in a background thread (best effort)."""

    def __init__(self) -> None:
        """Prepare an empty sampler."""
        self.util: list[float] = []
        self.mem: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5, check=True).stdout.strip().splitlines()[0]
                util, mem = (float(v) for v in out.split(","))
                self.util.append(util)
                self.mem.append(mem)
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                return  # nvidia-smi not accessible: report as unavailable
            time.sleep(0.5)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=6)

    def summary(self) -> dict[str, float | None]:
        """Mean/max GPU utilisation and peak memory (None when unavailable)."""
        if not self.util:
            return {"gpu_util_mean_pct": None, "gpu_util_max_pct": None, "gpu_mem_peak_mb": None}
        return {"gpu_util_mean_pct": float(np.mean(self.util)), "gpu_util_max_pct": float(np.max(self.util)),
                "gpu_mem_peak_mb": float(np.max(self.mem))}


def bench_training(configs: dict, device: str, update_steps: int = 300, learn_steps: int = 3000) -> dict[str, Any]:
    """SAC gradient updates/s (pure network) and learn-loop transitions/s (env + updates) on ``device``."""
    from stable_baselines3 import SAC
    from training import make_vec_env
    from training.train_sac import build_sac

    sac_cfg = load_yaml(PROJECT_ROOT / "configs" / "sac.yaml")
    sac_cfg["learning_starts"] = 500
    sac_cfg["buffer_size"] = 20_000
    env = make_vec_env(configs, 1, 0, 1, None)
    model: SAC = build_sac(sac_cfg, env, device, PROJECT_ROOT / "logs" / "_bench")
    model.learn(total_timesteps=1000, log_interval=1000)  # fill buffer past learning_starts
    result: dict[str, Any] = {"device": device}
    with GpuSampler() as sampler:
        start = time.perf_counter()
        model.train(gradient_steps=update_steps, batch_size=int(sac_cfg["batch_size"]))
        if device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        upd = update_steps / (time.perf_counter() - start)
    result.update({"updates_per_s": upd, **sampler.summary()})
    start = time.perf_counter()
    model.learn(total_timesteps=learn_steps, reset_num_timesteps=False, log_interval=10_000)
    result["learn_loop_transitions_per_s"] = learn_steps / (time.perf_counter() - start)
    env.close()
    return result


def main() -> None:
    """Run the benchmark and write ``results/benchmark.json``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()
    configs = load_all_configs()
    report: dict[str, Any] = {"physics": bench_physics(configs), "environment": bench_env(configs)}
    print(f"Physics steps/s            : {report['physics']['physics_steps_per_s']:.0f}")
    print(f"Env transitions/s          : {report['environment']['env_transitions_per_s']:.0f} "
          f"({report['environment']['ms_per_transition']:.3f} ms each)")
    if not args.no_train:
        import torch
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        report["training"] = [bench_training(configs, d) for d in devices]
        env_ms = report["environment"]["ms_per_transition"]
        for entry in report["training"]:
            upd_ms = 1000.0 / entry["updates_per_s"]
            entry["bottleneck"] = ("neural-network updates" if upd_ms > env_ms else "JSBSim/CPU environment")
            entry["ms_per_update"] = upd_ms
            print(f"[{entry['device']}] updates/s {entry['updates_per_s']:.1f} ({upd_ms:.2f} ms/update) | "
                  f"learn loop {entry['learn_loop_transitions_per_s']:.1f} transitions/s | GPU util mean "
                  f"{entry['gpu_util_mean_pct']} % | bottleneck: {entry['bottleneck']}")
    out = PROJECT_ROOT / "results" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
