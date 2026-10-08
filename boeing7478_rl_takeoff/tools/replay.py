"""Visual replay of one episode (SEPARATE from training - nothing is rendered during training).

Examples:
    python tools/replay.py --policy teacher --seed 3 --level 3                    # PNG dashboard
    python tools/replay.py --model models/sac_pilot/best_model.zip --gif          # animated GIF
    python tools/replay.py --model models/sac_pilot/checkpoint_100000.zip --live  # live matplotlib window

Outputs go to results/replay/.  The replay runs one deterministic episode with the chosen policy and
draws: top-down runway view, side view, airspeed (with estimated VR/V2), attitude, controls and
cumulative reward components.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

from envs import PROJECT_ROOT, load_all_configs  # noqa: E402
from envs.boeing7478_takeoff_env import Boeing7478TakeoffEnv  # noqa: E402
from envs.reward import COMPONENT_NAMES  # noqa: E402
from evaluation.evaluate import build_policy  # noqa: E402


def rollout(env: Boeing7478TakeoffEnv, policy, seed: int, level: int) -> tuple[dict[str, np.ndarray], dict]:
    """Run one episode and return time series plus the episode summary."""
    obs, reset_info = env.reset(seed=seed, options={"level": level})
    keys = ("time_s", "along_m", "lateral_m", "agl_ft", "airspeed_kt", "vertical_speed_fpm", "pitch_deg", "roll_deg",
            "heading_error_deg", "alpha_deg", "beta_deg")
    series: dict[str, list] = {k: [] for k in keys}
    series.update({"act": [], "reward": [], "comp": []})
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        for k in keys:
            series[k].append(info[k])
        series["act"].append(info["actuators"])
        series["reward"].append(reward)
        series["comp"].append([info["reward_components"][n] for n in COMPONENT_NAMES])
        if terminated or truncated:
            summary = info["episode_summary"]
            summary["vr_est_kt"], summary["v2_est_kt"] = reset_info["vr_est_kt"], reset_info["vr_est_kt"] / 1.08 * 1.15
            return {k: np.asarray(v) for k, v in series.items()}, summary


def draw_dashboard(fig, s: dict[str, np.ndarray], summary: dict, runway_len: float, half_w: float, title: str):
    """Draw all static panels; returns the axes dictionary."""
    axs = fig.subplot_mosaic([["top", "top", "side"], ["speed", "att", "ctrl"], ["rew", "rew", "rew"]])
    t, x, y = s["time_s"], s["along_m"], s["lateral_m"]
    ax = axs["top"]
    ax.axhspan(-half_w, half_w, color="0.85", label="runway")
    ax.axhline(0, color="w", ls="--", lw=1)
    ax.axvline(runway_len, color="r", lw=1)
    ax.plot(x, y, "b-", lw=1.5)
    ax.set(title="Top-down (runway frame)", xlabel="along runway [m]", ylabel="lateral [m] (+ right)",
           ylim=(-max(60, np.abs(y).max() * 1.1), max(60, np.abs(y).max() * 1.1)))
    ax = axs["side"]
    ax.plot(x, s["agl_ft"], "g-")
    ax.axhline(100, color="k", ls=":", label="100 ft")
    ax.set(title="Side view", xlabel="along runway [m]", ylabel="AGL [ft]")
    ax.legend(loc="upper left")
    ax = axs["speed"]
    ax.plot(t, s["airspeed_kt"], label="KCAS")
    ax.axhline(summary["vr_est_kt"], color="orange", ls="--", label="VR est")
    ax.axhline(summary["v2_est_kt"], color="r", ls="--", label="V2 est")
    ax.set(title="Airspeed", xlabel="t [s]", ylabel="kt")
    ax.legend(fontsize=7)
    ax = axs["att"]
    ax.plot(t, s["pitch_deg"], label="pitch")
    ax.plot(t, s["roll_deg"], label="roll")
    ax.plot(t, np.where(s["airspeed_kt"] >= 50.0, s["alpha_deg"], np.nan), label="AoA (>50 kt)", alpha=0.6)
    ax.plot(t, s["heading_error_deg"], label="hdg err", alpha=0.6)
    ax.set(title="Attitude [deg]", xlabel="t [s]")
    ax.legend(fontsize=7)
    ax = axs["ctrl"]
    for i, name in enumerate(("throttle", "elevator", "aileron", "rudder")):
        ax.plot(t, s["act"][:, i], label=name)
    ax.set(title="Controls (actuator positions)", xlabel="t [s]", ylim=(-1.05, 1.05))
    ax.legend(fontsize=7, ncol=2)
    ax = axs["rew"]
    cum = np.cumsum(s["comp"], axis=0)
    for i, name in enumerate(COMPONENT_NAMES):
        if np.abs(cum[:, i]).max() > 0.5:
            ax.plot(t, cum[:, i], label=name)
    ax.plot(t, np.cumsum(s["reward"]), "k", lw=2, label="TOTAL")
    ax.set(title="Cumulative reward components", xlabel="t [s]")
    ax.legend(fontsize=7, ncol=5)
    for a in axs.values():
        a.grid(alpha=0.3)
    fig.suptitle(title)
    return axs


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--policy", type=str, default="teacher", choices=["teacher", "full_throttle", "random"])
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--gif", action="store_true", help="also write an animated GIF")
    parser.add_argument("--live", action="store_true", help="open an interactive window (needs a GUI backend)")
    parser.add_argument("--out", type=str, default=None, help="output file stem (default results/replay/<label>_seed<N>)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    if not args.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = load_all_configs()
    policy = build_policy(args.policy, args.model, configs, args.device)
    env = Boeing7478TakeoffEnv(configs=configs)
    label = Path(args.model).parent.name + "_" + Path(args.model).stem if args.model else args.policy
    series, summary = rollout(env, policy, args.seed, args.level)
    rw = env.conditions.runway  # type: ignore[union-attr]
    title = (f"{label} | seed {args.seed} L{args.level} | {summary['reason']} | {summary['duration_s']:.0f} s | "
             f"{summary['mass_lbs'] / 1000:.0f}k lb, xwind {summary['crosswind_kt']:+.0f} kt, hwind {summary['headwind_kt']:+.0f} kt, "
             f"runway {rw.length_m:.0f} m, mu x{rw.static_friction_factor:.2f}")
    stem = Path(args.out) if args.out else PROJECT_ROOT / "results" / "replay" / f"{label}_seed{args.seed}"
    stem.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    axs = draw_dashboard(fig, series, summary, rw.length_m, rw.half_width_m, title)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=110)
    print(f"{title}\nwrote {stem}.png")

    if args.gif or args.live:
        from matplotlib.animation import FuncAnimation, PillowWriter
        step = 5  # one frame per 0.5 s of simulated time
        idx = list(range(0, len(series["time_s"]), step)) + [len(series["time_s"]) - 1]
        marks = [axs["top"].plot([], [], "ro", ms=9)[0], axs["side"].plot([], [], "ro", ms=9)[0]]
        cursors = [axs[k].axvline(0, color="r", lw=1) for k in ("speed", "att", "ctrl", "rew")]

        def update(frame: int):
            i = idx[frame]
            marks[0].set_data([series["along_m"][i]], [series["lateral_m"][i]])
            marks[1].set_data([series["along_m"][i]], [series["agl_ft"][i]])
            for c in cursors:
                c.set_xdata([series["time_s"][i]] * 2)
            return marks + cursors

        anim = FuncAnimation(fig, update, frames=len(idx), interval=100, blit=False)
        if args.gif:
            anim.save(f"{stem}.gif", writer=PillowWriter(fps=10))
            print(f"wrote {stem}.gif ({len(idx)} frames, 0.5 s of flight per frame)")
        if args.live:
            plt.show()


if __name__ == "__main__":
    main()
