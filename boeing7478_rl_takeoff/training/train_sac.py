"""Train the SAC takeoff controller (primary algorithm).

Usage:
    python -m training.train_sac --config configs/sac.yaml
    python -m training.train_sac --config configs/sac.yaml --resume models/sac/checkpoint_500000.zip
    python -m training.train_sac --config configs/sac.yaml --demos results/demos/teacher_demos.npz --bc-epochs 20

Progress/units: ``total_timesteps`` and every logged "step" count RL ENVIRONMENT TRANSITIONS.
Monitor with ``tensorboard --logdir logs``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402

from envs import PROJECT_ROOT, load_all_configs, load_yaml  # noqa: E402
from training import (activation_from_name, make_vec_env, prepare_run_dirs, select_device,  # noqa: E402
                      set_training_level)
from training.callbacks import EvalCheckpointCallback, MilestoneTracker, TrainingMonitorCallback  # noqa: E402
from training.curriculum import CurriculumManager  # noqa: E402
from evaluation.trajectory import ShowcaseRecorder  # noqa: E402


def build_sac(cfg: dict[str, Any], env: Any, device: str, log_dir: Path) -> SAC:
    """Create a SAC model from the YAML configuration (ReLU 3x256 actor/critic by default)."""
    pol = cfg["policy"]
    policy_kwargs = {"net_arch": {"pi": list(pol["actor_layers"]), "qf": list(pol["critic_layers"])},
                     "activation_fn": activation_from_name(pol["activation"])}
    return SAC("MlpPolicy", env, learning_rate=float(cfg["learning_rate"]), buffer_size=int(cfg["buffer_size"]),
               learning_starts=int(cfg["learning_starts"]), batch_size=int(cfg["batch_size"]), tau=float(cfg["tau"]),
               gamma=float(cfg["gamma"]), train_freq=int(cfg["train_freq"]), gradient_steps=int(cfg["gradient_steps"]),
               ent_coef=cfg["ent_coef"], policy_kwargs=policy_kwargs, tensorboard_log=str(log_dir),
               seed=int(cfg["seed"]), device=device, verbose=0)


def parse_args(description: str, default_config: str, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line shared by the SAC and PPO entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config, help="training YAML")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint .zip to continue from")
    parser.add_argument("--total-timesteps", type=int, default=None, help="absolute transition target (overrides YAML)")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--demos", type=str, default=None, help="teacher demonstrations .npz (SAC only)")
    parser.add_argument("--bc-epochs", type=int, default=None, help="behaviour-cloning epochs on the demos (SAC only)")
    parser.add_argument("--subproc", action="store_true", help="use subprocess vectorisation when n_envs > 1")
    parser.add_argument("--reward-profile", type=str, default=None,
                        help="reward profile from reward.yaml: 'clean' (passenger comfort, default) or 'baseline' (original reward)")
    parser.add_argument("--allow-profile-change", action="store_true",
                        help="allow resuming a checkpoint that was trained with a different reward profile")
    return parser.parse_args(argv)


def run_training(algorithm: str, model_cls: Any, build_model: Callable[[dict[str, Any], Any, str, Path], Any],
                 args: argparse.Namespace) -> int:
    """Shared training driver (create env, model, callbacks; train; checkpoint; handle Ctrl+C).

    Returns:
        Process exit code (0 normal, 130 interrupted, 2 stopped by the no-success guard).
    """
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_yaml(config_path)
    for key, value in (("run_name", args.run_name), ("device", args.device), ("seed", args.seed),
                       ("total_timesteps", args.total_timesteps)):
        if value is not None:
            cfg[key] = value
    configs = load_all_configs(config_path.parent)
    if args.reward_profile is not None:
        configs["reward"]["profile"] = args.reward_profile
    paths = prepare_run_dirs(cfg)
    print(f"reward profile    : {configs['reward'].get('profile', 'clean')}")
    print(f"=== {algorithm} training | run '{cfg['run_name']}' | target {int(cfg['total_timesteps'])} transitions ===")
    device = select_device(str(cfg["device"]))

    curriculum = CurriculumManager(cfg["curriculum"], start_level=int(configs["environment"]["initial_curriculum_level"]))
    milestones = MilestoneTracker(int(cfg["evaluation"]["success_window_episodes"]))
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        state_file = Path(str(resume_path.with_suffix("")) + "_state.json")
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            trained_with = state.get("reward_profile", "baseline")   # checkpoints from before the profiles existed = original reward
            current = configs["reward"].get("profile", "clean")
            if trained_with != current and not args.allow_profile_change:
                raise SystemExit(f"Checkpoint was trained with reward profile '{trained_with}' but the current profile is '{current}'. "
                                 f"Resume with --reward-profile {trained_with} (same learning problem), or pass --allow-profile-change "
                                 "and use a NEW run name if you really want to change the reward.")
            curriculum.load_state_dict(state["curriculum"])
            milestones.load_dict(state["milestones"])
            print(f"Restored curriculum level {curriculum.level} and milestones from {state_file.name}")
        else:
            print(f"[warning] no state file {state_file.name}: curriculum restarts at level {curriculum.level}")

    monitor_dir = paths.logs / "monitor" / str(cfg["run_name"])
    monitor_dir.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(configs, int(cfg["n_envs"]), int(cfg["seed"]), curriculum.level, monitor_dir,
                           subproc=args.subproc)
    if resume_path is not None:
        model = model_cls.load(str(resume_path), env=vec_env, device=device, tensorboard_log=str(paths.logs))
        buffer_file = Path(str(resume_path.with_suffix("")) + "_replay_buffer.pkl")
        if hasattr(model, "load_replay_buffer"):
            if buffer_file.exists():
                model.load_replay_buffer(str(buffer_file))
                print(f"Loaded replay buffer ({model.replay_buffer.size()} transitions)")
            else:
                print("[warning] replay buffer file not found; resuming with an empty buffer")
        print(f"Resumed {resume_path.name} at {model.num_timesteps} transitions")
    else:
        model = build_model(cfg, vec_env, device, paths.logs)

    demos_path = args.demos or cfg.get("teacher", {}).get("demonstrations_path")
    if demos_path and hasattr(model, "replay_buffer") and resume_path is None:
        from teacher.demonstrations import add_to_replay_buffer, behavior_clone_actor, load_demonstrations
        demos = load_demonstrations(PROJECT_ROOT / demos_path if not Path(demos_path).is_absolute() else demos_path)
        print(f"Teacher assistance: added {add_to_replay_buffer(model, demos)} demonstration transitions to the replay buffer")
        epochs = args.bc_epochs if args.bc_epochs is not None else int(cfg.get("teacher", {}).get("behavior_cloning_epochs", 0))
        if epochs > 0:
            losses = behavior_clone_actor(model, demos, epochs)
            print(f"Behaviour cloning: loss {losses[0]:.4f} -> {losses[-1]:.4f} over {epochs} epochs")

    set_training_level(vec_env, curriculum.level)
    showcase = ShowcaseRecorder(str(cfg["run_name"]), paths.results, cfg.get("showcase", {"enabled": False}),
                                float(configs["randomization"]["sampling"]["runway_width_m"]))
    monitor_cb = TrainingMonitorCallback(paths, milestones, curriculum, int(cfg["logging"]["log_every_steps"]), showcase)
    eval_cb = EvalCheckpointCallback(configs, cfg, paths, curriculum, milestones, showcase)
    start = time.time()
    start_steps = int(model.num_timesteps)
    remaining = max(int(cfg["total_timesteps"]) - start_steps, 0)
    exit_code = 0
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=CallbackList([monitor_cb, eval_cb]),
                        tb_log_name=str(cfg["run_name"]), reset_num_timesteps=resume_path is None,
                        progress_bar=bool(cfg["logging"]["progress_bar"]))
        if eval_cb.stop_reason:
            exit_code = 2
    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl+C received - saving model, replay buffer and metrics before exit ...")
        eval_cb.save_checkpoint(int(model.num_timesteps), f"interrupted_{int(model.num_timesteps)}")
        exit_code = 130
    finally:
        steps_done = int(model.num_timesteps)
        if exit_code != 130:
            eval_cb.save_checkpoint(steps_done, "final_model")
        wall = time.time() - start
        metrics = {"algorithm": algorithm, "run_name": cfg["run_name"], "device": device, "num_timesteps": steps_done,
                   "transitions_this_session": steps_done - start_steps, "wall_seconds": wall,
                   "transitions_per_second": (steps_done - start_steps) / wall if wall > 0 else None,
                   "final_curriculum_level": curriculum.level, "stop_reason": eval_cb.stop_reason,
                   "interrupted": exit_code == 130, "reward_profile": configs["reward"].get("profile", "clean"),
                   "milestones": milestones.to_dict(),
                   "last_evaluation": eval_cb.last_eval}
        (paths.results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        vec_env.close()
    print(json.dumps(metrics["milestones"], indent=2))
    print(f"Done: {steps_done} transitions in {wall:.0f} s. Models in {paths.models}, metrics in {paths.results / 'metrics.json'}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m training.train_sac``."""
    args = parse_args(__doc__ or "", "configs/sac.yaml", argv)
    return run_training("SAC", SAC, build_sac, args)


if __name__ == "__main__":
    raise SystemExit(main())
