# Boeing 747-8 RL Takeoff (simulation-only research project)

Train a reinforcement-learning controller (primary algorithm: **SAC**) to perform the **takeoff** portion of
flight in a JSBSim-based Boeing 747-8 *research approximation*: start stationary on a runway with engines
running, accelerate, track the centerline, rotate, lift off, establish a positive climb, reach **1,000 ft AGL**
and remain stable for **5 s** – in the shortest elapsed time that still satisfies the stability/safety limits.

> **Aircraft model status: `7478_RESEARCH_APPROXIMATION`.** No genuine, validated JSBSim 747-8 FDM was found.
> The model is the JSBSim 747-400 FDM with a short, documented list of published 747-8 numbers substituted.
> Read [`AIRCRAFT_MODEL_NOTES.md`](AIRCRAFT_MODEL_NOTES.md) before interpreting anything.
>
> **Scope:** simulation and university research only. Not certified software. Nothing here may be connected to a
> real aircraft, and simulation success says nothing about real-aircraft safety.

## Terminology (used consistently)

| Term | Meaning |
|---|---|
| Physics step | one JSBSim `run()`; 120 Hz → 120 per simulated second |
| RL environment transition | one policy action + **12** physics steps (0.1 s at 10 Hz) |
| Training steps | RL environment transitions (never physics steps) |

Both frequencies are configurable in `configs/environment.yaml` (`physics_hz: 120`, `policy_hz: 10`).

## Quick start (Windows / PowerShell)

```powershell
cd boeing7478_rl_takeoff
python -m venv .venv                      # Python 3.10-3.13
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build (12 GB GPU target)
pip install -r requirements.txt

python tools/inspect_aircraft.py          # aircraft identity / model status
python tools/check_environment.py         # MUST pass 10/10 before training
python -m pytest                          # unit + integration tests
python tools/benchmark.py                 # throughput (physics, env, SAC updates, GPU)
```

## Train

```powershell
python -m training.train_sac --config configs/sac.yaml                       # SAC from scratch (primary)
python -m training.train_sac --config configs/sac.yaml --resume models/sac/checkpoint_500000.zip   # continue
python -m training.train_ppo --config configs/ppo.yaml                       # PPO (comparison only)

# SAC assisted by simulation-teacher demonstrations
python -m teacher.demonstrations --episodes 200 --level 1 --out results/demos/teacher_demos.npz
python -m training.train_sac --config configs/sac.yaml --run-name sac_teacher --demos results/demos/teacher_demos.npz --bc-epochs 20
```

* Ctrl+C is handled: the model, replay buffer and metrics are saved (`models/<run>/interrupted_<N>.zip`), and the run can be resumed.
* Outputs: `models/<run>/` (`checkpoint_<N>.zip`, `best_model.zip`, `final_model.zip`, `*_replay_buffer.pkl`, `*_state.json`), `results/<run>/` (`metrics.json`, `milestones.json`, `eval_history.csv`, `training_episodes.csv`, per-checkpoint episode CSVs), `logs/` (TensorBoard).
* Checkpoints are saved and evaluated at 50k, 100k, 250k, 500k, 1M, 2M, 5M, 10M transitions.
* If **no** successful takeoff exists by ~500k transitions the run stops itself and tells you what to investigate instead of burning millions of steps.
* The device is auto-detected; PyTorch version, CUDA availability, GPU name and selected device are printed at start. JSBSim itself is CPU-bound; see `tools/benchmark.py`.

### TensorBoard

```powershell
tensorboard --logdir logs
```

Logged: total reward and every reward component, episode duration/transitions, takeoff time, altitude, airspeed,
vertical speed, pitch, roll, heading error, centerline error, AoA, sideslip, throttle/elevator/aileron/rudder,
mass, CG, wind, runway, success/failure outcome, curriculum stage, evaluation metrics, and the milestone
`steps_to_first_success`.

## Evaluate

```powershell
python -m evaluation.evaluate --model models/sac/best_model.zip --episodes 100 --level 3 --out results/eval_sac_L3.csv
python -m evaluation.evaluate --policy full_throttle --episodes 100 --level 3     # open-loop baseline (see notes)
python -m evaluation.evaluate --policy teacher --episodes 100 --level 3           # conventional controller
python -m evaluation.robustness_test --model models/sac/best_model.zip --episodes 100 --levels 3 4   # 1000+ for final
python tools/plot_training.py --runs sac sac_teacher                              # plots / milestone comparison
```

## Design summary

| Area | Implementation |
|---|---|
| Physics | JSBSim 1.3.x, 120 Hz, headless (no rendering) |
| Actions | `Box(-1,1,(4,))` = throttle, elevator, aileron, rudder; one common throttle for 4 engines; throttle action→lever [0,1]; per-axis **rate limits** (`aircraft.yaml`); nose-wheel steering follows rudder pedal |
| Pre-episode configuration | engines running, takeoff flaps 20°, gear down, brakes released, trim set – the agent never handles these |
| Observations | 27 normalised, clipped, finite values (`envs/observations.py: OBS_NAMES`), runway-relative: airspeed, ground speed, AGL, vertical speed, pitch, roll, relative heading, AoA, sideslip, p/q/r, lateral error, course error, distance travelled/remaining, weight-on-wheels, control positions, throttle, engine N1, mass, CG, head/crosswind, density ratio |
| Reward | dense, 18 separate readable components (`envs/reward.py`), all weights in `configs/reward.yaml`; time cost is tiny and the extra time bonus is paid **only** on a valid, stabilised takeoff |
| Success | all criteria in `environment.yaml` held for 5 s (50 transitions) → `terminated=True, success=True` |
| Failures | crash, runway excursion, loss of control, invalid simulator state, unstable after liftoff, runway exhausted (FAILED_ROTATION / INSUFFICIENT_RUNWAY), envelope violation, failure to accelerate → `terminated`; time limit → `truncated` |
| Randomization (from episode 1) | mass/fuel/payload, CG, runway length/elevation/friction/heading, temperature, pressure, density, head/cross/tail wind, gusts, turbulence, initial lateral/heading offset (runway slope **unsupported** by JSBSim) |
| Curriculum | 4 levels; advance when **evaluation** success rate > threshold (default 0.80) |
| Teacher | `teacher/simulated_teacher.py`: conventional simulation-only controller, demonstration collection, replay-buffer prefill, behaviour-cloning option |
| Errors | every JSBSim property is registered and verified at load; missing property → loud `PropertyNotFoundError` with similar names; no bare `except: pass` |

## Project layout

```
configs/      sac.yaml ppo.yaml aircraft.yaml environment.yaml reward.yaml domain_randomization.yaml
envs/         Gymnasium env, JSBSim interface (+ derived FDM generator), observations, actions, reward,
              termination, randomization, runway geometry
training/     train_sac.py train_ppo.py callbacks.py curriculum.py
teacher/      simulated_teacher.py demonstrations.py
evaluation/   evaluate.py robustness_test.py failure_analysis.py
tools/        check_environment.py inspect_aircraft.py benchmark.py plot_training.py
tests/        test_environment/actions/observations/reward/termination.py
models/ logs/ results/   generated artefacts
```

## Results

**Nothing is claimed here that has not been produced by an executed run.** Current status of the research quantities:

| Quantity | Status |
|---|---|
| Learned-policy training transitions | NOT YET MEASURED |
| First successful takeoff (steps_to_first_success) | NOT YET MEASURED |
| 10 / 50 / 90 % success thresholds | NOT YET MEASURED |
| Final evaluation success rate | NOT YET MEASURED |
| Mean successful takeoff time / runway distance | NOT YET MEASURED |
| Robustness (1000+ episodes) | NOT YET MEASURED |

Environment-validation measurements (teacher/baseline/benchmark/short pipeline runs) that *have* been executed are
listed in `results/` and summarised in [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) §9 once produced.

## Known limitations

* Aircraft is an approximation (see notes); aero core is a 747-400 FDM with "guesses".
* The FDM is very forgiving in pitch: a full-throttle, neutral-control policy can take off in mild conditions, so always compare against baselines.
* No runway slope; friction is a multiplier, not a measured braking action; fuel sits at the CG station.
* Reward weights and all thresholds are starting values, not tuned results.

## 3D replay (separate from training)

```powershell
python tools/replay_3d.py --open                      # latest sac_pilot checkpoint (3 seeds) + teacher, in the browser
python tools/replay_3d.py --model models/sac_teacher/best_model.zip --seeds 1000000 1000001 --open
python tools/replay.py --policy teacher --gif         # 2D flight-data dashboard / GIF
```

`tools/replay_3d.py` records real JSBSim episodes and writes `results/replay3d/replay3d.html` (Three.js; needs internet the
first time to load the library). Cameras 1-5: chase, side, tower, orbit, top-down. If `results/replay3d/boeing747erf.glb` exists it is
embedded as the aircraft model (toggle "Simple" in the page to switch back).

**Attribution:** the GLB is *"Boeing747ERF"* by **manilov.ap** (https://sketchfab.com/3d-models/boeing747erf-5153fd42231a4723a8634e02c4bb008c),
licensed **CC-BY-4.0** - see `results/replay3d/ATTRIBUTION.txt`. It is a 747-400ERF visual model scaled to the 747-8 span (68.4 m); it carries
**no** aerodynamic data and does not change the simulation, which is still the `7478_RESEARCH_APPROXIMATION` JSBSim model.

## Live 3D training showcase (updates as training runs)

```powershell
python tools/live_viewer.py --open                    # serves http://127.0.0.1:8765/ ; keep it running
python -m training.train_sac --config configs/sac.yaml  # in another terminal (recording is on by default)
python -m evaluation.trajectory --run sac_pilot       # optional: backfill history from existing checkpoints
```

During training (`showcase:` section in `configs/sac.yaml`) the following **events** are written to `results/<run>/showcase/events/`
and appear in the viewer within ~3 s, with a banner, a playlist entry and (with *Auto-follow newest*) immediate playback:

| Event | Trajectory shown |
|---|---|
| `FIRST_SUCCESS`, `TRAINING_SUCCESS` | the **exact** trajectory of the training episode that succeeded (every success for the first 25, then at most one per 10k transitions) |
| `FIRST_LIFTOFF` | first training episode that left the ground |
| `EVAL_FIRST_SUCCESS`, `BIG_PROGRESS` (+10 points evaluation success), `LEVEL_UP` | deterministic replays of the current policy (3 seeds); a model snapshot is saved to `models/<run>/showcase_<steps>_<event>.zip` |
| `CHECKPOINT` | one deterministic replay at each milestone checkpoint (50k, 100k, 250k, ...) |

The viewer also shows live training status per run (transitions, curriculum level, training successes, rolling success rate, latest
evaluation, `steps_to_first_success` or NOT YET). Everything shown is real recorded simulation output.

## Clean-trajectory reward (passenger comfort)

A takeoff that wanders left/right is "valid" by the success criteria but frightening for passengers, so the default reward profile
(`configs/reward.yaml: profile: clean`) adds comfort terms. All weights are in the YAML:

| Term | Effect |
|---|---|
| `clean_centerline` | extra reward per step only when really on the centerline (gaussian, scale 2 m, gated by ground speed) |
| `lateral_rate_penalty`, `lateral_accel_penalty` | penalise sideways velocity / acceleration (swaying, computed from the true velocity vector) |
| `yaw_rate_penalty`, `roll_rate_penalty` | penalise nose swinging and wing rocking |
| `clean_takeoff_bonus` | on a valid takeoff, `150 x cleanliness`, where `cleanliness = exp(-mean((stat/scale)^2))` over RMS/max lateral error, RMS lateral acceleration and max roll |

Measured with the final scales (episodes with the same kind of conditions): clean teacher takeoff cleanliness 1.00, return 432; teacher in strong
crosswind 0.47, return 301; open-loop drift 0.26, return 291; wobbly learned takeoff 0.12, return 132.  Profile `baseline` reproduces the ORIGINAL reward
exactly (the early `sac_pilot` / `sac_teacher_prev` runs used it).  Changing the profile changes the learning problem, so use a NEW run name:

```powershell
python -m training.train_sac --config configs/sac.yaml --run-name clean_scratch --reward-profile clean
python -m training.train_sac --config configs/sac.yaml --run-name clean_teacher --reward-profile clean --demos results/demos/teacher_demos.npz --bc-epochs 20
# resume an old run under its ORIGINAL reward (resuming under a different profile is refused unless --allow-profile-change):
python -m training.train_sac --config configs/sac.yaml --run-name sac_pilot --resume models/sac_pilot/checkpoint_350000.zip --reward-profile baseline --total-timesteps 500000
```

Every 50,000 transitions the viewer receives a `STEP_UPDATE` (2 deterministic replays of the current policy, with training/evaluation success in the
headline) - set `showcase.update_every_steps` in `configs/sac.yaml`.  Episode summaries and TensorBoard now include `cleanliness`.
