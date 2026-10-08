# EXPERIMENT PROTOCOL

Simulation-only university research. No result below exists until it has been produced by the commands
in this file. Anything not yet produced is reported as **NOT YET MEASURED**.

## 0. Terminology (use exactly)

* **Physics step** – one JSBSim `fdm.run()` (1/120 s). 120 physics steps ≈ 1 simulated second.
* **RL environment transition** – one policy action followed by 12 physics steps (0.1 s). 10 transitions ≈ 1 simulated second.
* **Training steps** – always means RL environment transitions unless explicitly stated otherwise.
* A typical episode (45–90 simulated seconds) is ≈ 450–900 transitions; do not confuse this with total training transitions.

## 1. Pre-flight gate (before any long run)

```powershell
python tools/inspect_aircraft.py            # confirms model status 7478_RESEARCH_APPROXIMATION
python tools/check_environment.py           # must print 10/10 checks passed
python -m pytest                            # all tests must pass
python tools/benchmark.py                   # record throughput; estimate wall-clock for planned runs
```

## 2. Baselines (measure before judging the learner)

```powershell
python -m evaluation.evaluate --policy full_throttle --episodes 200 --level 1 --out results/baseline_fullthrottle_L1.csv
python -m evaluation.evaluate --policy full_throttle --episodes 200 --level 3 --out results/baseline_fullthrottle_L3.csv
python -m evaluation.evaluate --policy teacher       --episodes 200 --level 3 --out results/baseline_teacher_L3.csv
```

The open-loop full-throttle baseline can succeed in mild conditions (see `AIRCRAFT_MODEL_NOTES.md` §9), so
"the agent takes off" is not by itself evidence of learning; compare success rate, centerline error and
takeoff time against these baselines.

## 3. Training runs

| Run | Command | Purpose |
|---|---|---|
| SAC from scratch (primary) | `python -m training.train_sac --config configs/sac.yaml` | main controller |
| SAC + teacher demos | `python -m teacher.demonstrations --episodes 200 --level 1` then `python -m training.train_sac --config configs/sac.yaml --run-name sac_teacher --demos results/demos/teacher_demos.npz --bc-epochs 20` | does assistance reach first success sooner? |
| PPO (comparison only) | `python -m training.train_ppo --config configs/ppo.yaml` | algorithm comparison |

Use **at least 3 seeds** (`--seed N --run-name sac_s{N}`) before drawing conclusions; report mean ± spread.
Do not use DQN. The action space is continuous.

### Checkpoint schedule

Models are saved and deterministically evaluated at **50k, 100k, 250k, 500k, 1M, 2M, 5M, 10M**
transitions (`configs/sac.yaml: evaluation.checkpoint_steps`). Rolling checkpoints for resume are saved every 50k.
Results: `results/<run>/eval_history.csv`, `results/<run>/checkpoint_<N>_episodes.csv`.

### Stop rule at ~500k transitions

If neither training nor evaluation has produced one complete successful takeoff by 500k transitions,
the training script **stops itself** (exit code 2) and prints a warning. Do **not** simply restart for 10M
steps. Investigate in this order: reward (`results/<run>/training_episodes.csv` columns `rc_*`),
action mapping signs (`tools/inspect_aircraft.py --signs`), observation normalisation, aircraft
configuration, runway configuration, JSBSim properties, termination conditions, rotation behaviour,
and environment bugs (run the teacher and `tools/check_environment.py`).

### Milestones recorded automatically (`results/<run>/milestones.json`)

* `steps_to_first_success` – transitions before the first COMPLETE successful takeoff in a training episode;
* `steps_to_10_percent_success`, `steps_to_50_percent_success`, `steps_to_90_percent_success` – first time the rolling success rate over the last 100 training episodes reaches the threshold;
* `eval_steps_to_*` – same thresholds on the deterministic evaluation success rate;
* a value of `null` means **NOT YET MEASURED** (not reached).

Planning ranges from the project brief (hypotheses, **not** guarantees; actual numbers must be measured):
first success ≈ 100k–500k transitions with good shaping/demonstrations, from scratch 300k–1.5M+, reliable
controller 2M–5M+, broad randomized robustness 5M–15M+.

## 4. Definition of success (per episode)

Airborne, ≥ 1,000 ft AGL (above wheel-contact height), climb ≥ 500 ft/min, |roll| ≤ 8°, pitch 0–20°,
heading error ≤ 10°, AoA ≤ 12°, |β| ≤ 5°, lateral error ≤ 60 m, airspeed ≥ 1.15 × the stall estimate,
no crash / excursion / invalid state – and **all of this held for 5 s (50 consecutive transitions)**,
`UNCLEAN_TAKEOFF` (terminated with `success=False`) if the whole run's cleanliness score is below
`success.min_cleanliness` (default 0.5).
Only then `terminated=True`, `success=True`. Time-limit expiry is `truncated=True` (reason `TIMEOUT`).
All thresholds are in `configs/environment.yaml`.

## 5. Evaluation protocol

* Separate evaluation environment instance; deterministic actions; fixed seeds `eval_seed_base + i` (default 1,000,000+), disjoint from training.
* Metrics per episode (CSV): success, failure reason, takeoff time, liftoff time and distance (runway distance used), centerline RMS and max error, max roll, max pitch, max AoA, max heading error, control roughness, 100-ft stabilization fraction, return, and all sampled conditions.
* Training reward alone is never accepted as evidence.

```powershell
python -m evaluation.evaluate --model models/sac/best_model.zip --episodes 100 --level 3 --out results/eval_sac_L3.csv
```

## 6. Robustness protocol

Unseen seeds (from 5,000,000), levels 3 and 4: runway length/elevation/friction, mass, CG, temperature,
pressure, headwind/crosswind/small tailwind, gusts, light turbulence.

```powershell
python -m evaluation.robustness_test --model models/sac/best_model.zip --episodes 100  --levels 3 4   # initial
python -m evaluation.robustness_test --model models/sac/final_model.zip --episodes 1000 --levels 3 4  # final research evaluation
```

Outputs a failure table (SUCCESS, RUNWAY_EXCURSION, FAILED_ROTATION, INSUFFICIENT_RUNWAY, LOSS_OF_CONTROL,
UNSTABLE_AFTER_LIFTOFF, CRASH, SIMULATOR_INVALID_STATE, ENVELOPE_VIOLATION, FAILED_TO_ACCELERATE, TIMEOUT)
and a per-condition success breakdown.

## 7. Curriculum

Four levels (`configs/domain_randomization.yaml`). Level *n* → *n+1* when the **evaluation** success
rate exceeds `curriculum.success_threshold` (default 0.80, ≥ 20 evaluation episodes). Level 3 is the full
intended training distribution; level 4 is the robustness/generalisation distribution. 20 % of training
episodes are drawn from lower levels to limit forgetting. Some sampled conditions at higher levels may be
physically infeasible (heavy + hot + high + short runway); the failure table shows how many.

## 8. Reporting rules

1. Report transitions, not physics steps.
2. Report `NOT YET MEASURED` for anything not yet produced by an executed run.
3. Report the model status `7478_RESEARCH_APPROXIMATION` with every result.
4. Report baselines (full-throttle, teacher) alongside the learner.
5. Report seeds, config files, commit/date, and the device used (GPU/CPU).
6. Simulation success is not evidence of safety on any real aircraft.

## 9. Results ledger (fill from executed runs only)

### 9a. Measured environment-validation baselines (executed 2026-10-06, 200 episodes each, evaluation seeds 1,000,000+, model status 7478_RESEARCH_APPROXIMATION)

Files: `results/baseline_*_L*.csv`. These are **not** learned-policy results.

| Policy | Level | Success rate | Mean takeoff time (successes) | Mean runway distance to liftoff | Mean centerline RMS | Outcomes |
|---|---|---|---|---|---|---|
| Open-loop full throttle, neutral controls | 1 | 0.930 | 50.1 s | 1780 m | 12.6 m | 186 SUCCESS, 14 RUNWAY_EXCURSION |
| Open-loop full throttle, neutral controls | 3 | 0.050 | 47.3 s | 1497 m | 16.9 m | 10 SUCCESS, 184 RUNWAY_EXCURSION, 5 ENVELOPE_VIOLATION, 1 UNSTABLE_AFTER_LIFTOFF |
| Conventional simulated teacher | 3 | 0.995 | 43.9 s | 1465 m | 6.4 m | 199 SUCCESS, 1 CRASH |

Reading: level 1 is nearly solved by "full throttle" alone (a property of this forgiving FDM), so a high
level-1 success rate is not evidence of learned lateral control; level 3 requires real centerline/crosswind
control (open-loop 5 %). The single teacher CRASH (seed 1,000,152: 605,715 lb, 3,140 ft elevation, +10.8 °C ISA,
15.8 kt headwind, 12.3 kt crosswind, turbulence severity 1.6) was a post-liftoff nose-down divergence (pitch −23°,
−4,400 ft/min) – the teacher is a simple PD law, not an optimal controller.

### 9b. Learned-policy results

| Quantity | Value |
|---|---|
| Training transitions completed | NOT YET MEASURED |
| steps_to_first_success (SAC scratch) | NOT YET MEASURED |
| steps_to_first_success (SAC + teacher) | NOT YET MEASURED |
| steps_to_10/50/90 % success | NOT YET MEASURED |
| Final evaluation success rate | NOT YET MEASURED |
| Mean successful takeoff time | NOT YET MEASURED |
| Mean runway distance | NOT YET MEASURED |
| Robustness (1000 episodes) | NOT YET MEASURED |

## 10. Reward profiles and run ledger (added)

* `baseline` = original reward (runs `sac_pilot`, `sac_teacher_prev`); `clean` = passenger-comfort reward (runs `clean_scratch`, `clean_teacher`).
  Success rates and `steps_to_first_success` are only comparable between runs with the SAME profile. Compare comfort across profiles with the
  profile-independent `cleanliness`, `centerline_rms_m`, `centerline_max_m`, `rms_lateral_accel_mps2` columns of the evaluation CSVs.
* `sac_pilot` and `sac_teacher` (original reward) were stopped at about 380k and 142k transitions (resumable from checkpoint_350000 / checkpoint_100000 with
  `--reward-profile baseline`); their results so far are in `results/sac_pilot`, `results/sac_teacher`, `results/sac_teacher_prev`. No final numbers are claimed.
