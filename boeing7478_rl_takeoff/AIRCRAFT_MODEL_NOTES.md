# AIRCRAFT MODEL NOTES

## 1. Status

```
MODEL STATUS:  7478_RESEARCH_APPROXIMATION
```

The flight-dynamics model (FDM) used in this project is **not** a validated or certified Boeing 747-8
model. It is a JSBSim **747-400** FDM in which a small, explicitly listed set of parameters has been
replaced by published 747-8 figures. It is a research stand-in for studying the *learning problem*
(takeoff control with domain randomization), not a statement about how a real 747-8 flies.

Do not read any result from this project as evidence about real 747-8 performance, handling, or safety.

## 2. What was searched (and what was found)

| Source | Finding |
|---|---|
| JSBSim 1.3.1 pip package bundled data (`jsbsim/aircraft/`, 60 aircraft dirs) | **No 747-8.** Closest are `B747` and `787-8`. Text search of the whole package for `747-8`, `747-800`, `747-8i`, `B748`, `B747-8` returned **no matches**. |
| `aircraft/B747/B747.xml` header | `<fdm_config name="B747-400" ... release="ALPHA">`, author "Unknown", note: *"created using publicly available data, publicly available technical reports, textbooks, and guesses ... educational and entertainment purposes only"*. This is a **747-400** with unvalidated coefficients. |
| JSBSim upstream `aircraft/` directory on GitHub (master, fetched 2026-10-06) | Directory list contains `B747`, `787-8`, `737`, `MD11`, ... but **no 747-8**. |
| FlightGear 747-8i (FGAddon / FGMEMBERS `747-8i`) | Exists, but its FDM is **YASim** (`747-8i-yasim.xml`; FlightGear wiki: "FDM: YASim"). Its README says the first generation was "based on the 747-400 model by Gijs de Rooy". YASim is **not** JSBSim, so it cannot be dropped in. |

Limits of this search: a general web search tool returned no results, and not every third-party
repository on the internet could be inspected. The statement is therefore "no genuine JSBSim 747-8 FDM
was found in the installed package, upstream JSBSim, or the FlightGear 747-8 project", **not** a proof
that none exists anywhere. If a validated JSBSim 747-8 FDM becomes available, replace the generated
aircraft (see section 8) and update this file.

## 3. Known data (published 747-8 values actually used)

All of these are set in `configs/aircraft.yaml` under `overrides:` with provenance and are applied by
`envs/jsbsim_interface.py:build_derived_aircraft()`.

| Parameter | Value used | Source |
|---|---|---|
| Wingspan | 224 ft 5 in (68.40 m) | Wikipedia "Boeing 747-8" specifications (Boeing data) |
| Wing area | 5,960 ft² (554 m²) | same |
| Operating empty weight (747-8I) | 485,300 lb | same (comparison table) |
| Max takeoff weight (747-8I) | 987,000 lb (design MTOW 975,000 lb also published) | same |
| Max payload (747-8I) | 167,700 lb | same |
| Usable fuel (747-8I) | 422,328 lb (63,034 US gal) | same |
| Engine | 4 × GEnx-2B67 | same |
| Takeoff thrust per engine | 66,500 lbf (EASA table lists 67,400 lbf) | Wikipedia "General Electric GEnx" data sheet |
| Takeoff bypass ratio | 8.0 | same |

These are secondary-source (encyclopedia) figures quoting Boeing/GE data; they were not checked against
Boeing's own documents.

## 4. Data inherited from another aircraft (JSBSim 747-400, **not** 747-8 data)

Everything not in section 3 is unchanged from `B747.xml`:

* all aerodynamic coefficients and tables (CL-alpha, drag polar, Mach drag, control derivatives, damping, side force, flap/gear/speedbrake increments);
* mean chord (27.31 ft), horizontal/vertical tail areas and arms;
* moments/products of inertia (`Ixx 1.82e7`, `Iyy 3.31e7`, `Izz 4.97e7`, `Ixz -9.7e5` slug·ft²);
* CG station (x = 1327 in), aerodynamic reference point (x = 1377 in);
* landing-gear geometry, spring and damping coefficients, friction coefficients, nose-wheel steering limit (5°);
* engine placement, direct-thrust model, and the **thrust-lapse, idle and TSFC tables of the GE CF6-80C2-B1F** file (normalised, then scaled to 66,500 lbf);
* control-surface travel (elevator −0.35/+0.175 rad, aileron ±0.35, rudder ±0.35), yaw damper, flap range (0–30°), gear/flap kinematics.

Differences between the two real aircraft that are therefore **not** represented: the 747-8's thicker,
re-designed wing with raked tips, different flap system, lengthened fuselage (+18.3 ft), revised
inertia, fly-by-wire roll control, GEnx thrust/lapse characteristics, and different landing-gear loads.

## 5. Estimated by this project (engineering assumptions)

| Item | Assumption | Where |
|---|---|---|
| Takeoff flap setting | 20° (typical 747 takeoff setting; FDM supports 0–30°) | `aircraft.yaml: takeoff_config` |
| CLmax with takeoff flaps | 2.2 (read from FDM tables: 1.2 at the alpha peak + 0.05 × 20°) | `aircraft.yaml: envelope` |
| VR / V2 | 1.08 × / 1.15 × the 1-g stall estimate `sqrt(2W/(rho_sl·S·CLmax))` | `aircraft.yaml: envelope` |
| Fuel representation | 5 tanks (as in the base FDM), total capacity = published usable fuel, all located at the CG station (so fuel does not move the CG) | `build_derived_aircraft` |
| Payload / CG | one point mass ("PAYLOAD"); its station is chosen so that the CG shifts by the sampled amount (clamped to stations 300–2900 in). The achieved CG is read back from JSBSim (`inertia/cg-x-in`) | `randomization.py` |
| Runway friction | multipliers on the FDM tire friction through `ground/static-friction-factor` and `ground/rolling_friction-factor`; "dry/wet/contaminated" labels are only a naming convention, **not** measured braking action | `domain_randomization.yaml` |
| Tail-strike proxy | 12° pitch on the main gear (the FDM has no tail contact point) | `aircraft.yaml: envelope.ground_pitch_limit_deg` |
| Gusts | first-order Gauss-Markov process (τ = 3 s) added to the steady wind | `boeing7478_takeoff_env.py` |
| Turbulence | JSBSim MIL-F-8785C model, severity 0–3 | `domain_randomization.yaml` |
| All termination limits (roll, pitch, sideslip, AoA, sink rate, ...) and success-region limits | engineering judgement | `environment.yaml` |

## 6. Unavailable

* A validated 747-8 aerodynamic database, wind-tunnel or flight-test data.
* 747-8 inertia, gear and ground-handling data; ground-effect model (the base FDM has none).
* GEnx-2B thrust-vs-Mach/altitude tables, spool dynamics, FADEC logic, fuel flow.
* Boeing takeoff-performance charts (V1/VR/V2, field lengths) to compare against.
* Real flight-control laws (747-8 has fly-by-wire spoilers/ailerons); trim schedules; tiller/rudder steering authority.
* Runway slope: JSBSim's built-in ground is flat, so slope randomization is **not supported** (documented in `domain_randomization.yaml`).

## 7. Why this cannot be treated as a certified Boeing model

1. The aerodynamic and inertial core is an unvalidated, community-made 747-400 FDM whose own header says it contains "guesses".
2. Only geometry/weight/thrust headline numbers were replaced; the coupled aerodynamics were not re-derived, so changing wing area/span scales forces and moments dimensionally but does not reproduce the 747-8's real aerodynamics.
3. No comparison against Boeing performance data has been made; the model is **not** validated.
4. Nothing here is certified, endorsed by the manufacturer, or suitable for operational decisions.

## 8. How the model is produced / how to swap in a real FDM

`JSBSimInterface.__init__` calls `build_derived_aircraft()` which reads the installed `aircraft/B747/B747.xml`,
applies the overrides from `configs/aircraft.yaml`, adds the payload point mass, writes
`models/_aircraft_build/aircraft/B747-8_RA/B747-8_RA.xml` and a `GEnx-2B67_RA` engine file, and JSBSim loads
those. Inspect with `python tools/inspect_aircraft.py`. To use a genuine JSBSim 747-8 FDM, replace the
builder with a loader for that file (keeping the property registry in `jsbsim_interface.py`, which will
fail loudly if properties differ) and change `model_status`.

## 9. Behaviours of the approximation observed during implementation

(Measured in this project's simulations; **not** validated against a real aircraft.)

* At rest the FDM sits **nose-low by about 2.9°** (nose gear is much softer than the main gear), and the CG is about 12.5 ft above the ground contact plane. The environment therefore measures AGL relative to the settled ground reference height (a raw `position/h-agl-ft` of ~12.5 ft is "on the ground").
* With full throttle and **neutral controls** the model can lift off and climb by itself in benign conditions (it is strongly self-stabilising in pitch). A do-nothing "full-throttle" baseline therefore succeeds in some episodes; every learning result must be reported next to that baseline (`python -m evaluation.evaluate --policy full_throttle`).
* Open-loop, the aircraft drifts laterally (several metres to tens of metres), so centerline tracking is a real control task.
* JSBSim sign conventions (verified with `python tools/inspect_aircraft.py --signs`): positive `elevator-cmd` = nose **down**, positive `rudder-cmd` = yaw **left**, positive `aileron-cmd` = roll right, positive `steer-cmd` = nose wheel right. The action mapper applies the signs so the agent sees +elevator = nose-up, +aileron = roll right, +rudder = yaw right.
* `position/distance-from-start-*` properties are **unsigned**; the interface recovers the sign from latitude/longitude and cross-checked the result against velocity integration (agreement ≈ 0.3 m over 600 m).
* Sideslip angle is meaningless below flying speed (a parked aircraft in a crosswind shows |β| up to 90°), so β-based checks are only applied above 50 KCAS.
