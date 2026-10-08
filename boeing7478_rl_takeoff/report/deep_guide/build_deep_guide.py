"""Build a beginner-first project handbook as LaTeX source and a verified PDF fallback."""
from __future__ import annotations

import json
import hashlib
import re
import sys
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

sys.path.append(r"C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages")
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = ROOT / "report" / "assets"
BRAND = HERE / "assets"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / "Autonomous_Airplane_From_Zero_Handbook.pdf"
TEX = HERE / "Autonomous_Airplane_From_Zero_Handbook.tex"
SNAP = json.loads((ROOT / "report" / "status_snapshot.json").read_text(encoding="utf-8"))
FONT_DIR = ROOT / ".venv" / "Lib" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("CMRoman", str(FONT_DIR / "cmr10.ttf")))
pdfmetrics.registerFont(TTFont("CMBold", str(FONT_DIR / "cmb10.ttf")))
pdfmetrics.registerFont(TTFont("CMMono", str(FONT_DIR / "cmtt10.ttf")))

blocks: list[tuple] = []
def h1(s): blocks.append(("h1", s))
def h2(s): blocks.append(("h2", s))
def p(s): blocks.append(("p", s))
def bullet(s): blocks.append(("bullet", s))
def code(s): blocks.append(("code", s))
def fig(name, caption, width=165): blocks.append(("fig", name, caption, width))
def table(rows, widths=None): blocks.append(("table", rows, widths))
def page(): blocks.append(("page",))
def callout(s): blocks.append(("callout", s))

h1("The autonomous airplane, from zero")
p("A visual, step-by-step field guide to rebuilding and continuing the Boeing 747-8 reinforcement-learning takeoff research project. Written for a reader who has never studied machine learning.")
p("Edition: 8 October 2026. The code and measurements describe the repository as inspected on that date. The simulator is a research approximation, and the project is for simulation only.")
fig("simulation_3d.png", "A real replay from this project. The 3D aircraft is a visual model; JSBSim calculates the motion.", 165)
callout("Memory path: SEE the state -> STEER with four controls -> SCORE the result -> SAVE what worked -> TEST on new conditions.")
page()

h1("1. The whole story in two minutes")
p("Imagine teaching someone to take off in a simulator. You put the airplane at the start of a runway, let the student move four controls, tell them what happened, and repeat. The student is a computer program called an agent. The simulator is JSBSim. The teacher is optional example code. The score is a reward. The final exam uses runway and weather conditions that the training run did not memorize.")
fig("project_flow.png", "The project loop: conditions -> simulator -> observations -> agent -> controls -> new simulator state -> reward.")
p("One attempt is an episode. An episode begins with engines running, flaps set and gear down. It ends in a stable success, a named failure, or a time limit. The agent only chooses throttle, elevator, aileron and rudder; it does not configure flaps, gear, brakes or trim during an episode.")
table([
 ["Everyday idea", "Project name", "Why it exists"],
 ["Student", "agent / policy", "Chooses the next control action."],
 ["Practice room", "environment", "Runs physics and returns feedback."],
 ["What the student sees", "observation", "27 measured or derived numbers."],
 ["What the student does", "action", "Four requested control values."],
 ["Score", "reward", "Small clues plus a large valid-success bonus."],
 ["One try", "episode", "A full runway-to-outcome attempt."],
 ["Harder lessons", "curriculum", "Expands weather, weight and runway difficulty."],
])

h1("2. Before machine learning: what is a takeoff?")
p("The airplane must accelerate along the runway while staying near its center line. Near an estimated rotation speed, it raises its nose smoothly. It lifts off, climbs, reaches 1,000 feet above the runway, and stays within the allowed speed and attitude limits for five seconds. A rough or wandering path can still fail the cleanliness rule.")
fig("takeoff_story.png", "The takeoff as a story: roll, steer, rotate, lift off, climb, stabilize.")
p("Airspeed is motion relative to the air; ground speed is motion over the runway. In a headwind they differ. Pitch is nose up or down. Roll is wing tilt. Heading is where the nose points. Course is where the airplane actually moves. Centerline error is sideways distance from the runway middle. AGL means height above the local ground reference. The code uses meters for runway position, knots for airspeed, feet for height, and feet per minute for climb rate; conversions happen before values enter the policy.")

h1("3. Reinforcement learning without jargon")
p("Ordinary supervised learning says, 'Here is the right answer for this example.' Reinforcement learning says, 'Try an action; here is what happened and how useful it was.' The agent must discover a sequence of actions whose total score is high. A takeoff is a sequence problem: a good throttle choice now changes the speed available for rotation later.")
p("At time t, the observation is s(t), the action is a(t), the reward is r(t), and the next observation is s(t+1). The policy is a function that maps s(t) to a(t). Its goal is to maximize the expected sum of future rewards, often written r(0) + gamma r(1) + gamma squared r(2) + ... . Gamma, set to 0.995 here, makes near-future and later consequences matter.")
callout("Tiny example: a sharp rudder movement might give an immediate centerline improvement, but it can cause a later swing. The agent should value the whole flight, not only the next tenth of a second.")
p("Exploration means trying uncertain actions during training. Evaluation turns that randomness off so repeated exams are comparable. A replay buffer is a notebook of past (observation, action, reward, next observation, done) transitions. The agent samples old notebook entries to learn without repeating every flight immediately.")

h1("4. Why SAC, neural networks and a teacher?")
p("The four controls are continuous values between -1 and 1. Soft Actor-Critic (SAC) is designed for continuous actions. A DQN-style menu of discrete actions would need an artificial grid of control positions. The SAC actor is a neural network that proposes actions. Two critic networks estimate how valuable actions are; using two helps limit overly optimistic estimates. 'Soft' means the objective also rewards enough exploration, controlled by an automatically adjusted entropy coefficient.")
p("A neural network here is a set of adjustable numbers. Three hidden layers with 256 units each give the actor and critics enough flexibility to connect speed, wind, runway error and control choices. ReLU is the activation function. Backpropagation adjusts the network weights after sampling experiences from the replay buffer. The learning rate, 0.0003, controls the size of those adjustments.")
p("The teacher is a plain feedback controller. It gives full thrust, steers toward the runway center, waits for an estimated rotation speed and then targets a climb pitch. Its demonstrations can fill the replay buffer, and behavior cloning can teach the actor to imitate example actions before SAC exploration. The teacher is a learning aid and a debugging baseline, not a certified flight controller.")
table([
 ["Setting in configs/sac.yaml", "Value", "Plain purpose"],
 ["buffer_size", "1,000,000", "How many past transitions the notebook can hold."],
 ["learning_starts", "10,000", "Gather experience before regular gradient updates."],
 ["batch_size", "256", "Past transitions sampled for one update."],
 ["gamma", "0.995", "How much later rewards count."],
 ["tau", "0.005", "How slowly target critics follow learned critics."],
 ["train_freq / gradient_steps", "1 / 1", "One learning update per new transition."],
 ["ent_coef", "auto", "Let SAC tune exploration pressure."],
])

h1("5. One episode, viewed as a flowchart")
p("First, reset chooses an episode seed and difficulty, samples aircraft loading, runway and weather, and initializes the simulator. Then the program reads sensors and converts them to a 27-number observation. The agent chooses four controls. An actuator mapper limits how fast each control may move. JSBSim runs 12 small physics steps while that command is held. The program reads the new state, checks success and failure, computes reward and sends the new observation back. The cycle repeats at 10 actions per simulated second.")
table([
 ["Stage", "Code owner", "What to inspect when it fails"],
 ["Load YAML", "envs/__init__.py", "Wrong config directory or missing key."],
 ["Build aircraft", "envs/jsbsim_interface.py", "XML generation, properties, model status."],
 ["Draw conditions", "envs/randomization.py", "Difficulty level, seed, ranges."],
 ["Run physics", "envs/jsbsim_interface.py", "JSBSim property values and non-finite state."],
 ["Build observations", "envs/observations.py", "Order, units, scaling, clipping."],
 ["Apply actions", "envs/actions.py", "Signs, limits, throttle mapping."],
 ["Decide outcome", "envs/termination.py", "Success hold, failure priority, timeout."],
 ["Score flight", "envs/reward.py", "Each reward component and profile."],
 ["Train / save", "training/train_sac.py", "SAC, teacher data, resume, callbacks."],
])

h1("6. What the agent sees: all 27 observations")
p("The policy cannot look at the 3D picture. It receives a fixed list of numbers, always in the order below. The order is part of the interface: changing it without updating a saved policy breaks that policy. Values are divided by a configured scale and clipped to -5 through 5 so very large numbers do not overwhelm the neural network. A non-finite value is treated as a simulator error.")
table([
 ["Number", "Names", "Meaning and reason"],
 ["1-4", "airspeed, ground_speed, agl, vertical_speed", "Know whether to accelerate, rotate and climb."],
 ["5-8", "pitch, roll, relative_heading, alpha", "Know attitude and whether the wing is near a dangerous angle."],
 ["9-12", "beta, p, q, r", "Know side slip and how fast the aircraft rotates."],
 ["13-16", "lateral_error, course_error, distance_travelled, distance_remaining", "Stay centered and know runway left."],
 ["17-21", "weight_on_wheels, elevator_pos, aileron_pos, rudder_pos, throttle", "Know airborne/ground state and actual actuator positions."],
 ["22-24", "engine_n1, mass_norm, cg_norm", "Know engine response and loading."],
 ["25-27", "headwind, crosswind, density_ratio", "Adapt to weather and thinner air."],
])
p("Example: if the airplane is 6 meters right of center and the scale is 30 meters, the lateral input is 6/30 = 0.2. The positive sign means right. The teacher interprets this and commands a small leftward heading. 'Normalized' means made comparable in size; it does not mean the physical quantity disappears.")

h1("7. What the agent does: four requested actions")
p("The agent outputs [throttle, elevator, aileron, rudder], each in [-1, 1]. Throttle -1 means idle and +1 means full lever; the other three use +elevator for nose up, +aileron for roll right and +rudder for yaw right. JSBSim uses different signs for some axes, so the mapper changes signs before sending commands. Nose-wheel steering follows rudder on the ground. Each actuator moves toward the request at a limited rate, rather than jumping instantly.")
table([
 ["Action", "Sign sent to JSBSim", "Rate limit per second", "Why this matters"],
 ["Throttle", "Mapped from [-1,1] to [0,1]", "0.5", "Avoid instant engine-lever jumps."],
 ["Elevator", "-1", "1.0", "Make agent-positive mean nose up."],
 ["Aileron", "+1", "1.5", "Make agent-positive mean right roll."],
 ["Rudder", "-1; steering +1", "1.5", "Make agent-positive mean right yaw and wheel turn."],
])
p("The action mapper checks shape and rejects NaN or infinity. It clips commands to the allowed range, maps throttle, computes the difference from the current actuator position, clips that difference by rate times 0.1 second, and only then sends JSBSim commands. This is why the policy observes actuator positions: the requested control and the actual control may temporarily differ.")

h1("8. The airplane model: what is real and what is assumed")
p("JSBSim is the physics calculator. It is not the 3D viewer. The repository found no validated JSBSim 747-8 flight-dynamics model. It starts from JSBSim's community Boeing 747-400 model and changes a short list of public 747-8 figures: wingspan, wing area, empty weight, fuel capacity and engine thrust. Aerodynamic tables, inertia, landing gear and many engine details remain from the 747-400. The generated aircraft is explicitly named 7478_RESEARCH_APPROXIMATION.")
p("A picture of a 747 in the replay can be visually convincing while the underlying physics remains approximate. The 3D mesh is a display stand-in and never supplies lift or thrust to JSBSim. Therefore, every performance result here is a result about this simulator setup, not a real 747-8.")
p("The model also estimates rotation speed from a simplified stall-speed formula. This helps the agent and teacher know roughly when rotation is plausible; it is not a Boeing takeoff chart. Fuel is placed at the center of gravity in the approximation, runway slope is unsupported, tire friction uses multipliers, and some limits are engineering choices. For a more faithful study, a validated aircraft model and real performance comparisons are essential.")

h1("9. Randomization: the practice room changes")
p("If every practice flight had the same weight, wind and runway, the agent could memorize one trick. The randomizer changes mass, center of gravity, runway length/elevation/friction/heading, temperature, pressure, wind, gusts, turbulence and small starting errors. A fixed seed reproduces the same draw, which is valuable for debugging.")
p("The curriculum begins with level 1: easy but still variable. Levels 2, 3 and 4 widen the conditions. Level 5 in this repository explores very slippery runway cases. Twenty percent of training episodes can be drawn from a lower level, so the policy keeps practicing old skills. The promotion rule uses deterministic evaluation on at least 20 episodes and requires success strictly above 80 percent.")
table([
 ["Level", "Plain interpretation", "Example challenge"],
 ["1", "Learn the basic roll and climb", "Light wind, long runway, moderate mass."],
 ["2", "Add noticeable variation", "More crosswind, CG shift and runway drag."],
 ["3", "Intended broad training", "Heavier mass, stronger wind, shorter runway."],
 ["4", "Robustness stress", "Wider weather and runway ranges."],
 ["5", "Slippery-runway curiosity", "Very low tire-friction factors."],
])

h1("10. Reward: the coach's point system")
p("The final success rule is the goal. Dense reward gives useful clues along the way so the agent can learn before its first full takeoff. A small positive score encourages forward progress, correct heading, centerline tracking, acceleration, controlled rotation, climb and time spent stable. Penalties discourage sideways drift, large roll or heading error, excessive angle of attack, sideslip, fast control changes, early rotation and named failures.")
p("The clean profile adds a tighter centerline reward and penalties for sideways speed/acceleration and rocking. At a valid takeoff it pays a bonus proportional to whole-flight cleanliness. The baseline profile zeros those comfort terms for comparison. Changing reward profiles changes the task; resume a checkpoint only with its original profile unless deliberately starting a new experiment.")
table([
 ["Question", "Answer"],
 ["Why reward progress?", "A full takeoff can be rare at first; progress gives an early learning signal."],
 ["Why a large success bonus?", "The complete stable takeoff must matter more than farming small step rewards."],
 ["Why a small time penalty?", "We want timely takeoff only after safe completion, not speed at any cost."],
 ["Why speed-gate alignment reward?", "Otherwise the policy could earn centerline points while standing still."],
 ["Why cap squared penalties?", "A single extreme state should not create an unstable giant training signal."],
 ["Why separate components?", "The training CSV can reveal which behavior is rewarded or punished."],
])
p("A useful debugging example: if the plane accelerates but swerves, inspect centerline reward, lateral penalty, heading penalty and control-oscillation penalty in the episode CSV. Do not change five weights at once; change one hypothesis, run the same evaluation seeds, and compare against the prior result.")

h1("11. Success, failure and timeouts")
p("Success is deliberately harder than 'the wheels left the ground.' The aircraft must be airborne, reach 1,000 ft AGL, climb at least 500 ft/min, stay within roll/pitch/heading/angle-of-attack/sideslip/centerline limits, and maintain enough airspeed. These conditions must hold for 50 consecutive 0.1-second policy steps. Then the whole flight needs cleanliness at least 0.5. If the geometry holds but cleanliness is too low, the outcome is UNCLEAN_TAKEOFF, not success.")
p("Other terminal labels include RUNWAY_EXCURSION, FAILED_ROTATION, INSUFFICIENT_RUNWAY, LOSS_OF_CONTROL, UNSTABLE_AFTER_LIFTOFF, CRASH, SIMULATOR_INVALID_STATE, ENVELOPE_VIOLATION and FAILED_TO_ACCELERATE. Reaching the 180-second time limit is a truncation: the experiment was cut off, rather than one of those task-terminal outcomes. This distinction matters to reinforcement-learning algorithms.")
p("The order of checks matters. A stable completed success is checked first. Then crash/return, loss of control, unstable angle of attack, envelope limits, runway excursion, runway exhaustion, failed acceleration and timeout are checked. A 5-step liftoff confirmation avoids classifying one noisy wheel-contact sample as a real departure.")

h1("12. One action in the real code, line by line")
p("This is the central path in envs/boeing7478_takeoff_env.py. The line numbers are from the inspected repository. Read this once, then follow the calls in your editor.")
table([
 ["Lines", "What the code does", "Why we do it"],
 ["225, 231-232", "Enter step(action); reject calls made before reset", "A simulator state must exist before movement."],
 ["233", "Apply the action mapper", "Convert the four requested actions into bounded, rate-limited commands."],
 ["234", "Remember the previous state", "We need old and new states to measure acceleration and progress."],
 ["236", "Send throttle, surfaces and steering to JSBSim", "Put the command into the physics engine."],
 ["237", "Update wind gust", "Weather can change during the episode."],
 ["238", "Run physics_steps_per_action", "At 120/10 Hz, this means 12 small physics steps."],
 ["239-241", "Read sensors, derive flight state and normalize observation", "Produce the next 27 inputs for the policy."],
 ["242-243", "Handle invalid simulator state", "Return a named failure instead of hiding bad numbers."],
 ["244-248", "Count the transition and summarize motion/cleanliness", "Track progress and whole-flight comfort."],
 ["249", "Update consecutive success-region counter", "A single good reading is not enough."],
 ["250-252", "Check success, named failures and timeout", "Know whether this attempt is over."],
 ["253-257", "Build reward context and sum component rewards", "Give SAC readable feedback."],
 ["258-261", "Update statistics and next-step tracker fields", "Keep complete episode measurements."],
 ["262-265", "Store new state and return the Gymnasium five-tuple", "The agent can make its next decision."],
])
code("obs, reward, terminated, truncated, info = env.step(action)\n# obs: 27 numbers for the next decision\n# reward: score for this transition\n# terminated: real success or task failure\n# truncated: the time limit cut off the attempt\n# info: readable physics, reason, components and final summary")
p("In reset(), lines 179-204 choose either fixed conditions or a seeded random draw, start JSBSim, reset actuators and counters, build the first observation and return the sampled conditions. This is why two runs with the same seed can reproduce the same scenario while different seeds test generality.")

h1("13. Four more small code paths, line by line")
table([
 ["File and lines", "Read it as", "Why it is there"],
 ["observations.py 128-142", "Make the 27 values in OBS_NAMES order; divide by scales", "Comparable, fixed-size neural-network input."],
 ["observations.py 143-147", "Check finite, clip to +/-5, cast float32", "Catch invalid simulator readings and match Gym space."],
 ["actions.py 84-88", "Validate, map throttle, limit per-step change", "Keep commands bounded and smooth."],
 ["actions.py 90-96", "Apply verified JSBSim sign corrections", "Agent control directions remain intuitive."],
 ["termination.py 46-56", "Check all success conditions at one instant", "Defines the real task objective."],
 ["termination.py 58-77", "Count confirmed liftoff and consecutive success steps", "Reject brief false-positive readings."],
 ["termination.py 109-149", "Check success, failures and timeout in priority order", "Every episode receives a precise reason."],
 ["reward.py 231-247", "Call every component method, sum values", "A readable reward audit trail."],
])
p("Why separate state, reward and termination files? They answer different questions. State says 'what happened?' Reward says 'how useful was it?' Termination says 'is this attempt over, and why?' Keeping them separate makes failures easier to diagnose and prevents a reward edit from silently changing the success definition.")

h1("14. SAC training code, line by line")
table([
 ["train_sac.py lines", "Plain explanation", "Why"],
 ["34-43", "Construct SAC from YAML values, network layers and device", "One auditable place for learning settings."],
 ["46-62", "Parse config, resume, seed, demos and reward-profile options", "Experiments should be reproducible from commands."],
 ["72-86", "Load YAML, apply command-line overrides, create output dirs, choose CPU/GPU", "Keep files and settings tied to one named run."],
 ["88-107", "Create curriculum/milestone trackers; restore them on resume", "Do not silently restart difficulty or milestones."],
 ["109-124", "Create environment; load model and optional replay buffer or build a new model", "Resume the learning state as fully as possible."],
 ["126-134", "Load teacher demonstrations and optionally clone actor actions", "Give early training useful examples."],
 ["136-149", "Create logging and evaluation callbacks; run remaining transitions", "Let SAC learn and save measurable progress."],
 ["150-169", "Handle stop/Ctrl+C, save model and metrics, close environment", "Make interruptions recoverable."],
])
p("The replay buffer is crucial on resume. A .zip holds network weights and algorithm settings; the matching _replay_buffer.pkl holds past experience. The _state.json holds curriculum level, milestone history and reward profile. GitHub excludes replay buffers because they are huge; a collaborator can run and evaluate the pushed models, but an exact training continuation needs the locally saved buffer as well.")
p("The curriculum manager's update method is compact: it checks the current level is below max, has enough evaluation episodes, and the success rate is strictly greater than the configured threshold; then it increments the level. The evaluation callback uses fixed seeds, logs CSV rows, records milestones, saves best models and can stop an unproductive run after 500,000 transitions without any success.")

h1("15. Learn from real success and failure")
fig("teacher_success_dashboard.png", "SUCCESS: the teacher's centerline error stays small, speed rises, and height climbs. The control traces are comparatively calm.")
p("Read the top-down trace first: it stays near the runway middle. Read airspeed next: it rises to rotation. Then read height and climb: they continue upward. Finally check pitch, roll and controls: no large oscillation is needed. This dashboard records an actual successful teacher simulation, not a hypothetical example.")
fig("training_success_3d.png", "SUCCESS IN 3D: a recorded training-success replay; the event list contains multiple real successful episodes.")
fig("early_agent_failure.png", "FAILURE: an early SAC policy wanders off the runway; the controls swing rapidly and the altitude does not show a successful climb.")
p("On the failure dashboard, centerline error grows instead of shrinking and control traces become jagged. The right response is to inspect action signs, rate limits, lateral reward components and the observation values on that trajectory. A failure image is a debugging clue, not proof that the whole algorithm is broken.")

h1("16. How to rebuild from an empty folder")
p("A clean rebuild needs Python 3.10-3.13, Git, the dependencies in requirements.txt, and a terminal. The project uses a Python virtual environment to keep packages separate. On Windows PowerShell, the following is the shortest reproducible path.")
code("git clone https://github.com/malekawlaqi/Autonomous-airplane-project.git\ncd Autonomous-airplane-project\\boeing7478_rl_takeoff\npython -m venv .venv\n.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n.venv\\Scripts\\python.exe tools\\inspect_aircraft.py\n.venv\\Scripts\\python.exe tools\\check_environment.py\n.venv\\Scripts\\python.exe -m pytest")
p("The clone command needs access to the private GitHub repository. If PyTorch GPU support is desired, follow the project's README for the compatible CUDA wheel; CPU mode works for setup checks but training speed can differ. Run commands from the boeing7478_rl_takeoff directory so relative config and output paths resolve correctly.")
p("To reimplement the *software* from scratch rather than clone it, build in this order: (1) YAML configs and an honest aircraft-model note, (2) JSBSim adapter with verified sensor/control properties, (3) runway frame and seeded randomizer, (4) observation builder, (5) action mapper, (6) success/failure tracker, (7) reward components, (8) Gymnasium reset/step, (9) teacher and baselines, (10) environment tests, (11) SAC driver, (12) evaluation/curriculum/checkpoint callbacks, and (13) replay/report tools. Validate each small layer before adding the next.")
table([
 ["Build milestone", "Small proof before moving on"],
 ["Simulator adapter", "Load model; read every required property; step physics without NaN."],
 ["Runway geometry", "Forward distance increases; right-side error has the expected sign."],
 ["Actions", "Measured nose/roll/yaw move in the documented positive directions."],
 ["Gym environment", "Reset gives 27 finite values; step gives a bounded observation and four controls."],
 ["Teacher", "At least one valid takeoff under easy conditions."],
 ["SAC", "Checkpoint, evaluate, interrupt and resume all work."],
 ["Generalization", "Compare several fixed seeds and harder levels to teacher and full-throttle baselines."],
])

h1("17. First hands-on experiment")
p("Begin by watching the teacher and a simple full-throttle baseline. This confirms whether the environment is solvable and shows that takeoff alone does not prove learning. Then generate teacher demonstrations, train a *new* SAC run, inspect the CSV and replay. Avoid reusing the active run name if the original trainer is still writing to it.")
code(".venv\\Scripts\\python.exe -m evaluation.evaluate --policy teacher --episodes 20 --level 1\n.venv\\Scripts\\python.exe -m evaluation.evaluate --policy full_throttle --episodes 20 --level 1\n.venv\\Scripts\\python.exe -m teacher.demonstrations --episodes 200 --level 1 --out results/demos/teacher_demos.npz\n.venv\\Scripts\\python.exe -m training.train_sac --config configs/sac.yaml --run-name beginner_rebuild --demos results/demos/teacher_demos.npz --bc-epochs 20 --total-timesteps 500000")
p("A training transition is one action plus 12 physics steps, about 0.1 simulated seconds. A 50-second episode is roughly 500 transitions. The 500,000-transition target is therefore many whole attempts, not 500,000 rendered video frames. Training uses headless physics for speed; 3D replay records trajectories later.")

h1("18. How to judge whether learning is real")
p("Use deterministic evaluation with fixed seeds separate from training. Report success rate and its episode count, takeoff time, runway distance, centerline RMS/max error, cleanliness and failure reasons. Compare with the teacher and full-throttle baseline on the same difficulty. Repeat the training with at least three training seeds before claiming one method is better. A 20-episode result is a progress signal, not a final confidence estimate.")
p("The reward can be gamed. A policy might earn progress points while staying unsafe, or look good on level 1 while failing in crosswind. That is why the project has separate terminal success rules, comfort metrics, baseline policies, held-out evaluation seeds and robustness tests. A curriculum level-up may cause the displayed success rate to drop because the exam became harder.")
fig("evaluation_progress.png", "Evaluation success over transitions. Interpret each point together with its curriculum level and episode count.")

h1("19. Where this project stands today")
p(f"The included report snapshot was captured at {SNAP['captured_at']}. The named run was {SNAP['run']}, with {SNAP['num_timesteps']:,} transitions, {SNAP['training_episodes']} training episodes and {SNAP['training_successes']} training successes. It was on curriculum level {SNAP['curriculum_level']}. The latest 100 training episodes had {SNAP['rolling_success_rate']:.0%} success; the latest deterministic level-{SNAP['latest_evaluation']['level']} evaluation had {SNAP['latest_evaluation']['success_rate']:.0%} success over {SNAP['latest_evaluation']['episodes']} episodes. The first successful training episode occurred at {SNAP['milestones']['steps_to_first_success']:,} transitions. Training may have continued after this snapshot.")
p("What this means: the system learned complete takeoffs at easier levels and has reached level 3, but level-3 reliability is still low. The next developer should diagnose level-3 outcomes, especially unclean takeoffs and insufficient acceleration, with fixed-seed comparisons. The repository is a research prototype, not a finished robust pilot.")
fig("recent_outcomes.png", "A snapshot of recent outcomes. Look beyond the total success count and inspect each failure category.")

h1("20. Practical continuation and common questions")
table([
 ["Question", "Answer"],
 ["Where do I start reading?", "README, this guide, aircraft notes, then envs/boeing7478_takeoff_env.py."],
 ["What if check_environment fails?", "Inspect the named check; verify JSBSim, config files and aircraft property names before training."],
 ["Why does GPU use look low?", "JSBSim physics is CPU-bound; the GPU only helps neural-network updates."],
 ["Why does the viewer differ from training?", "Viewer uses a visual 747 model; flight physics comes from the JSBSim approximation."],
 ["Why can a neutral controller take off?", "The inherited model is forgiving; compare centerline, comfort and robustness, not lift-off alone."],
 ["Why is success lower after level-up?", "New evaluation conditions are harder. Read level and seed alongside rate."],
 ["Can I change reward weights and resume?", "Use a new run name/profile; a changed reward defines a different learning problem."],
 ["What is missing from GitHub?", "Virtual environments, logs, temp files and huge replay buffers; recreate or copy them separately."],
 ["What makes a resume exact?", "Matching model zip, replay-buffer pickle, state JSON, config and reward profile."],
 ["Can this control a real aircraft?", "No. The model is unvalidated and this is simulation research only."],
])
p("Next-work checklist: read the newest results; preserve the active trainer's files; run checks when the trainer is paused; evaluate the current checkpoint on a fixed set of level-3 seeds; count failure reasons; replay representative successes and failures; change one hypothesis at a time; test against teacher/full-throttle baselines; repeat over multiple training seeds; update the results table and report with dated measurements.")

h1("21. Source map and reading order")
p("This guide was checked against the repository files listed below. Follow the order for a code review: configuration and aircraft notes -> environment reset/step -> observation/action helpers -> termination/reward -> teacher -> SAC driver -> callbacks -> evaluation -> tests -> result CSVs. A source line number is a signpost for learning, not a promise that future commits keep the same numbering.")
table([
 ["Source", "Purpose"],
 ["README.md; AIRCRAFT_MODEL_NOTES.md; EXPERIMENT_PROTOCOL.md", "Scope, caveats, commands and experiment rules."],
 ["configs/*.yaml", "Physics rate, success limits, reward weights, aircraft overrides, SAC settings and randomization."],
 ["envs/*.py", "The Gymnasium task, JSBSim bridge, runway math, sensors, actions, reward and outcomes."],
 ["teacher/*.py", "Simple controller, demonstration collection and behavior cloning."],
 ["training/*.py", "SAC training, curriculum, logging, evaluation and checkpointing."],
 ["evaluation/*.py", "Policy exams, robustness, failure tables and replay trajectories."],
 ["tests/*.py; tools/check_environment.py", "Automated checks of shapes, signs, reward arithmetic and episode behavior."],
 ["results/replay/*.png; results/replay3d/*.png", "The actual success/failure visual evidence used here."],
])
p("The UM6P College of Computing and Forgebots marks on the cover were extracted from the existing club report in this repository, so this handbook uses the same visual identity.")

h1("22. Reimplementation acceptance checklist")
p("A rebuild is complete only when another person can run the same checks and explain the results. Use this short checklist as the last page of the handover.")
table([
 ["Check", "Expected evidence"],
 ["Aircraft identity", "The inspector prints 7478_RESEARCH_APPROXIMATION and names the generated model."],
 ["Gymnasium interface", "reset returns 27 finite values; step returns observation, reward, terminated, truncated and info."],
 ["Control directions", "Positive nose-up, right-roll and right-yaw agent commands move the simulator in the documented directions."],
 ["Reproducible weather", "A fixed seed gives the same episode conditions and a changed seed gives a different draw."],
 ["Teacher and baselines", "The teacher reaches at least one valid takeoff; full-throttle baseline is measured too."],
 ["Training recovery", "A checkpoint saves model zip, state JSON and local replay buffer; interruption and resume are tested."],
 ["Honest evaluation", "Fixed unseen seeds produce a success rate, episode count, failure table and comfort metrics."],
])
p("If a check fails, stop at that layer and repair it before a long training run. This prevents a neural network from learning around a simulator bug. Keep the 3D viewer as a way to understand recorded flights, while using the numeric evaluator as the source of performance claims.")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitleX", parent=styles["Title"], fontName="CMBold", fontSize=25, leading=31, textColor=colors.black, alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="CoverSubX", parent=styles["Title"], fontName="CMRoman", fontSize=17, leading=22, textColor=colors.black, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="FrontHeadingX", parent=styles["Heading1"], fontName="CMBold", fontSize=13, leading=16, textColor=colors.black, alignment=TA_CENTER, spaceAfter=16))
styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontName="CMBold", fontSize=14, leading=18, textColor=colors.black, spaceBefore=15, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], fontName="CMBold", fontSize=11, leading=15, textColor=colors.black, spaceBefore=10, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontName="CMRoman", fontSize=10, leading=14.5, spaceAfter=8))
styles.add(ParagraphStyle(name="BulletX", parent=styles["BodyText"], fontName="CMRoman", fontSize=10, leading=14.5, leftIndent=12, firstLineIndent=-7, spaceAfter=5))
styles.add(ParagraphStyle(name="CaptionX", parent=styles["BodyText"], fontName="CMRoman", fontSize=8.5, leading=12, textColor=colors.black, spaceBefore=4, spaceAfter=11))
styles.add(ParagraphStyle(name="CalloutX", parent=styles["BodyText"], fontName="CMBold", fontSize=9.5, leading=14.5, textColor=colors.black, backColor=colors.HexColor("#f3f3f3"), borderPadding=8, spaceBefore=8, spaceAfter=9))
styles.add(ParagraphStyle(name="CellX", parent=styles["BodyText"], fontName="CMRoman", fontSize=8.2, leading=11.3))
styles.add(ParagraphStyle(name="CellHeadX", parent=styles["BodyText"], fontName="CMBold", fontSize=8.3, leading=11.3, textColor=colors.black))
styles.add(ParagraphStyle(name="CodeX", fontName="CMMono", fontSize=7.3, leading=10.5, backColor=colors.HexColor("#f6f6f6"), borderPadding=7, spaceBefore=4, spaceAfter=9))
styles.add(ParagraphStyle(name="TocX", fontName="CMRoman", fontSize=10.5, leading=17, leftIndent=8, firstLineIndent=-8, rightIndent=10, spaceAfter=4))

def xml(s): return html_escape(str(s).replace("->", " then ")).replace("\n", "<br/>")
def fit_image(path: Path, max_w_mm: float, max_h_mm: float):
    w, h = PILImage.open(path).size
    scale = min(max_w_mm * mm / w, max_h_mm * mm / h)
    return Image(str(path), width=w*scale, height=h*scale)

logo_table=Table([[fit_image(BRAND/"cover_logo_1.png",100,28),fit_image(BRAND/"cover_logo_2.png",39,39)],
                  ["",Paragraph("Forgebots",styles["FrontHeadingX"])]],colWidths=[111*mm,52*mm],hAlign="CENTER")
logo_table.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
story=[Spacer(1,27*mm),logo_table,Spacer(1,35*mm),HRFlowable(width="100%",thickness=0.8,color=colors.black),Spacer(1,14*mm),
       Paragraph("Teaching a Giant Airliner to Take Off",styles["CoverTitleX"]),
       Paragraph("A from-zero guide to reinforcement learning, this project, and rebuilding it step by step",styles["CoverSubX"]),
       Spacer(1,17*mm),HRFlowable(width="100%",thickness=0.8,color=colors.black),Spacer(1,35*mm),
       Paragraph("Project handbook: 8 October 2026",styles["CoverSubX"]),PageBreak(),
       Paragraph("Abstract",styles["FrontHeadingX"]),
       Paragraph("One sentence: A simulated pilot learns to control throttle, elevator, aileron and rudder so a research approximation of a Boeing 747-8 can perform a clean, stable takeoff in changing conditions.",styles["BodyX"]),
       Spacer(1,5*mm),fit_image(ASSETS/"simulation_3d.png",165,100),
       Paragraph("Figure 1: The project replay viewer. The displayed aircraft does not supply the flight physics; JSBSim does.",styles["CaptionX"]),
       Paragraph("How to use this report",styles["H1X"]),
       Paragraph("Read sections 1-5 for the basic idea, 6-14 for the exact implementation, 15 for visual evidence, and 16-22 when you want to rebuild or continue the work. No machine-learning background is required.",styles["BodyX"]),PageBreak(),
       Paragraph("Contents",styles["H1X"])]
toc=TableOfContents(); toc.levelStyles=[styles["TocX"]]; story += [toc,PageBreak()]
for block in blocks[6:]:
    kind=block[0]
    if kind=="h1": story.append(Paragraph(xml(block[1]),styles["H1X"]))
    elif kind=="h2": story.append(Paragraph(xml(block[1]),styles["H2X"]))
    elif kind=="p": story.append(Paragraph(xml(block[1]),styles["BodyX"]))
    elif kind=="bullet": story.append(Paragraph("- "+xml(block[1]),styles["BulletX"]))
    elif kind=="callout": story.append(Paragraph(xml(block[1]),styles["CalloutX"]))
    elif kind=="code": story.append(Preformatted(block[1],styles["CodeX"],maxLineLength=102))
    elif kind=="page": story.append(PageBreak())
    elif kind=="fig":
        path=ASSETS/block[1]
        story.append(KeepTogether([fit_image(path,block[3],105),Paragraph(xml(block[2]),styles["CaptionX"])]))
    elif kind=="table":
        rows=block[1]; widths=block[2] or ([39,36,90] if len(rows[0])==3 else [47,118])
        data=[[Paragraph(xml(cell),styles["CellHeadX"] if i==0 else styles["CellX"]) for cell in row] for i,row in enumerate(rows)]
        t=Table(data,colWidths=[v*mm for v in widths],repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eeeeee")),("VALIGN",(0,0),(-1,-1),"TOP"),("LINEABOVE",(0,0),(-1,0),0.7,colors.black),("LINEBELOW",(0,0),(-1,0),0.7,colors.black),("LINEBELOW",(0,-1),(-1,-1),0.7,colors.black),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#fafafa")]),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,4*mm)])

def footer(canvas,doc):
    if doc.page==1: return
    canvas.saveState(); canvas.setStrokeColor(colors.black); canvas.line(20*mm,16*mm,190*mm,16*mm)
    canvas.setFont("CMRoman",8); canvas.setFillColor(colors.black)
    canvas.drawString(20*mm,12*mm,"Boeing 747-8 RL Takeoff - from-zero project handbook")
    canvas.drawRightString(190*mm,12*mm,str(doc.page)); canvas.restoreState()

class GuideDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=="H1X" and flowable.getPlainText() not in {"Contents", "How to use this report"}:
            title=flowable.getPlainText()
            key="section-"+hashlib.md5(title.encode("utf-8")).hexdigest()
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry",(0,title,self.page,key))

doc=GuideDoc(str(PDF),pagesize=A4,rightMargin=20*mm,leftMargin=20*mm,topMargin=19*mm,bottomMargin=21*mm,title="Teaching a Giant Airliner to Take Off - From Zero",author="Forgebots / UM6P College of Computing")
doc.multiBuild(story,onFirstPage=footer,onLaterPages=footer)

def tex_escape(s):
    repl={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$","#":r"\#","_":r"\_","{":r"\{","}":r"\}","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    return "".join(repl.get(c,c) for c in str(s))

tex=[r"\documentclass[11pt,a4paper]{article}",r"\usepackage[margin=20mm]{geometry}",r"\usepackage[T1]{fontenc}",r"\usepackage{graphicx}",r"\usepackage{longtable}",r"\usepackage{array}",r"\usepackage{fancyvrb}",r"\usepackage{hyperref}",r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black}",r"\setlength{\parskip}{0.35em}",r"\begin{document}",
     r"\begin{titlepage}",r"\vspace*{25mm}",r"\begin{center}",
     r"\includegraphics[width=0.56\linewidth]{assets/cover_logo_1.png}\hfill\includegraphics[width=0.24\linewidth]{assets/cover_logo_2.png}\\[3mm]",
     r"\hfill{\bfseries Forgebots}\\[35mm]",r"\rule{\linewidth}{0.7pt}\\[15mm]",
     r"{\Huge\bfseries Teaching a Giant Airliner to Take Off}\\[12mm]",
     r"{\Large A from-zero guide to reinforcement learning, this project, and rebuilding it step by step}\\[16mm]",
     r"\rule{\linewidth}{0.7pt}\\[35mm]",r"{\large Project handbook: 8 October 2026}",r"\end{center}",r"\end{titlepage}",
     r"\begin{abstract}","One sentence: A simulated pilot learns to control throttle, elevator, aileron and rudder so a research approximation of a Boeing 747-8 can perform a clean, stable takeoff in changing conditions.",r"\end{abstract}",
     r"\begin{figure}[htbp]\centering\includegraphics[width=0.9\linewidth]{../assets/simulation_3d.png}\caption{The project replay viewer. The displayed aircraft does not supply the flight physics; JSBSim does.}\end{figure}",
     r"\section*{How to use this report}","Read sections 1-5 for the basic idea, 6-14 for the exact implementation, 15 for visual evidence, and 16-22 when you want to rebuild or continue the work. No machine-learning background is required.",
     r"\clearpage",r"\tableofcontents",r"\clearpage"]
for block in blocks[6:]:
    kind=block[0]
    if kind=="h1": tex.append(r"\section{"+tex_escape(re.sub(r"^\d+\.\s*", "", block[1]))+"}")
    elif kind=="h2": tex.append(r"\subsection{"+tex_escape(block[1])+"}")
    elif kind in ("p","callout"): tex.append(tex_escape(block[1])+"\n")
    elif kind=="bullet": tex.append(r"\noindent\textbullet\ "+tex_escape(block[1])+"\n")
    elif kind=="code": tex.append(r"\begin{Verbatim}[fontsize=\small]"+"\n"+block[1]+"\n"+r"\end{Verbatim}")
    elif kind=="page": tex.append(r"\clearpage")
    elif kind=="fig": tex.append(r"\begin{figure}[htbp]\centering\includegraphics[width=0.96\linewidth]{../assets/"+block[1]+r"}\caption{"+tex_escape(block[2])+r"}\end{figure}")
    elif kind=="table":
        rows=block[1]; n=len(rows[0]); spec="|"+"|".join(["p{"+str(round(0.94/n,3))+r"\linewidth}"]*n)+"|"
        tex.append(r"\begin{longtable}{"+spec+r"}\hline")
        for row in rows: tex.append(" & ".join(tex_escape(x) for x in row)+r" \\ \hline")
        tex.append(r"\end{longtable}")
tex.append(r"\end{document}")
TEX.write_text("\n".join(tex),encoding="utf-8")
print(f"PDF: {PDF}\nTEX: {TEX}\nBLOCKS: {len(blocks)}")
