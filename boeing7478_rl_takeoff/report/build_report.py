from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import sys
sys.path.append(r"C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages")
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
ASSETS = REPORT / "assets"
OUT = REPORT / "output"
TMP = REPORT / "tmp" / "pdfs"
for folder in (ASSETS, OUT, TMP):
    folder.mkdir(parents=True, exist_ok=True)

STATUS_PATH = ROOT / "results" / "clean_teacher_lv4" / "showcase" / "status.json"
EVAL_PATH = ROOT / "results" / "clean_teacher_lv4" / "eval_history.csv"
MILESTONES_PATH = ROOT / "results" / "clean_teacher_lv4" / "milestones.json"
TRAIN_PATH = ROOT / "results" / "clean_teacher_lv4" / "training_episodes.csv"

status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
milestones = json.loads(MILESTONES_PATH.read_text(encoding="utf-8"))
snapshot_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

with EVAL_PATH.open(newline="", encoding="utf-8") as stream:
    eval_rows = list(csv.DictReader(stream))
with TRAIN_PATH.open(newline="", encoding="utf-8") as stream:
    train_rows = list(csv.DictReader(stream))

latest_steps = int(status["num_timesteps"])
latest_level = int(status["level"])
latest_eval = status.get("eval", {})
episodes = int(status.get("episodes", 0))
successes = int(status.get("training_successes", 0))
rolling = float(status.get("rolling_success_rate", 0.0))


def save_project_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.3, 2.0, 1.9, 1.1, "Random world\nwind + runway + mass", "#d9edf7"),
        (2.7, 2.0, 1.9, 1.1, "JSBSim\nflies the aircraft", "#dff0d8"),
        (5.1, 2.0, 1.9, 1.1, "27 sensors\nwhat the agent sees", "#fcf8e3"),
        (7.5, 2.0, 1.9, 1.1, "SAC brain\nchooses 4 controls", "#f2dede"),
        (9.9, 2.0, 1.9, 1.1, "Reward + result\nlearn, save, repeat", "#e8ddf5"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#24445f", linewidth=1.8))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=11, weight="bold")
    for x in (2.2, 4.6, 7.0, 9.4):
        ax.annotate("", xy=(x + 0.45, 2.55), xytext=(x, 2.55), arrowprops=dict(arrowstyle="->", lw=2, color="#24445f"))
    ax.annotate("repeat thousands of times", xy=(8.5, 1.6), xytext=(3.5, 1.0),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.35", lw=2, color="#e17c05"),
                color="#b65f00", fontsize=11, weight="bold")
    ax.set_title("The whole project in one picture", fontsize=17, weight="bold", color="#173b57", pad=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "project_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_takeoff_story() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 3)
    ax.axis("off")
    stages = [(3, "1. Accelerate", "Stay centered"), (25, "2. Rotate", "Lift the nose near VR"),
              (47, "3. Lift off", "Positive climb"), (68, "4. Reach 1,000 ft", "Remain stable"),
              (88, "5. Hold 5 seconds", "Success if clean")]
    ax.plot([5, 95], [0.8, 0.8], lw=8, color="#c9d8e3", solid_capstyle="round")
    for i, (x, title, note) in enumerate(stages):
        ax.scatter([x], [0.8], s=520, color="#2b7bba", edgecolor="white", linewidth=2.5, zorder=3)
        ax.text(x, 0.8, str(i+1), ha="center", va="center", color="white", fontsize=12, weight="bold")
        ax.text(x, 1.65 if i % 2 == 0 else -0.1, title, ha="center", fontsize=11, weight="bold", color="#173b57")
        ax.text(x, 1.32 if i % 2 == 0 else -0.43, note, ha="center", fontsize=9, color="#445b6b")
    ax.set_title("What one successful episode must do", fontsize=17, weight="bold", color="#173b57")
    fig.tight_layout()
    fig.savefig(ASSETS / "takeoff_story.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_eval_progress() -> None:
    xs, ys, levels = [], [], []
    for row in eval_rows:
        if not row.get("num_timesteps"):
            continue
        xs.append(float(row["num_timesteps"]) / 1e6)
        ys.append(float(row["success_rate"]) * 100)
        levels.append(int(float(row["level"])))
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    colors_by_level = {1: "#2e8b57", 2: "#e18b13", 3: "#7a4eb2", 4: "#b33b3b", 5: "#333333"}
    for level in sorted(set(levels)):
        xx = [x for x, lv in zip(xs, levels) if lv == level]
        yy = [y for y, lv in zip(ys, levels) if lv == level]
        ax.scatter(xx, yy, label=f"curriculum level {level}", color=colors_by_level[level], s=32, zorder=3)
        ax.plot(xx, yy, color=colors_by_level[level], alpha=0.35)
    ax.axhline(80, ls="--", lw=1.5, color="#b33b3b", label="advance target: above 80%")
    ax.axvline(0.225, ls=":", color="#2e8b57")
    ax.axvline(1.275, ls=":", color="#7a4eb2")
    ax.text(0.225, 103, "level 1 -> 2", ha="center", fontsize=9)
    ax.text(1.275, 103, "level 2 -> 3", ha="center", fontsize=9)
    ax.set(xlabel="RL transitions (millions)", ylabel="deterministic evaluation success (%)", ylim=(-5, 110))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Learning progress: harder levels reset the difficulty", weight="bold", color="#173b57")
    fig.tight_layout()
    fig.savefig(ASSETS / "evaluation_progress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_recent_outcomes() -> None:
    reasons = status.get("recent_reasons", {})
    order = sorted(reasons, key=lambda key: reasons[key], reverse=True)
    values = [reasons[k] for k in order]
    labels = [k.replace("_", " ").title() for k in order]
    bar_colors = ["#2e8b57" if k == "SUCCESS" else "#d9826b" if k == "UNCLEAN_TAKEOFF" else "#7796ad" for k in order]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1])
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.5, bar.get_y() + bar.get_height()/2, str(value), va="center", fontsize=9)
    ax.set_xlabel("episodes in the latest rolling window")
    ax.set_title("What happened in the latest 100 training episodes", weight="bold", color="#173b57")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "recent_outcomes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def copy_simulation_images() -> None:
    mapping = {
        ROOT / "results" / "replay3d" / "newelves.png": ASSETS / "simulation_3d.png",
        ROOT / "results" / "replay3d" / "ui_test.png": ASSETS / "training_success_3d.png",
        ROOT / "results" / "replay3d" / "wheels_side.png": ASSETS / "successful_runway_start_side.png",
        ROOT / "results" / "replay" / "teacher_seed1000000.png": ASSETS / "teacher_success_dashboard.png",
        ROOT / "results" / "replay" / "sac_pilot_150k_L1.png": ASSETS / "early_agent_failure.png",
    }
    for source, target in mapping.items():
        shutil.copy2(source, target)


save_project_flow()
save_takeoff_story()
save_eval_progress()
save_recent_outcomes()
copy_simulation_images()

snapshot = {
    "captured_at": snapshot_time,
    "run": status.get("run_name"),
    "num_timesteps": latest_steps,
    "curriculum_level": latest_level,
    "training_episodes": episodes,
    "training_successes": successes,
    "rolling_success_rate": rolling,
    "latest_evaluation": latest_eval,
    "milestones": milestones,
    "note": "Training was active when this snapshot was captured; later values may be larger.",
}
(REPORT / "status_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

TEX = r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.9cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{fancyhdr}
\hypersetup{colorlinks=true,urlcolor=blue!60!black,linkcolor=blue!60!black}
\pagestyle{fancy}\fancyhf{}\lhead{Boeing 747-8 RL Takeoff}\rhead{Project handover}\cfoot{\thepage}
\lstset{basicstyle=\ttfamily\small,breaklines=true,frame=single,backgroundcolor=\color{gray!7}}
\title{Teaching a Giant Airliner to Take Off\\\large A Friendly Guide to the Boeing 747-8 Reinforcement-Learning Project}
\author{Project handover report}\date{STATUS_DATE}
\begin{document}\maketitle
\begin{center}\fcolorbox{blue!45!black}{blue!5}{\parbox{0.88\linewidth}{\textbf{One sentence:} A simulated pilot learns to control throttle, elevator, aileron and rudder so a research approximation of a Boeing 747-8 can perform a clean, stable takeoff in many changing conditions.}}\end{center}
\tableofcontents\clearpage
\section{The one-minute story}
Imagine a student pilot practising the same takeoff thousands of times. The pilot sees 27 instrument values, moves four controls, and receives a score. Good actions earn points; dangerous or messy actions lose points. After every attempt, the SAC algorithm adjusts a neural network so the next attempt can be better. A hand-written ``teacher pilot'' supplies good examples so learning does not start from pure chaos.
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{assets/project_flow.png}\caption{The learning loop. Remember: \textbf{world - physics - sensors - brain - score}.}\end{figure}
\section{What success means}
The goal is not just ``leave the ground.'' The aircraft must accelerate, stay near the centerline, rotate around the estimated rotation speed, climb past 1,000 ft AGL, remain stable for five seconds, and keep a cleanliness score of at least 0.5.
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{assets/takeoff_story.png}\caption{A complete successful episode.}\end{figure}
\section{The cast of characters}
\begin{longtable}{p{0.22\linewidth}p{0.70\linewidth}}\toprule
\textbf{Word} & \textbf{Plain-English meaning}\\\midrule
Agent & The neural-network pilot.\\
Environment & The practice world that runs the flight and judges it.\\
Observation & The 27-number instrument panel presented to the agent.\\
Action & Four commands: throttle, elevator, aileron and rudder.\\
Reward & The score that says ``more like this'' or ``less like this.''\\
Episode & One takeoff attempt, from the runway start to success or failure.\\
Teacher & A conventional hand-tuned controller that demonstrates sensible takeoffs.\\
Curriculum & Five difficulty levels that add harder wind, runway and aircraft conditions.\\
Checkpoint & A saved brain plus replay buffer and training state.\\\bottomrule\end{longtable}
\section{The aircraft: useful, but not a real certified 747-8}
The physics engine is JSBSim. Because no validated JSBSim 747-8 model was found, the project starts from JSBSim's Boeing 747-400 model and replaces a limited set of values with published 747-8 numbers: wingspan, wing area, empty weight, fuel capacity and engine thrust. Aerodynamic tables, inertia, landing gear and many engine details still come from the older 747-400 model. The project therefore labels the aircraft \texttt{7478\_RESEARCH\_APPROXIMATION}. Results are for simulation research only.
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{assets/simulation_3d.png}\caption{The 3D replay viewer. The GLB is a visual stand-in only and does not change the flight physics.}\end{figure}
\section{What the agent sees and controls}
The 27 observations include speed, height, climb rate, pitch, roll, heading, angle of attack, sideslip, body rates, runway errors, distance, wheel contact, actuator positions, engine state, mass, centre of gravity, wind and air-density ratio. Values are normalised and clipped so the neural network receives numbers of a predictable size.

The action vector has four values in $[-1,1]$: throttle, elevator, aileron and rudder. An actuator layer clips invalid commands, rate-limits movement, maps throttle to $[0,1]$, converts signs to JSBSim conventions, and links rudder pedals to nose-wheel steering. The policy acts at 10 Hz; JSBSim runs 12 smaller physics steps at 120 Hz for each policy action.
\section{How the score teaches clean flying}
The reward is a collection of readable terms. Positive terms reward runway progress, centreline tracking, correct heading, acceleration, rotation, climbing, altitude progress, stability and final success. Negative terms penalise time, lateral error, excessive roll/heading/AoA/sideslip, control oscillation, premature rotation and terminal failures. The \texttt{clean} profile adds passenger-comfort terms for tight centreline tracking, low sideways motion, low lateral acceleration, low yaw/roll rates and a cleanliness bonus.

Memorise the rule as \textbf{fast enough, straight enough, smooth enough, stable enough}.
\section{How learning is organised}
The main algorithm is Soft Actor-Critic (SAC), with three 256-unit ReLU layers for actor and critics, a replay buffer of one million transitions and automatic entropy tuning. The teacher dataset is placed into the replay buffer, and behaviour cloning gives the actor a sensible starting point. SAC then continues learning from its own experience.

Curriculum advancement uses deterministic evaluation, never noisy training reward. The threshold is above 80 percent across at least 20 evaluation episodes. Level 1 is gentle; levels 2 and 3 widen the conditions; levels 4 and 5 include stronger wind, shorter or slippery runways, high elevation, gusts and turbulence.
\begin{figure}[h]\centering\includegraphics[width=0.96\linewidth]{assets/evaluation_progress.png}\caption{Evaluation success by training step. A drop after a level-up is expected because the exam becomes harder.}\end{figure}
\section{Current stage: STATUS_STEPS transitions}
The strongest active run is \texttt{clean\_teacher\_lv4}. At this snapshot it has completed STATUS_STEPS RL transitions, STATUS_EPISODES training episodes and STATUS_SUCCESSES training successes. Its rolling success rate over the latest 100 episodes is STATUS_ROLLING percent. It moved from level 1 to level 2 at 225,000 transitions, and from level 2 to level 3 at 1,275,000 transitions. The latest deterministic level-3 evaluation is STATUS_EVAL percent over 20 episodes. This means the system has learned convincing behaviour through level 2 and is now adapting to the harder level-3 conditions; it is not yet a final, fully robust policy.

Milestones: first successful training takeoff at 107,676 transitions; 10 percent rolling success at 249,695; 50 percent at 732,075; and 90 percent at 1,106,454 transitions.
\begin{figure}[h]\centering\includegraphics[width=0.92\linewidth]{assets/recent_outcomes.png}\caption{Latest rolling outcomes. ``Unclean takeoff'' means the aircraft reached the goal region but the whole flight was not smooth enough.}\end{figure}
\section{What a good and bad episode look like}
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{assets/teacher_success_dashboard.png}\caption{Successful teacher trajectory: centered runway track, smooth rotation and stable climb.}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{assets/training_success_3d.png}\caption{A recorded training success in the 3D replay. The green history list confirms repeated successful episodes.}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{assets/successful_runway_start_side.png}\caption{Side view of a successful replay at the runway start. Check alignment, wheel contact and heading before judging the later climb.}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{assets/early_agent_failure.png}\caption{Early learned-policy failure: the aircraft wanders, uses rough controls and exits the runway.}\end{figure}
\section{Project map: where to look}
\begin{longtable}{p{0.24\linewidth}p{0.66\linewidth}}\toprule
\textbf{Path} & \textbf{Responsibility}\\\midrule
\texttt{configs/} & Default aircraft, environment, reward, randomisation and SAC/PPO settings.\\
\texttt{configs\_975/} & Special 975,000-lb configuration for the MSFS-oriented experiment.\\
\texttt{envs/} & Gymnasium environment, JSBSim bridge, observations, actions, runway geometry, reward, termination and randomisation.\\
\texttt{training/} & SAC/PPO entry points, callbacks, checkpointing and curriculum logic.\\
\texttt{teacher/} & Conventional teacher controller and demonstration collection.\\
\texttt{evaluation/} & Evaluation, robustness tests, failure analysis and trajectory recording.\\
\texttt{tools/} & Environment checks, benchmarks, plots, 2D/3D replay and live viewer.\\
\texttt{tests/} & Unit, integration and showcase tests.\\
\texttt{models/} & Saved policy checkpoints, replay buffers and training state.\\
\texttt{results/} & CSV/JSON metrics, milestones, demonstrations, replay files and viewer events.\\
\texttt{logs/} & TensorBoard event files and console logs.\\
\texttt{msfs\_sac\_runtime.py} & Experimental SimConnect bridge that can read MSFS telemetry and optionally send policy controls.\\\bottomrule\end{longtable}
\section{How to reproduce the safe simulation workflow}
\begin{lstlisting}[language=bash]
cd C:\Users\asus\Documents\project\boeing7478_rl_takeoff
.venv\Scripts\Activate.ps1
python tools\check_environment.py
python -m pytest
tensorboard --logdir logs
python tools\live_viewer.py --open
\end{lstlisting}
To continue the active run, use a matching checkpoint, replay buffer and \texttt{\_state.json} file. Keep the same \texttt{clean} reward profile unless intentionally starting a new experiment. Example:
\begin{lstlisting}[language=bash]
python -m training.train_sac --config configs/sac.yaml \
  --run-name clean_teacher_lv4 \
  --resume models/clean_teacher_lv4/checkpoint_1350000.zip \
  --reward-profile clean --total-timesteps 2000000
\end{lstlisting}
Training was active while this report was produced. Stop or pause that process before launching another writer against the same run directory.
\section{Validation and honest limitations}
On the report date, \texttt{tools/check\_environment.py} passed all 10 checks: aircraft loading, shapes, physics/action timing, bounded observations, rate-limited actions, deterministic seeds, reward arithmetic, terminations, random actions and Gymnasium validation.

The full pytest run executed 79 tests: 72 passed and 7 showcase tests could not complete because the active training process held the generated aircraft XML while Windows also denied pytest's temporary directory. These are test-environment/locking errors, not seven confirmed behavioural failures; rerun pytest after stopping the training process.

Major scientific limitations remain: the aircraft model is an approximation; no validated 747-8 aerodynamics or Boeing performance tables are used; runway slope is unsupported; tyre friction is simplified; fuel does not move the centre of gravity; and current level-3 evaluation is still low. The MSFS bridge is experimental and fills several unavailable telemetry channels with neutral values, so it is not equivalent to the JSBSim training environment. It must remain simulation-only.
\section{The continuation checklist}
\begin{enumerate}
\item Let \texttt{clean\_teacher\_lv4} learn on level 3, but watch evaluation instead of training reward alone.
\item Preserve every checkpoint trio: model ZIP, replay-buffer PKL and state JSON.
\item Re-run 10/10 environment checks and the full tests after training stops.
\item Compare the learned policy with the teacher and full-throttle baselines on identical seeds.
\item Investigate the dominant level-3 failures, especially unclean takeoffs and failures to accelerate.
\item Run 1000+ robustness episodes only after the policy stabilises across levels 3 and 4.
\item Treat level 5 as a special slippery-runway challenge, not proof of real-aircraft capability.
\item Improve the MSFS telemetry mapping before evaluating transfer to Microsoft Flight Simulator.
\item Update this report's snapshot after the next major checkpoint or curriculum level-up.
\end{enumerate}
\section{Memory card}
\begin{center}\fcolorbox{orange!70!black}{orange!8}{\parbox{0.85\linewidth}{\centering\Large\textbf{SEE $\rightarrow$ STEER $\rightarrow$ SCORE $\rightarrow$ SAVE}\\[4pt]\normalsize 27 observations $\rightarrow$ 4 controls $\rightarrow$ reward and outcome $\rightarrow$ checkpoint and replay}}\end{center}

The project is currently a capable research simulator with a learning policy that has graduated to curriculum level 3. The next person can continue it by protecting the experiment state, evaluating on fixed seeds, fixing the test lock after training stops, and improving robustness before making any claim beyond simulation.
\section*{Primary project sources}
\texttt{README.md}; \texttt{AIRCRAFT\_MODEL\_NOTES.md}; \texttt{EXPERIMENT\_PROTOCOL.md}; YAML files in \texttt{configs/} and \texttt{configs\_975/}; Python modules in \texttt{envs/}, \texttt{training/}, \texttt{teacher/}, \texttt{evaluation/} and \texttt{tools/}; current status, milestones and CSV results under \texttt{results/clean\_teacher\_lv4/}.
\end{document}
'''

tex = TEX.replace("STATUS_DATE", snapshot_time)
tex = tex.replace("STATUS_STEPS", f"{latest_steps:,}")
tex = tex.replace("STATUS_EPISODES", f"{episodes:,}")
tex = tex.replace("STATUS_SUCCESSES", f"{successes:,}")
tex = tex.replace("STATUS_ROLLING", f"{rolling*100:.0f}")
tex = tex.replace("STATUS_EVAL", f"{float(latest_eval.get('success_rate', 0))*100:.0f}")
(REPORT / "Boeing7478_RL_Project_Guide.tex").write_text(tex, encoding="utf-8")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyFriendly", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.4, spaceAfter=7))
styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.2, textColor=colors.HexColor("#40596b")))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=11, leading=15, backColor=colors.HexColor("#eaf4fb"), borderColor=colors.HexColor("#2b7bba"), borderWidth=1, borderPadding=10, spaceAfter=10))
styles.add(ParagraphStyle(name="Memory", parent=styles["BodyText"], alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=colors.HexColor("#9a5000"), backColor=colors.HexColor("#fff5df"), borderColor=colors.HexColor("#e18b13"), borderWidth=1, borderPadding=12, spaceBefore=8, spaceAfter=12))
styles.add(ParagraphStyle(name="CodeSmall", fontName="Courier", fontSize=7.5, leading=9.5, backColor=colors.HexColor("#f4f6f8"), borderPadding=7, spaceAfter=8))
styles["Title"].fontName = "Helvetica-Bold"
styles["Title"].fontSize = 25
styles["Title"].leading = 30
styles["Title"].textColor = colors.HexColor("#173b57")
styles["Heading1"].textColor = colors.HexColor("#173b57")
styles["Heading1"].fontSize = 16
styles["Heading1"].leading = 19
styles["Heading1"].keepWithNext = True
styles["Heading2"].textColor = colors.HexColor("#2b6487")
styles["Heading2"].keepWithNext = True

def p(text: str, style: str = "BodyFriendly") -> Paragraph:
    return Paragraph(text, styles[style])

def fig(name: str, caption: str, max_h_mm: float = 105) -> KeepTogether:
    path = ASSETS / name
    with PILImage.open(path) as im:
        w, h = im.size
    width = 168 * mm
    height = width * h / w
    if height > max_h_mm * mm:
        scale = max_h_mm * mm / height
        width *= scale
        height *= scale
    return KeepTogether([Image(str(path), width=width, height=height), Spacer(1, 2*mm), p(caption, "Tiny")])

def table(rows, widths):
    data = [[p(str(cell), "Tiny") for cell in row] for row in rows]
    tab = Table(data, colWidths=widths, repeatRows=1)
    tab.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dbeaf4")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#173b57")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#9fb3c2")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return tab

story = [
    p("Teaching a Giant Airliner to Take Off", "Title"),
    p("A friendly guide to the Boeing 747-8 reinforcement-learning project", "Heading2"),
    p(f"Project handover snapshot: {snapshot_time}"), Spacer(1, 5*mm),
    p("One sentence: A simulated pilot learns to control throttle, elevator, aileron and rudder so a research approximation of a Boeing 747-8 can perform a clean, stable takeoff in many changing conditions.", "Callout"),
    fig("simulation_3d.png", "The project's 3D replay viewer. The detailed aircraft is a visual stand-in; the flight physics come from JSBSim."),
    p("How to use this report", "Heading1"),
    p("Read pages 2-4 to understand the idea. Read the current-stage pages to see what works today. Use the final checklist when you want to continue training or evaluation."),
    PageBreak(),
    p("1. The one-minute story", "Heading1"),
    p("Imagine a student pilot practising the same takeoff thousands of times. The pilot sees 27 instrument values, moves four controls, and receives a score. Good actions earn points; dangerous or messy actions lose points. After every attempt, SAC adjusts a neural network so the next attempt can improve. A hand-written teacher supplies good examples so learning does not start from pure chaos."),
    fig("project_flow.png", "The learning loop. Remember: world - physics - sensors - brain - score.", 78),
    p("2. What one successful takeoff must do", "Heading1"),
    p("Success is stricter than simply leaving the runway. The aircraft must pass the complete sequence below and remain clean enough over the whole trajectory."),
    fig("takeoff_story.png", "The five-step takeoff story.", 56),
]

vocab = [["Word", "Plain-English meaning"], ["Agent", "The neural-network pilot."], ["Environment", "The practice world that runs and judges the flight."], ["Observation", "The 27-number instrument panel."], ["Action", "Throttle, elevator, aileron and rudder."], ["Reward", "A score saying 'more like this' or 'less like this'."], ["Episode", "One complete takeoff attempt."], ["Teacher", "A conventional controller that demonstrates sensible takeoffs."], ["Curriculum", "Five levels that make conditions harder."], ["Checkpoint", "A saved brain, replay buffer and training state."]]
story += [p("3. The cast of characters", "Heading1"), table(vocab, [38*mm, 128*mm]), Spacer(1, 5*mm), p("4. The aircraft: useful, but not certified", "Heading1"), p("The physics engine is JSBSim. No validated JSBSim 747-8 model was found, so the project starts from JSBSim's Boeing 747-400 and replaces a limited set of public numbers: wingspan, wing area, empty weight, fuel capacity and engine thrust. Aerodynamics, inertia, gear and many engine details still come from the 747-400 model. The name 7478_RESEARCH_APPROXIMATION is an honest warning: simulation research only."), PageBreak(), p("5. What the agent sees and controls", "Heading1"), p("The 27 observations cover speed, height, climb rate, attitude, runway position, control positions, engine state, mass, centre of gravity, wind and air density. Values are normalised and clipped before the neural network sees them."), p("The action vector contains four values in [-1,1]. An actuator layer clips invalid commands, rate-limits movement, maps throttle to [0,1], fixes JSBSim sign conventions and connects rudder to nose-wheel steering. The policy acts at 10 Hz; JSBSim performs 12 physics steps at 120 Hz for each decision."), p("6. The score: fast, straight, smooth, stable", "Heading1"), p("Positive reward comes from progress, centreline tracking, correct heading, acceleration, rotation, climb, altitude, stability and success. Penalties cover time, lateral error, excessive attitude, rough controls, premature rotation and failures. The clean profile adds passenger-comfort terms for tight centreline tracking, low sideways acceleration, low yaw/roll rate and a clean-takeoff bonus."), p("FAST ENOUGH - STRAIGHT ENOUGH - SMOOTH ENOUGH - STABLE ENOUGH", "Memory"), p("7. How the pilot learns", "Heading1"), p("The main algorithm is Soft Actor-Critic (SAC). Its actor and critics use three 256-unit ReLU layers. It learns from a replay buffer of one million transitions. Teacher demonstrations enter the buffer first, and behaviour cloning gives the actor a sensible starting point. SAC then improves through its own exploration."), p("Curriculum advancement depends on deterministic evaluation, not noisy training reward. The policy advances when evaluation is above 80 percent with at least 20 episodes. Each new level makes the exam harder."), fig("evaluation_progress.png", "Evaluation progress. Drops after level-ups are expected because the conditions become harder.", 75), PageBreak()]

status_rows = [["Snapshot item", "Value"], ["Active run", status.get("run_name", "")], ["RL transitions", f"{latest_steps:,}"], ["Curriculum level", str(latest_level)], ["Training episodes", f"{episodes:,}"], ["Training successes", f"{successes:,}"], ["Rolling success (latest 100)", f"{rolling*100:.0f}%"], ["Latest deterministic evaluation", f"{float(latest_eval.get('success_rate',0))*100:.0f}% at level {int(latest_eval.get('level', latest_level))}"], ["First success", f"{int(milestones['steps_to_first_success']):,} transitions"], ["Level 1 -> 2", "225,000 transitions"], ["Level 2 -> 3", "1,275,000 transitions"]]
story += [p("8. Current stage", "Heading1"), p("Training was active when this report was built, so these numbers are a dated snapshot. The system has learned convincing behaviour through level 2 and is now adapting to the harder level-3 conditions. It is not yet a final, fully robust policy."), table(status_rows, [67*mm, 99*mm]), Spacer(1, 5*mm), fig("recent_outcomes.png", "Latest 100 training outcomes. Unclean takeoff means the target region was reached, but the complete trajectory was not smooth enough.", 78), p("9. What good and bad flights look like", "Heading1"), fig("teacher_success_dashboard.png", "SUCCESS: the dashboard shows a centered runway track, a smooth rotation and a stable climb. A beginner should check that the centerline error stays small, the roll stays controlled and the climb continues.", 88), PageBreak(), fig("training_success_3d.png", "SUCCESS IN 3D: a recorded training success viewed from behind. The aircraft begins on the runway centerline, and the green event list confirms repeated successful episodes.", 88), fig("successful_runway_start_side.png", "SUCCESS CHECKPOINT: this side view lets us check correct wheel contact, runway alignment and aircraft attitude at the start of the replay.", 82), PageBreak(), fig("early_agent_failure.png", "FAILURE: an early learned policy wanders away from the centerline, moves the controls roughly and eventually leaves the runway. This tells the next developer to inspect centerline error, heading error and control smoothness.", 88), p("How to read these pictures", "Heading2"), p("Use three quick questions: (1) Is the aircraft close to the runway centerline? (2) Are heading and roll small and controlled? (3) After rotation, does altitude rise steadily without wild control movement? A success answers yes to all three. A failure picture is useful evidence: it points to the observation, reward or controller behaviour that needs investigation.")]

layout_rows = [["Path", "Responsibility"], ["configs/", "Default aircraft, environment, reward, randomisation and learning settings."], ["configs_975/", "Special 975,000-lb experiment configuration."], ["envs/", "Gymnasium environment, JSBSim bridge, observations, actions, reward and termination."], ["training/", "SAC/PPO entry points, callbacks, checkpoints and curriculum."], ["teacher/", "Teacher controller and demonstrations."], ["evaluation/", "Evaluation, robustness, failure analysis and trajectory recording."], ["tools/", "Checks, benchmarks, plots and 2D/3D viewers."], ["tests/", "Unit, integration and showcase tests."], ["models/", "Policy ZIPs, replay-buffer PKLs and state JSON files."], ["results/", "Metrics, milestones, CSVs, demonstrations and replay events."], ["logs/", "TensorBoard and console logs."], ["msfs_sac_runtime.py", "Experimental SimConnect bridge to Microsoft Flight Simulator."]]
story += [p("10. Project map", "Heading1"), table(layout_rows, [45*mm, 121*mm]), PageBreak(), p("11. Safe reproduction", "Heading1"), Preformatted(r'''cd C:\Users\asus\Documents\project\boeing7478_rl_takeoff
.venv\Scripts\Activate.ps1
python tools\check_environment.py
python -m pytest
tensorboard --logdir logs
python tools\live_viewer.py --open''', styles["CodeSmall"]), p("To resume, keep the same reward profile and use the matching model, replay buffer and state file:"), Preformatted(r'''python -m training.train_sac --config configs/sac.yaml ^
  --run-name clean_teacher_lv4 ^
  --resume models/clean_teacher_lv4/checkpoint_1350000.zip ^
  --reward-profile clean --total-timesteps 2000000''', styles["CodeSmall"]), p("Stop or pause the active trainer before starting another process that writes to the same run or generated aircraft directory."), p("12. Validation status", "Heading1"), p("The project environment checker passed all 10 checks: aircraft loading, shapes, action timing, bounded observations, rate-limited actions, deterministic seeds, reward arithmetic, terminations, random-action stability and Gymnasium validation."), p("The full pytest run executed 79 tests: 72 passed and 7 showcase tests could not complete because the active trainer held the generated aircraft XML while Windows also denied pytest's temporary directory. These are environment/locking errors rather than seven confirmed behaviour failures. Rerun pytest after stopping training."), p("13. Honest limitations", "Heading1"), p("The aircraft is an approximation. There is no validated 747-8 aerodynamic database, no Boeing takeoff chart comparison, no runway slope, simplified friction, fuel fixed at the CG, and a visual model that does not affect physics. Level-3 evaluation is still low. The MSFS bridge fills several unavailable telemetry values with neutral defaults, so transfer to Microsoft Flight Simulator is experimental and must remain simulation-only."), PageBreak(), p("14. Continuation checklist", "Heading1")]

checklist = ["Let the current run continue on level 3, but judge progress using deterministic evaluation.", "Protect each checkpoint trio: model ZIP, replay-buffer PKL and state JSON.", "After training stops, rerun the 10/10 checker and the full pytest suite.", "Compare the learned policy with teacher and full-throttle baselines on identical seeds.", "Investigate the dominant level-3 outcomes: unclean takeoff and failed acceleration.", "Run 1000+ robustness episodes only after performance stabilises across harder levels.", "Treat level 5 as a special slippery-runway stress test.", "Improve the MSFS telemetry mapping before testing transfer.", "Refresh this report after the next major checkpoint or curriculum level-up."]
for i, item in enumerate(checklist, 1):
    story.append(p(f"<b>{i}.</b> {item}"))
story += [p("SEE -> STEER -> SCORE -> SAVE", "Memory"), p("27 observations -> 4 controls -> reward and outcome -> checkpoint and replay", "Callout"), p("15. Final handover", "Heading1"), p("The project is currently a capable research simulator with a teacher-guided SAC policy that has graduated to curriculum level 3. A new contributor can continue safely by preserving experiment state, evaluating on fixed seeds, rerunning tests after the trainer stops, and improving robustness before making any claim beyond simulation."), p("Primary project sources", "Heading1"), p("README.md; AIRCRAFT_MODEL_NOTES.md; EXPERIMENT_PROTOCOL.md; YAML files under configs/ and configs_975/; Python modules under envs/, training/, teacher/, evaluation/ and tools/; and the current status, milestone and CSV files under results/clean_teacher_lv4/.")]

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c5d3dd"))
    canvas.line(19*mm, 17*mm, 191*mm, 17*mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#496678"))
    canvas.drawString(19*mm, 12*mm, "Boeing 747-8 RL Takeoff - friendly project handover")
    canvas.drawRightString(191*mm, 12*mm, f"{doc.page}")
    canvas.restoreState()

pdf_path = OUT / "Boeing7478_RL_Project_Guide.pdf"
doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=19*mm, rightMargin=19*mm, topMargin=18*mm, bottomMargin=22*mm, title="Boeing 747-8 RL Takeoff - Friendly Project Guide", author="Project handover")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(pdf_path)
