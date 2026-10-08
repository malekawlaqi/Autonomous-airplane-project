# Autonomous Airplane Project

This repository contains the autonomous-aircraft simulation work in this project folder, including the Boeing 747-8 reinforcement-learning takeoff environment, training code, evaluation tools, reports, and retained visual results.

## Main project

Start with [`boeing7478_rl_takeoff/README.md`](boeing7478_rl_takeoff/README.md). The beginner-friendly handover report is available under [`boeing7478_rl_takeoff/report/output/`](boeing7478_rl_takeoff/report/output/).

For readers new to reinforcement learning, use the illustrated [from-zero handbook](boeing7478_rl_takeoff/report/deep_guide/output/Autonomous_Airplane_From_Zero_Handbook.pdf). Its [LaTeX source](boeing7478_rl_takeoff/report/deep_guide/Autonomous_Airplane_From_Zero_Handbook.tex) and [builder](boeing7478_rl_takeoff/report/deep_guide/build_deep_guide.py) are included.

## Large generated files

Python virtual environments, logs, temporary files, TensorBoard events, and replay-buffer checkpoints are intentionally excluded from Git. They are machine-specific or generated during training and can exceed GitHub's file-size limits. Follow the project setup instructions to recreate the Python environment and resume training with locally retained checkpoints.

## Safety and scope

This is a simulation and research project. The aircraft model is an approximation and is not suitable for real-world flight operations or certification.
