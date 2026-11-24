# Physical intelligence

Gopi's learn-by-implementing project for robot learning: sys-id, simulation, RL, IL, VLAs, evaluation and deployment, end to end, on SO-101 arms. The goal is deep practical mastery, including choosing between RL / IL / VLA for a given problem and defending the design decisions.

## Start of every session

1. Read `PROGRESS.md` for the current focus and what was done last.
2. At the end of a session that changed anything, update `PROGRESS.md` (current focus, what was done, next step, open questions).

## How to work with Gopi

- **Default to coaching, not writing.** Gopi implements the algorithms personally to learn them. Explain, review, point out bugs, and suggest the next step; do not write a full algorithm implementation unless asked. Boilerplate, tooling, plotting and repo chores are fine to write directly.
- When reviewing, say what is wrong and why, and leave the fix to Gopi unless asked for it.
- When a design choice comes up (algorithm, sim, action space, data size), give a recommendation with the reasoning and the trade-off, so it can be recorded in `decision_guides/`.
- Be direct about weak results. A falling loss curve is not evidence a policy works.

## Hardware and compute

- Robot: SO-101 arms (Feetech STS3215 servos, position control, no torque sensing, noticeable backlash).
- Machine: laptop with RTX 3050 Ti, 4 GB VRAM. Enough for ACT, small Diffusion Policy, SAC/PPO on state or small images. Not enough to fine-tune pi0 or OpenVLA; SmolVLA is tight. The VLA track needs a cloud GPU.
- `lerobot` is installed from source at `/home/gopi/lerobot`.
- Python 3.10, torch 2.6.0, mujoco 3.6.0, robosuite 1.5.2.

## Folder map

| Folder | What goes there |
|---|---|
| `foundations/` | kinematics, dynamics, classical control, state estimation |
| `hardware/so101/` | calibration, teleoperation, motor characterisation, cameras, bimanual |
| `sys_id/` | actuator modelling, friction/backlash, latency, parameter estimation |
| `sim/` | robot assets, environments, domain randomisation, real2sim, parallel sim |
| `data/` | collection, formats, curation, augmentation |
| `perception/` | visual encoders, camera calibration, world models |
| `RL/` | model-free (value-based, on-policy, off-policy), model-based, offline, real-world, residual, reward design, sim2real |
| `IL/` | BC, DAgger, ACT, Diffusion Policy, flow matching, VQ-BeT, inverse RL |
| `VLA/` | architectures, action tokenisation, fine-tuning, RL fine-tuning, hierarchical planning (reasoning model over a low-level policy), inference, benchmarks |
| `evaluation/` | sim benchmarks, real-world protocols, statistics, failure analysis |
| `deployment/` | control loop, async inference, model optimisation, safety |
| `projects/` | end-to-end SO-101 capstones (teleop → train → evaluate → deploy) |
| `decision_guides/` | write-ups of approach choices, backed by Gopi's own ablations |
| `papers/` | reading notes |
| `common/` | shared code: encoders, buffers, logging, eval utilities |

`RL/policy_based_methods/` holds SAC and TD3; taxonomically they belong in `RL/model_free/off_policy/` and may be moved there.

## Conventions

- Each algorithm folder is self-contained: code, a `config.py` dataclass, `requirements.txt`, and a README stating the environment, how to run it, and the measured result.
- Results are reported as task success rate over N episodes with seeds stated, plus a confidence interval where N allows. Loss curves alone do not count.
- Datasets and checkpoints stay out of git (they are ignored under `results/`); only small plots are committed.
- Shared utilities go in `common/` rather than being copied between folders.

## Git

- One repo, branch `main`, remote `origin` = `github.com/gopisainath1718/physical_intelligence` (public). The local folder is spelled `physical_inteligence`.
- Git does not track empty folders, so most of the scaffold exists only locally until files are added.
- Commit or push only when asked.
