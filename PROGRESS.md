# Progress

Living state of the project. Read at the start of a session; update at the end.

## Current focus

**Phase 0 (now): the big picture.** Understand why each folder exists and how they connect into one end-to-end pipeline, until the whole picture is crystal clear. No implementation in this phase. Done when Gopi can explain, unprompted, what each folder is for, what it takes as input and hands to the next stage, and what breaks if it is skipped.

**Phase 1: the IL and VLA path.** hardware → data → learn → evaluate → deploy. IL first (building on the existing BC and DAgger code), then VLAs. The VLA part includes the two-level design in `VLA/hierarchical_planning/`: a slow reasoning model plans the steps (e.g. Gemini Robotics-ER through the Gemini API) and a fast low-level policy trained on Gopi's own teleop data (ACT or SmolVLA) executes them.

**Phase 2: the RL path.** hardware → sys-id → sim → learn → sim2real → evaluate → deploy.

**Phase 3: combinations of RL with IL/VLA.** These live in existing folders, filed under the base they build on:

- `RL/Residual_methods/` — RL learns a small correction on top of a frozen IL or VLA policy.
- `RL/real_world_rl/` — RL on the real arm seeded with demonstrations and human interventions (RLPD, HIL-SERL).
- `RL/offline_rl/` — learning from a fixed demo dataset with reward, then offline-to-online fine-tuning.
- `VLA/rl_finetuning/` — improving a VLA with RL after imitation pre-training.
- `IL/inverse_rl/` — recovering a reward from demonstrations, then running RL on it (GAIL, AIRL).

## Next steps

- [ ] Phase 0: walk through the pipeline stage by stage (hardware → sys-id → sim → data → learning → evaluation → deployment), one folder at a time.
- [ ] Phase 0: write the picture down in Gopi's own words (e.g. `decision_guides/big_picture.md`) as the check that it has landed.
- [ ] Phase 1: choose the first IL method and task.
- [ ] Fix the replay buffer in `RL/policy_based_methods/SAC.py`: `size` never increments, and `__len__` calls `len()` on an int.
- [ ] Decide whether to move `RL/policy_based_methods/` into `RL/model_free/off_policy/`.
- [ ] Fill in the empty READMEs (`IL/README.md`, `RL/policy_based_methods/README.md`).
- [ ] Add success-rate evaluation to the existing BC and DAgger code (only loss plots exist today).

## Status by area

| Area | Status |
|---|---|
| `IL/` | BC and DAgger implemented on robosuite Panda Lift, state and camera variants. No success-rate eval yet. ACT, Diffusion Policy, flow matching, VQ-BeT, inverse RL not started. |
| `RL/` | SAC partly written (has the buffer bug above), TD3 is a stub. PPO, model-based, offline, real-world RL not started. |
| `VLA/` | Not started. Planned project: Gemini Robotics-ER as planner over an own low-level policy on SO-101 (`VLA/hierarchical_planning/`); check current model availability first. |
| `hardware/so101/` | Not started. Nothing has run on the real arms yet. |
| `sys_id/`, `sim/`, `data/`, `perception/` | Not started. |
| `evaluation/`, `deployment/` | Not started. |
| `foundations/`, `projects/`, `decision_guides/`, `papers/`, `common/` | Empty. |

## Open questions

- Single arm or bimanual first?
- Which cloud GPU provider for the VLA track?

## Log

- **2026-11-16** — Scaffolded the folder structure, consolidated `IL` and `RL/policy_based_methods` into one repo, pushed to `gopisainath1718/physical_intelligence`. Added `CLAUDE.md` and this file.
- **2026-11-16** — Set the phase order (big picture → IL and VLA → RL → combinations). Added `VLA/hierarchical_planning/` for the planner-plus-policy pattern.
