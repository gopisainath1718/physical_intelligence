# Progress

Living state of the project. Read at the start of a session; update at the end.

## Current focus

**Phase 1 (now): the big picture.** Understand why each folder exists and how they connect into one end-to-end pipeline, until the whole picture is crystal clear. No implementation in this phase. Done when Gopi can explain, unprompted, what each folder is for, what it takes as input and hands to the next stage, and what breaks if it is skipped.

**Phase 2 (next): IL methods.** Start implementing with imitation learning, building on the existing BC and DAgger code.

## Next steps

- [ ] Phase 1: walk through the pipeline stage by stage (hardware → sys-id → sim → data → learning → evaluation → deployment), one folder at a time.
- [ ] Phase 1: write the picture down in Gopi's own words (e.g. `decision_guides/big_picture.md`) as the check that it has landed.
- [ ] Phase 2: choose the first IL method and task.
- [ ] Fix the replay buffer in `RL/policy_based_methods/SAC.py`: `size` never increments, and `__len__` calls `len()` on an int.
- [ ] Decide whether to move `RL/policy_based_methods/` into `RL/model_free/off_policy/`.
- [ ] Fill in the empty READMEs (`IL/README.md`, `RL/policy_based_methods/README.md`).
- [ ] Add success-rate evaluation to the existing BC and DAgger code (only loss plots exist today).

## Status by area

| Area | Status |
|---|---|
| `IL/` | BC and DAgger implemented on robosuite Panda Lift, state and camera variants. No success-rate eval yet. ACT, Diffusion Policy, flow matching, VQ-BeT, inverse RL not started. |
| `RL/` | SAC partly written (has the buffer bug above), TD3 is a stub. PPO, model-based, offline, real-world RL not started. |
| `VLA/` | Not started. |
| `hardware/so101/` | Not started. Nothing has run on the real arms yet. |
| `sys_id/`, `sim/`, `data/`, `perception/` | Not started. |
| `evaluation/`, `deployment/` | Not started. |
| `foundations/`, `projects/`, `decision_guides/`, `papers/`, `common/` | Empty. |

## Open questions

- Single arm or bimanual first?
- Which cloud GPU provider for the VLA track?

## Log

- **2026-11-16** — Scaffolded the folder structure, consolidated `IL` and `RL/policy_based_methods` into one repo, pushed to `gopisainath1718/physical_intelligence`. Added `CLAUDE.md` and this file.
