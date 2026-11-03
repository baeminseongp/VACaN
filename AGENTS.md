# VACaN — Project Instructions

These instructions apply to the entire repository.

## Research Objective

VACaN (Visual Analytics for DRL Navigation Decision-Making) investigates why a
DRL-based mobile robot selects inefficient navigation paths in dynamic
pedestrian environments.

The analysis focuses on three types of environmental information:

1. **Congestion:** observed or estimated pedestrian density.
2. **Intention:** predicted pedestrian movement and future occupancy.
3. **Freshness:** temporal age of environmental observations.

The final objective is to support visual analysis of inefficient navigation
decisions through **Spatial–Temporal Failure Localization** and
**Counterfactual Intervention**.

Do not redefine this research objective.

## Mandatory Technology Stack

- **Simulation:** NVIDIA Isaac Sim
- **Environments:** official Isaac Sim Office, Hospital, and Warehouse assets
- **Pedestrians:** Pegasus Simulator People API and `PersonController`
- **Mobile robot:** NVIDIA Nova Carter
- **RL framework:** Isaac Lab
- **Navigation policy:** Recurrent PPO
- **Action interface:** differential-drive linear/angular velocity
- **Crowd model:** Mask-Aware ConvGRU
- **Intention model:** GRU with interaction attention
- **Freshness:** observation timestamp-based decay

Do not replace any mandatory component without documenting the technical
incompatibility and obtaining user approval.

## Scientific Constraints

1. Separate simulator Ground Truth from the robot's observed state.
2. Ground Truth is allowed for supervised training labels, reference planning,
   evaluation, and counterfactual experiments.
3. CCTV-covered grid cells provide exact current crowd density at every
   simulation step as a controlled research assumption.
4. CCTV observations have no freshness decay.
5. Robot camera observations update only visible non-CCTV regions.
6. Track freshness for non-CCTV cells.
7. Keep Congestion, Intention, and Freshness independently identifiable in
   observations and logs.
8. Freeze prediction models during initial DRL training.
9. Freeze the DRL policy during counterfactual experiments.
10. Use the same navigation cost evaluator for the observed-state DRL and the
    full-information reference.
11. Distinguish prediction errors from policy limitations.
12. Never claim that a reference planner is globally optimal unless this has
    been established.

## Engineering Constraints

- Inspect existing repository files before editing.
- Inspect installed versions and APIs instead of assuming compatibility.
- Prefer official Isaac Sim, Pegasus, and Isaac Lab APIs.
- Keep simulator initialization and shutdown explicit.
- Use YAML configuration files.
- Save deterministic seeds and model checkpoint identifiers.
- Save actual commands, logs, test results, and exceptions.
- Keep training, validation, and evaluation datasets separate.
- Do not silently fall back to another simulator, robot, or pedestrian
  framework.
- Do not automatically proceed to the next stage.

## Stage Completion Report

After each stage, report:

1. Implemented components
2. Created and modified files
3. Exact execution commands
4. Tests performed and actual results
5. Generated assets and artifacts
6. Performance measurements where relevant
7. Known limitations
8. Requirements for the next stage

Mark a stage complete only when its acceptance tests pass.
