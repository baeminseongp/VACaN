# VACaN reproduction guide

This guide describes how to reconstruct the implemented VACaN pipeline from a
fresh clone. It does not claim that unfinished research components are complete.
Use the per-stage reports as the record of actual acceptance results.

## 1. Reproduction boundary

The Git repository contains VACaN source, YAML configuration, tests, validated
pedestrian routes, documentation, and small visual fixtures. It does not contain
Isaac Sim, Isaac Lab, Pegasus, official NVIDIA USD assets, full experiment logs,
or trained policy checkpoints.

The mandatory runtime combination used by this snapshot is:

| Component | Version or implementation |
|---|---|
| Isaac Sim | 5.1.0 standalone |
| Isaac Lab | 2.3.2 |
| Pegasus Simulator | 5.1.0, verified commit `644da37e` |
| Python | Isaac Sim bundled 3.11 for simulator processes |
| Robot | NVIDIA Nova Carter |
| Environments | Official Office, Hospital, and Warehouse assets |
| Pedestrians | Pegasus People API and `PersonController` |
| Policy | `sb3-contrib` Recurrent PPO, one environment at a time |
| Control | Differential-drive linear and angular velocity |

Do not substitute another simulator, robot, or pedestrian framework. If the
pinned combination cannot run, record the incompatibility before changing it.

## 2. Hardware and external software

Use an RTX GPU that satisfies the Isaac Sim 5.1 requirements. The recorded host
used Ubuntu 24.04, driver 580.178.04, two RTX 3090 GPUs, and a separate T1000.
Those exact devices are evidence from one machine, not universal requirements.

Install Isaac Sim 5.1.0 and clone Isaac Lab 2.3.2. Place Pegasus 5.1.0 under the
Isaac Lab `external/PegasusSimulator` directory or pass its actual location to
the launchers. A typical layout is:

```text
/path/to/IsaacLab/
  _isaac_sim -> /path/to/isaac-sim-5.1.0
  source/isaaclab/
  external/PegasusSimulator/
```

Install Isaac Lab and Pegasus into Isaac Sim's bundled interpreter:

```bash
cd /path/to/IsaacLab
./_isaac_sim/python.sh -m pip install --no-build-isolation -e source/isaaclab
./_isaac_sim/python.sh -m pip install --no-build-isolation -e \
  external/PegasusSimulator/extensions/pegasus.simulator
./_isaac_sim/python.sh -m pip install h5py==3.14.0
```

Do not install the simulator into VACaN's `.venv`. Simulator processes use
Isaac Sim's Python; host checks, orchestration, tests, and PPO packages use the
project environment.

## 3. Project environment

From the repository root:

```bash
uv venv --python python3.11 .venv
uv pip install --python .venv/bin/python -r requirements/stage0.txt
uv pip install --python .venv/bin/python -e '.[dev,rl,simulation]'
.venv/bin/pytest -q
```

The prepared release passed all 94 host-side tests. GPU and simulator acceptance
still require a real Isaac Sim host.

Most launchers default to `/home/engineer/IsaacLab`. On another machine, use
their explicit path flags:

```bash
.venv/bin/python scripts/run_stage1_scenarios.py --help
.venv/bin/python scripts/run_stage2_scenarios.py --help
.venv/bin/python scripts/run_stage4_mapping.py --help
.venv/bin/python scripts/run_stage5.py --help
```

The Stage 0 shell runner accepts `ISAACLAB_ROOT` and `ISAACSIM_ROOT` environment
variables. Do not copy the recorded GPU index blindly; select a valid RTX device
for the new host.

## 4. Stage 0: runtime and API smoke test

Inspect the host and save the result:

```bash
ISAACLAB_ROOT=/path/to/IsaacLab \
ISAACSIM_ROOT=/path/to/IsaacLab/_isaac_sim \
.venv/bin/python scripts/check_stage0_host.py \
  --output artifacts/stage0/host-inspection.json
```

Run two independent simulator integrations:

```bash
ISAACLAB_ROOT=/path/to/IsaacLab \
ISAACSIM_ROOT=/path/to/IsaacLab/_isaac_sim \
VACAN_GPU_INDEX=0 VACAN_HEADLESS=1 \
bash scripts/run_stage0_smoke.sh
```

Acceptance requires both runs to load an official environment, spawn Nova
Carter and a Pegasus person, observe pedestrian movement, read transforms, save
passing JSON, and shut down cleanly. See `docs/stage-0-report.md` for the exact
recorded result and known display/GPU constraints.

## 5. Stage 1: official scenes and pedestrians

Generate one deterministic training episode in each environment and replay its
resolved routes:

```bash
.venv/bin/python scripts/run_stage1_scenarios.py \
  --isaacsim-root /path/to/IsaacLab/_isaac_sim \
  --pegasus-root /path/to/IsaacLab/external/PegasusSimulator \
  --scenario all --split train --episodes 1 \
  --output artifacts/stage1-isaac --headless \
  --verify-reproducibility
.venv/bin/python scripts/validate_stage1.py artifacts/stage1-isaac
```

Each accepted episode contains `ground_truth.jsonl`, `resolved_routes.json`, a
resolved scenario, a summary, a simulator log, and `top_down.png`. Ground Truth
is a privileged recording and must not enter policy observations. The release
tracks only the three resolved route files required by later stages.

## 6. Stage 2: Nova Carter navigation interface

Run and validate the differential-drive baseline:

```bash
.venv/bin/python scripts/run_stage2_scenarios.py \
  --isaacsim-root /path/to/IsaacLab/_isaac_sim \
  --pegasus-root /path/to/IsaacLab/external/PegasusSimulator \
  --scenario all --split train --episodes 1 \
  --output artifacts/stage2-isaac --headless
.venv/bin/python scripts/validate_stage2.py artifacts/stage2-isaac
```

The action is `(linear_x_m_s, angular_z_rad_s)` and is converted to Nova Carter
wheel velocities using the configured wheel radius and wheel base. Collision,
goal, timeout, action limiting, and clean process isolation are validated. The
policy-facing observation contains robot/goal state; pedestrian state remains
under a privileged log boundary.

## 7. Stage 3: dynamic costmap

Stage 3 replays synchronized Stage 2 state and updates at 5 Hz:

```bash
.venv/bin/python scripts/run_stage3_costmaps.py \
  --scenario all --output artifacts/stage3
.venv/bin/python scripts/validate_stage3.py artifacts/stage3
```

The policy record contains cost, static-obstacle, walkable, and CCTV-coverage
layers. Diagnostics separately retain CCTV counts, robot visibility, FOV
overlap, freshness, update type, intention events, and before/after costs.
Privileged Ground Truth counts are written separately.

The controlled sensing assumptions are important:

- CCTV cells receive exact current density and never decay.
- Visible non-CCTV cells are updated by noisy robot observations.
- Non-CCTV freshness decays linearly over 300 seconds.
- The implemented Stage 3 intention estimator is constant velocity with
  transition confirmation. It is not yet the final GRU with interaction
  attention required by the research plan.
- Geometric occlusion is configurable and was disabled in the recorded Stage 3
  configurations; do not describe the current observation model as full vision.

## 8. Stage 4: static maps and RL contracts

Generate static maps directly from official scene collision geometry:

```bash
.venv/bin/python scripts/run_stage4_mapping.py \
  --isaacsim-root /path/to/IsaacLab/_isaac_sim \
  --scenario all --output artifacts/stage4 --gpu 0 --headless
.venv/bin/python scripts/validate_stage4.py artifacts/stage4
```

The policy observation contract is exactly:

```text
map:   float32 [2, 61, 61]  = static_obstacle, dynamic_crowd
robot: float32 [8]          = goal x/y/distance/angle,
                              linear/angular velocity,
                              previous linear/angular action
action: float32 [2] in [-1, 1] -> limited (linear velocity, angular velocity)
```

Freshness stays independently identifiable in diagnostics but is not a direct
Stage 5 policy channel. This design choice must be preserved or explicitly
revisited; silently adding privileged data to the observation is invalid.

## 9. Stage 5: Recurrent PPO

Validate all Stage 4 maps before starting training:

```bash
.venv/bin/python scripts/validate_stage4.py artifacts/stage4
.venv/bin/python scripts/run_stage5.py \
  --isaacsim-root /path/to/IsaacLab/_isaac_sim \
  --isaaclab-root /path/to/IsaacLab \
  --mode smoke --scenario office --phase dynamic_crowd \
  --seed 5101 --gpu 0 --headless
```

The policy uses a CNN for the two map channels, an MLP for the eight robot
values, and a 128-unit LSTM in `sb3-contrib` Recurrent PPO. Office, Hospital,
and Warehouse checkpoints are isolated by scenario. Never load a checkpoint
under the wrong environment identity.

For checkpoint evaluation, pass `--mode evaluate --checkpoint <path>`. For
continued training, pass the appropriate prior-phase checkpoint and retain the
same resolved configuration. The full curriculum, recorded commands, successes,
failures, and remaining acceptance work are in `docs/stage-5-report.md`.

## 10. Reproducibility records

For every run, retain:

- the YAML configuration and resolved overrides;
- deterministic seed and train/validation/test split;
- model checkpoint identifier and hash;
- exact command and simulator log;
- package/runtime versions;
- policy-visible JSONL separately from privileged Ground Truth;
- summary, monitor metrics, termination reason, and performance timing.

Never train on validation or test seeds. Freeze prediction models during the
initial DRL training, and freeze the policy during counterfactual experiments.

## 11. Known implementation gaps

The repository implements a working simulator-to-Recurrent-PPO path, the
Office time-expanded reference planner, and the Office frozen-policy
counterfactual workflow. The final research system is still incomplete: the
required Mask-Aware ConvGRU and GRU interaction-attention intention model are
not yet implemented, and the redesigned Case 1-2/Case 2 physical failure run
has not yet been accepted. Office-only Stage 7 code must not be presented as a
validated Hospital or Warehouse result.

Stage 4 and Stage 5 reports remain marked in progress. Do not claim stage
completion until their own acceptance tests and required experiments pass.

## 12. Common failure modes

- Wrong interpreter: simulator code must start through Isaac Sim's `python.sh`.
- Wrong Gymnasium/Torch import order: use `scripts/run_stage5.py`; it preserves
  Isaac Sim's CUDA Torch while selecting VACaN's compatible Gymnasium/SB3.
- Missing route files: regenerate Stage 1 or use the three tracked validated
  `resolved_routes.json` files at their configured paths.
- Missing Stage 2 recordings: Stage 3 cannot run until Stage 2 trajectory JSONL
  exists locally.
- Missing Stage 4 NPZ: Stage 5 requires a generated and validated static map.
- GPU/display mismatch: use headless mode first and record the selected device.
- Apparent policy success from a smoke checkpoint: smoke tests prove execution,
  not learned navigation or robustness.
