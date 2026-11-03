# Stage 2 — Nova Carter Navigation Interface Report

Date: 2026-10-31 (Asia/Seoul)

## Outcome

Stage 2 is complete. The acceptance validator passes for Office, Hospital, and
Warehouse, including physical wheel-driven navigation, synchronized Pegasus
states, reset/stop behavior, goal approach, and controlled Office collision
detection. Isaac Sim and Pegasus were read from the Isaac Lab checkout, but
that checkout was not modified.

## 1. Implemented components

- Three Stage 2 YAML configurations for the existing official Office, Hospital,
  and Warehouse scenes and resolved Stage 1 pedestrian routes.
- Official Nova Carter asset configuration with verified wheel joints
  `joint_wheel_left` and `joint_wheel_right`, 0.14 m wheel radius, and 0.4132 m
  wheel base.
- Isaac Sim `DifferentialController` conversion from
  `a_t = [linear_velocity, angular_velocity]` to wheel velocity targets.
- Deterministic velocity and acceleration limiting.
- A `reset(seed, scenario)`, `step(action)`, observation, reward, termination,
  and timeout interface backed by physics stepping. Robot pose writes occur only
  during reset; navigation steps apply wheel targets to the articulation.
- Robot pose, odometry velocities, wheel positions/velocities, contact force,
  actions, reward terms, and privileged pedestrian state logging in the
  `isaac_world_enu` frame.
- Separation of the policy observation from simulator Ground Truth. The
  observation exposes robot proprioception and relative goal only; pedestrian
  Ground Truth is stored under `privileged_log`.
- A heading-first waypoint baseline for testing only. No DRL model is trained.
- Goal, timeout, controlled contact-force collision, and stop/reset checks.
- One-process-per-episode launcher, top-down trajectory renderer, and acceptance
  validator. All output paths are inside VACaN.

## 2. Created and modified files

- `configs/stage2/office.yaml`
- `configs/stage2/hospital.yaml`
- `configs/stage2/warehouse.yaml`
- `src/vacan/robot/stage2.py`
- `src/vacan/robot/__init__.py`
- `scripts/isaacsim/run_stage2_episode.py`
- `scripts/run_stage2_scenarios.py`
- `scripts/visualize_stage2.py`
- `scripts/validate_stage2.py`
- `tests/test_stage2_navigation.py`
- `docs/stage-2-report.md`
- `README.md`

## 3. Exact execution commands

Host checks already executed:

```bash
.venv/bin/python -m py_compile \
  src/vacan/robot/stage2.py \
  scripts/isaacsim/run_stage2_episode.py \
  scripts/run_stage2_scenarios.py \
  scripts/visualize_stage2.py \
  scripts/validate_stage2.py
.venv/bin/pytest -q
.venv/bin/python scripts/run_stage2_scenarios.py \
  --scenario all --dry-run --headless
```

Final validation command:

```bash
.venv/bin/python scripts/validate_stage2.py artifacts/stage2-isaac
```

The launcher invokes `/home/engineer/IsaacLab/_isaac_sim/python.sh` and reads
the Pegasus extension under `/home/engineer/IsaacLab/external/`. It neither
installs packages there nor writes to those directories. GPU 1 is the default;
pass `--gpu 0` only if that RTX 3090 is intentionally available.

## 4. Tests performed and actual results

- Python compilation: passed for all Stage 2 modules and scripts.
- Host unit suite: `23 passed in 1.17s` after the final Office fix.
- Launcher dry run: passed and emitted three isolated Isaac commands with train
  seeds 4101, 4201, and 4301.
- First Isaac Sim Office attempt: failed before episode stepping with
  `AttributeError: 'NoneType' object has no attribute 'sensor_count'`. The
  installed Isaac Sim log reported that `/World/NovaCarter/chassis_link` did
  not match a rigid contact entry. The cause was contact-reporter preparation
  after the first physics reset. The implementation now prepares the reporter
  before `world.reset()` and initializes the contact view from that reset's
  physics simulation view. Runtime acceptance must be rerun.
- The subsequent visible Office run completed, but inspection showed no useful
  viewer presentation and invalid dynamics: the robot started at z=0, produced
  a 392 kN planar contact reading at step 1, and stopped the baseline after a
  false collision. The configuration now uses the official MobilityGen Nova
  Carter z-offset of 0.25 m, clears residual wheel targets on reset, installs a
  chase-style viewer camera, paces visible runs in real time, and prints episode
  progress in the parent terminal. This correction is still pending rerun.
- The next Office run correctly failed the strengthened runtime checks. Its
  trajectory showed the robot launched upward from the narrow `(0, -6.7)`
  boundary location (z reached 1.125 m), so that pedestrian-walkable point was
  not safe for Nova Carter's larger footprint. The Office robot route now starts
  in the configured central open area, reset clears joint positions as well as
  velocities/drive targets, and every reset performs 30 zero-command settling
  steps before returning an observation. This correction is pending rerun.
- The first visible preview used an exterior camera 7 m above the scene. The
  official Office USD includes exterior city/vegetation set dressing, so this
  showed the background instead of the intended indoor workspace. The visible
  viewport now follows 2.2 m behind Nova Carter at indoor height and updates
  after each physics step. The configured official Office USD is unchanged.
- The stabilized Office run subsequently passed reset, rotation, stop, and
  baseline navigation checks, but its controlled probe was not reported. The
  original contact view monitored only `chassis_link`; Nova Carter's caster or
  wheels can contact an obstacle first. Contact reporting now covers every USD
  prim carrying `UsdPhysics.RigidBodyAPI`, and `runtime-checks.json` preserves
  probe displacement and maximum force even when acceptance fails.
- Final Office seed 4101 acceptance passed using the official environment and
  Nova Carter assets. The robot reached the goal without baseline collision,
  moved 4.658 m along a 4.684 m path, and stopped at 0.00067 m/s. Rotation was
  0.679 rad, reset error was 0 m, and the controlled probe was detected at step
  198 with 998.65 N maximum planar contact force. Runtime was 68.84 s.

The final validator returned `PASS` for all three environments. Office,
Hospital, and Warehouse respectively recorded 722, 640, and 588 synchronized
states with six pedestrians per state; all three reached their configured goal.

## 5. Generated assets and artifacts

All three runs generated the expected artifacts:

```text
artifacts/stage2-isaac/
  manifest.json
  train/{office,hospital,warehouse}/episode-000/
    isaac-sim.log
    scenario.yaml
    summary.json
    runtime-checks.json
    trajectory.jsonl
    top_down.png
```

## 6. Performance measurements

Recorded wall times were 68.84 s for Office, 83.31 s for Hospital, and 44.28 s
for Warehouse.

## 7. Known limitations

- Example artifacts cover one training seed per environment; additional split
  seeds remain supported but were not generated.
- The baseline follows configured waypoints and does not plan around new
  dynamic obstacles. It exists to test the interface, not to replace the later
  recurrent PPO policy.
- Collision is detected from the summed planar contact force over all eight Nova
  Carter rigid bodies with a configured threshold. The controlled Office probe
  validates this path; other environments have no artificial probe.
- Stage 1 pedestrians retain deterministic priority yielding and cached routes;
  Pegasus does not supply SFM or ORCA.

## 8. Requirements for the next stage

Stage 3 may consume only the non-privileged robot observation interface for its
navigation output. Simulator pedestrian Ground Truth remains permitted for CCTV
cells, supervised/reference diagnostics, and separately stored privileged logs.
