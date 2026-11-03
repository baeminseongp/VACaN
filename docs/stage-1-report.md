# Stage 1 — Dynamic Pedestrian Simulation Report

Date: 2026-10-30 (Asia/Seoul)

## Outcome

Stage 1 is complete. Its acceptance validator passes for Office, Hospital, and
Warehouse using Isaac Sim 5.1 official environment assets and Pegasus 5.1
`Person`/`PersonController`. Stage 2 has not been started.

## 1. Implemented components

- Three YAML scenarios using the official Office, Hospital, and Warehouse with
  Shelves USD assets.
- Six configurable Pegasus people per scenario with explicit spawn points,
  destinations, walking speeds, route cycles, density zones, and split seeds.
- An Isaac NavMesh controller for static-obstacle avoidance and route planning.
- Deterministic ID-priority yielding for nearby pedestrians. Inspection of the
  installed Pegasus source confirmed that `PersonController` is only a callback
  base and does not provide Social Force Model, ORCA, obstacle avoidance, or
  social interaction automatically. The extension does not claim to implement
  SFM or ORCA.
- Deterministic fixed-step pose integration synchronized back to the Pegasus
  animation graph, eliminating nondeterministic animation root-motion drift.
- Per-episode `resolved_routes.json`: the primary run records spawn/destination
  snaps and paths from its baked NavMesh; its same-seed replay uses those exact
  validated route points. This makes the replay bit-for-bit reproducible while
  preserving obstacle-aware route generation in the primary run.
- A Ground Truth writer for timestamp, pedestrian ID, world pose, orientation,
  finite-difference velocity, current destination, robot-independent scene
  metadata, exact per-cell counts/density, congestion zones, and NavMesh
  distance. All coordinates use the `isaac_world_enu` world frame.
- A multi-process dataset generator with train/validation/test seed sets,
  multiple episodes, clean Isaac startup/shutdown per episode, visualization,
  manifests, and quantitative replay comparison.

## 2. Created and modified files

Created or replaced for the configured Stage 1 path:

- `configs/scenarios/office_lobby.yaml`
- `configs/scenarios/hospital_corridor.yaml`
- `configs/scenarios/warehouse.yaml`
- `src/vacan/simulation/stage1.py`
- `scripts/isaacsim/run_pegasus_environment.py`
- `scripts/run_stage1_scenarios.py`
- `scripts/visualize_stage1.py`
- `scripts/validate_stage1.py`
- `tests/test_stage1_simulation.py`
- `docs/stage-1-report.md`
- `README.md`

## 3. Exact execution commands

Final dataset and exact replay validation:

```bash
.venv/bin/python scripts/run_stage1_scenarios.py \
  --scenario all --split train --episodes 1 \
  --output artifacts/stage1-isaac --headless \
  --verify-reproducibility
.venv/bin/python scripts/validate_stage1.py artifacts/stage1-isaac
```

Host tests and syntax checks:

```bash
.venv/bin/python -m py_compile \
  scripts/isaacsim/run_pegasus_environment.py \
  scripts/run_stage1_scenarios.py
.venv/bin/pytest -q
```

The generator invokes `/home/engineer/IsaacLab/_isaac_sim/python.sh` once per
episode, with the selected config, seed, split, episode index, GPU 1, and
`--headless`. Exact child commands are retained in every `summary.json`.

## 4. Tests performed and actual results

Unit tests: `18 passed in 1.16s` before the final Isaac run.

The final validator returned `PASS` for all scenarios:

| Scenario | States | People | Max displacement | Max NavMesh distance | Zone range |
|---|---:|---:|---:|---:|---:|
| Office | 361 | 6 | 11.450 m | 0.089 m | 0–4 |
| Hospital | 421 | 6 | 13.706 m | 0.124 m | 0–1 |
| Warehouse | 421 | 6 | 14.000 m | 0.083 m | 0–2 |

The validator checked all required fields, contiguous/increasing simulation
time, official Isaac 5.1 asset URLs, robot-independent state, exact density
count totals, movement, NavMesh adherence, changing local congestion, all
three scenario names, resolved route artifacts, and top-down images.

Same-seed replay results:

| Scenario | Position samples | Max/RMS error | Exact density frames | Digest |
|---|---:|---:|---:|---|
| Office | 2,166 | 0 / 0 m | 361/361 | identical |
| Hospital | 2,526 | 0 / 0 m | 421/421 | identical |
| Warehouse | 2,526 | 0 / 0 m | 421/421 | identical |

The maximum observed walkable-mesh distance is 0.124 m, below the validator's
0.55 m limit. This is evidence against systematic wall penetration; it is not
a claim that the controller provides a formal collision-proof guarantee.

## 5. Generated assets and artifacts

`artifacts/stage1-isaac/` contains 37 files totaling approximately 51 MiB:

- primary recordings under `train/{office,hospital,warehouse}/episode-000/`;
- replay recordings under `reproducibility/train/...`;
- one copied scenario YAML, JSONL Ground Truth stream, resolved route file,
  summary, full Isaac log, and top-down PNG per run; and
- `manifest.json` with all episode paths and reproducibility measurements.

## 6. Performance measurements

Primary episode wall times recorded by Isaac were:

- Office, 360 steps: 33.85 s;
- Hospital, 420 steps: 70.89 s; and
- Warehouse, 420 steps: 30.31 s.

Hospital is slower because its official USD has substantially more referenced
geometry and materials. Initial uncached composition can take longer than the
final measurements.

## 7. Known limitations

- The official assets resolve through NVIDIA's Isaac 5.1 asset root; a cold
  cache requires network access and can be slow.
- Isaac's NavMesh extension intermittently emits a completed bake event without
  a mesh while large references settle. Startup performs at most three bounded
  bake attempts and fails loudly after that.
- Pegasus 5.1 supplies no social-force/ORCA controller. Priority yielding is a
  deliberately limited interaction policy, not a validated crowd-dynamics
  model.
- Exact replay depends on the primary episode's `resolved_routes.json`. A new
  primary episode rebakes and records a new route artifact; its paired replay
  remains exact.
- Example artifacts cover one training seed per environment. The configs and
  generator support three training seeds and independent validation/test seeds,
  but those additional episodes were not generated in this acceptance run.
- A ROS2 Bridge startup warning is present because ROS2 is not installed; Stage
  1 does not use ROS2.

## 8. Requirements for the next stage

Stage 2 must not begin automatically. Before it starts, retain these scenario
configs, seed/split boundaries, `isaac_world_enu` convention, and privileged
Ground Truth separation. Any observation model must consume only its allowed
robot/CCTV observations, never the Ground Truth stream. The user must explicitly
approve the next stage and its acceptance criteria.
