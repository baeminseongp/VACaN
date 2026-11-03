# Stage 3 — Dynamic Costmap Report

Date: 2026-10-31 (Asia/Seoul)

## Outcome

Stage 3 is complete. Host tests and the artifact validator pass for Office,
Hospital, and Warehouse. The implementation uses only Stage 2 robot pose and
Pegasus pedestrian position/velocity state. It renders no camera images, uses
no computer vision, trains no model, and does not implement DRL training.

## 1. Implemented functionality

- Configurable world-aligned grid geometry, world/grid conversion, walkable
  mask, static obstacle mask, and fixed CCTV coverage mask.
- Exact virtual CCTV cell counts and a mean that includes zero-count cells.
  CCTV cells never decay and cannot be overwritten by robot or intention data.
- Deterministic virtual Nova Carter FOV with camera offset, angle, range,
  configurable subcell overlap threshold, overlap-ratio logging, optional grid
  occlusion, and seeded Gaussian position/velocity noise.
- Direct non-CCTV updates replace the cost in every sufficiently observed cell,
  including visible zero-count cells.
- Independent 300-second linear decay toward the current CCTV mean. Anchor time
  and direct-observation time are stored separately.
- Constant-velocity destination prediction with configurable horizon.
- `(pedestrian_id, destination_grid)` event state and confirmed-transition
  hysteresis prevent duplicate/noise-oscillation additions.
- Exact update order: CCTV, mean, decay, robot replacement, intention addition,
  anchor reset, export, and logging.
- Simulation-time 5 Hz scheduling independent of the Stage 2 physics rate.
- Clean policy output containing the costmap and non-privileged masks. Exact
  Ground Truth counts are written to a separate privileged gzip stream.
- Deterministic replay, provenance layers, compressed structured logs, intention
  event logs, and eight-panel costmap visualizations covering every requested
  visual layer.

## 2. Created or modified files

- `configs/stage3/{office,hospital,warehouse}.yaml`
- `src/vacan/mapping/dynamic_costmap.py`
- `src/vacan/mapping/__init__.py`
- `scripts/run_stage3_costmaps.py`
- `scripts/visualize_stage3.py`
- `scripts/validate_stage3.py`
- `tests/test_stage3_costmap.py`
- `docs/stage-3-report.md`
- `docs/stage-2-report.md`
- `README.md`

## 3. Actual execution commands

```bash
.venv/bin/python -m py_compile \
  src/vacan/mapping/dynamic_costmap.py \
  scripts/run_stage3_costmaps.py \
  scripts/visualize_stage3.py \
  scripts/validate_stage3.py \
  tests/test_stage3_costmap.py
.venv/bin/pytest -q
.venv/bin/python scripts/run_stage3_costmaps.py \
  --scenario all --output artifacts/stage3
.venv/bin/python scripts/run_stage3_costmaps.py \
  --scenario all --output /tmp/vacan-stage3-repeat
.venv/bin/python scripts/validate_stage3.py artifacts/stage3 \
  --replay-output /tmp/vacan-stage3-repeat
```

## 4. Validation results

The final host suite returned `41 passed in 1.18s`. Tests cover all 17 requested
categories, including exact counts, zero-cell mean, FOV, deterministic noise,
replacement semantics, update ordering, deduplication, hysteresis, CCTV
protection, coordinate consistency, fixed simulation-time scheduling, and
privileged-state separation.

The required numerical example passed exactly: 5 at t=0, 4 at t=150, new
anchor 6 after two incoming people, 4.5 at t=300, and 3 at t=450. The supplied
`[7, 7, 5, 1]` CCTV values produce mean 5; no mean is hard-coded.

The artifact validator returned `PASS` for all environments and exact equality
for decompressed same-seed replay streams.

| Scenario | Shape | Ticks | Observations | Intention events | CCTV mean range |
|---|---:|---:|---:|---:|---:|
| Office | 6×6 | 61 | 141 | 17 | 0–1.0 |
| Hospital | 4×8 | 54 | 46 | 4 | 0–1.0 |
| Warehouse | 4×5 | 49 | 83 | 10 | 0–1.0 |

## 5. Example costmap outputs

`artifacts/stage3/{scenario}/` contains:

- `policy_costmap.jsonl.gz`: navigation costmap and authorized masks;
- `diagnostics.jsonl.gz`: CCTV, FOV overlap, noisy observations, prediction,
  anchors, freshness, incoming counts, and provenance;
- `privileged_ground_truth.jsonl.gz`: separate exact GT count layer;
- `intention_events.jsonl`: deduplicated event audit trail;
- `costmap.png`, `scenario.yaml`, and `summary.json`.
- `frames/frame-XXXX-tSSS.sss.png`: one eight-layer image per 5 Hz costmap
  tick, plus `frames/frames.json` mapping images to simulation timestamps.
  The manifest stores a SHA-256 fingerprint of the uncompressed costmap,
  diagnostics, and scenario configuration; repeated runs reuse matching frames
  and new renders print progress every ten images.

All policy costmaps use square 4 m by 4 m cells. Retaining each scenario's
configured physical extent produces a 6×6 Office grid, a 4×8 Hospital grid,
and a 4×5 Warehouse grid.

## 6. Measured update frequency and computational cost

The configured rate is 5 Hz, giving a 200 ms budget. Measured replay processing
costs were 1.97 ms/tick for Office, 1.75 ms/tick for Hospital, and 1.50
ms/tick for Warehouse. They remain well below the 200 ms budget, so no frequency
reduction was needed.

Per-tick visualization generated 61 Office, 54 Hospital, and 49 Warehouse PNGs
(164 total, approximately 14 MiB) with timestamp-index manifests.

## 7. Unresolved limitations

- Stage 3 example episodes last about 9.6–12.0 simulation seconds. The complete
  300-second decay is numerically validated by unit tests rather than observed
  end-to-end in these short Isaac recordings.
- Walkable regions and static obstacle regions are explicit YAML masks. The
  outer walkable bounds are coarse, and Office/Hospital internal furniture and
  walls are not yet rasterized from USD/NavMesh. Warehouse includes configured
  shelf regions. These masks must be upgraded before using obstacle occlusion or
  collision planning as authoritative geometry.
- Optional geometric occlusion is enabled only for Warehouse and operates on
  the configured static grid mask, not ray casts against USD geometry.
- The intention model is deliberately constant velocity. The mandatory GRU with
  interaction attention belongs to a later prediction-training stage and is
  not trained here.
- CCTV means are small because the definition averages exact counts over every
  covered cell, including zeros, as required.

## 8. Interfaces required by the next stage

The navigation interface may consume `policy.costmap`, static obstacle mask,
walkable mask, and CCTV coverage mask. It must not consume the privileged GT
stream. Diagnostics preserve congestion, intention, freshness, overlap, and
provenance separately for later failure localization and counterfactual work.

Do not proceed automatically to the next stage. Its acceptance criteria require
explicit user approval.
