# Web development guide

The web layer should help analysts localize inefficient navigation decisions in
space and time, then compare controlled counterfactual interventions. It must
preserve the distinction between what the robot observed and what the simulator
knew.

## 1. Available fixtures

`examples/web-data/` contains small fixtures for Office, Hospital, and
Warehouse. They are sufficient for layout, parsing, map transforms, legends,
and timeline prototypes without committing complete experiment runs.

| File | Use in a frontend |
|---|---|
| `stage3/<scenario>/costmap.png` | Static visual reference for factor-aware map styling. |
| `stage3/<scenario>/intention_events.jsonl` | Timeline events with simulation time, pedestrian ID, and destination grid. |
| `stage4/<scenario>/occupancy_2d.json` | World frame, origin, resolution, axes, and grid shape. |
| `stage4/<scenario>/occupancy_2d.png` | Static occupancy preview. |
| `stage4/<scenario>/summary.json` | Scene/map provenance and aggregate occupancy counts. |

Additional visual evidence is indexed in `docs/media/README.md`.

## 2. Coordinate and timing conventions

- World frame: `isaac_world_enu`.
- Grid arrays: `row_y,column_x`.
- Shapes are recorded as rows then columns.
- Stage 3 and policy control update at 5 Hz; physics runs at 60 Hz.
- Use `simulation_time_s` for synchronization. Wall-clock timestamps are run
  provenance and should not drive animation.
- Convert grid to world with the declared origin and resolution; never infer
  geometry from PNG dimensions.

## 3. Required visual distinctions

The UI and any API feeding it should expose these as separate concepts:

| Category | Examples | Visibility |
|---|---|---|
| Policy-visible state | dynamic costmap, obstacle map, robot vector, action | May appear in policy analysis views. |
| Observation diagnostics | freshness, FOV mask, source/update type, intention events | May explain why the policy state changed. |
| Privileged Ground Truth | exact global pedestrian poses/counts | Evaluation/reference views only; label clearly. |
| Outcome/evaluation | reward terms, collision, goal, timeout, path length | Post-hoc analysis. |

Never merge Ground Truth and observed density into one unlabeled layer. CCTV
cells are an explicit exception: their exact current density is policy-visible
under the controlled research assumption, and they have no freshness decay.

## 4. Recommended linked views

1. Spatial view: occupancy, route, robot pose, CCTV coverage, FOV, observed
   congestion, predicted intention, and optional privileged overlay.
2. Temporal view: synchronized reward, distance-to-goal, action, freshness,
   congestion, intention-event, and termination tracks.
3. Factor inspector: independent Congestion, Intention, and Freshness values at
   the selected time/cell, including observation source and age.
4. Failure localization: ranked intervals/cells associated with inefficient
   path decisions, with evidence links back to raw policy-visible records.
5. Counterfactual comparison: observed rollout beside a frozen-policy
   intervention, using the same cost evaluator and clearly showing changed
   factors.

The final two views describe the research objective; the repository does not
yet provide their complete computation pipeline.

## 5. Runtime data contract

`DynamicCostmap.tick()` returns three top-level objects:

```text
policy:
  simulation_time_s, costmap, static_obstacle_mask,
  walkable_mask, cctv_coverage_mask
diagnostics:
  cctv_counts, observations, intention_events, robot_visibility_mask,
  fov_overlap_ratio, freshness, intention_incoming_count,
  last_update_type, cost_before, cost_after, timestamps
privileged:
  ground_truth_count
```

Stage 5 evaluation adds robot state, goal, applied action, per-term reward,
nearest-person distance, path length, termination reason, and costmap
diagnostics. Keep privileged pedestrian state in a separate response or file.

For production web use, add a versioned exporter that transforms JSONL/NPZ into
chunked, browser-safe JSON or binary arrays. Do not make the browser load full
simulator logs or multi-million-line evaluation files. An export manifest should
include schema version, scenario, seed, split, checkpoint hash, coordinate
frame, time range, layer definitions, and source-artifact hashes.

## 6. Suggested frontend directory

No current frontend framework is mandated. A future client can live under
`apps/web/` while generated files remain ignored:

```text
apps/web/
  src/                 application code
  public/              static shell assets
  tests/               parser and interaction tests
  source-data/         generated local exports; never commit full runs
  dist/                build output; ignored
```

The retired ATC/CAVIAR dashboard was intentionally not restored because its
schema describes a different prototype. Reuse visual ideas only after mapping
them to the current Isaac Sim data contract.

## 7. Minimum web acceptance checks

- Parse all three fixture scenarios without hard-coded grid sizes.
- Verify world/grid conversion against origin, resolution, and axis order.
- Synchronize events by simulation time.
- Toggle policy, diagnostics, and Ground Truth layers independently.
- Display observation source and freshness without collapsing them into cost.
- Label every privileged overlay and prevent it from entering policy inputs.
- Handle missing frames and partial runs without inventing values.
- Record scenario, seed, split, and checkpoint in exported screenshots/results.
