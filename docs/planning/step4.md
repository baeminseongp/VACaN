# Step 4: congestion-aware zone-graph planning

## Reproducible command

```bash
make atc-planning
```

Configuration is versioned in `configs/planning/atc-step4.json`. Generated
artifacts are written to `data/processed/atc/atc-20121114/planning/`.

## Experiment design

The 70 Step 3 zones form a connected four-neighbour graph with 108 undirected
edges. Three fixed start/goal pairs are planned every five minutes over 132
timestamps.

- `BASELINE` minimizes metric edge distance.
- `CONGESTION_AWARE` applies observed density plus state penalties for
  robot-recovered, stale, and unknown information.
- Both routes are evaluated against the same full ATC ground-truth density.
- Replanning events are emitted only when the congestion-aware path signature
  changes from its preceding five-minute snapshot.

For an edge of length `d`, the planner averages its endpoint penalties. The
congestion term normalizes density by the Step 3 high threshold and caps the
ratio at four. All weights and state penalties remain editable in the config.

`actual_simulated_eta_s` is a controlled proxy, not measured robot travel
time. It assumes 1.2 m/s nominal speed and a configured density slowdown.

## Current result

| Measure | Result |
|---|---:|
| Robot/start-goal pairs | 3 |
| Planning timestamps | 132 |
| Paired route comparisons | 396 |
| Different route geometry | 359 |
| Congestion-aware route longer | 20 |
| Better ground-truth ETA proxy | 301 |
| Lower ground-truth exposure | 313 |
| Mean distance difference | +0.5051 m |
| Mean ground-truth ETA difference | -3.2793 s |
| Mean exposure difference | -0.29135 |
| Replanning events | 316 |

There are many equal-distance alternatives in the grid, so a changed path is
not itself a performance improvement. Paired ETA and exposure deltas are the
relevant outcomes.

## Event provenance

Every event records previous/new path signatures, the zone with the largest
observed change along either path, its source, ETA/distance deltas, and whether
the trigger was classified as congestion or observation-state change. Current
events comprise 189 state changes and 127 congestion changes.

## Limitations

- This is repeated counterfactual planning from fixed endpoints, not continuous
  robot execution.
- Robots do not yet reserve shared edges or avoid one another.
- The grid graph does not model turning radius, acceleration, social distance,
  or local pedestrian collision avoidance.
- ETA is a transparent simulation proxy and must not be described as measured
  travel time.
- The causal trigger is an explainability heuristic based on largest cost
  change, not proof that one sensor observation alone caused the route change.
