# Canonical data contracts

The machine-readable registry is [`schemas/contracts.json`](../schemas/contracts.json).
CSV headers are versioned and validated before a scenario enters an experiment.

## Required tables

- `zones.geojson`: stable zone geometry and walkability.
- `congestion.csv`: count, continuous density, display level, provenance, and
  observation freshness per `(timestamp, zone_id)`.
- `observations.csv`: sensor pose, coverage, visible zones, and confidence.
- `robots.csv`: timestamped robot pose and goal.
- `paths.csv`: ordered baseline or congestion-aware path segments and costs.
- `replanning_events.csv`: explicit causal link between new information and a
  changed path.

## Observation state

The canonical states are:

| State | Meaning |
|---|---|
| `FIXED_COVERED` | A fixed CCTV currently supplies the value. |
| `ROBOT_RECOVERED` | A robot recently recovered a fixed-camera blind spot. |
| `STALE` | A value exists, but its observation age exceeds the scenario limit. |
| `UNKNOWN` | No valid observation exists. |

Congestion level is one of `LOW`, `MEDIUM`, `HIGH`, or `UNKNOWN`. Thresholds
belong to the scenario configuration so analysts can change them without
discarding continuous density.

## Step 7 independent evaluation contracts

JRDB `frame_metrics.csv` records robot-view observation yield and occlusion;
`track_continuity.csv` records temporal persistence. Mall `frame_metrics.csv`
records paired image-level count and congestion-class errors. These tables are
dataset-scoped evaluation products, not canonical ATC scene state, and must not
be joined spatially across venues.
