# Web development fixtures

These small files let a frontend be developed without committing the complete
local experiment directory.

See [`docs/web-development.md`](../../docs/web-development.md) for coordinate,
timing, access-boundary, linked-view, and export guidance. Additional trajectory
and map visuals are indexed in [`docs/media/README.md`](../../docs/media/README.md).

For each `stage3/<scenario>/` directory:

- `costmap.png` is a representative rendered congestion/intention/freshness
  visualization.
- `intention_events.jsonl` is a timestamped stream of predicted destination-grid
  changes. Each line is an independent JSON object.

For each `stage4/<scenario>/` directory:

- `occupancy_2d.json` defines map axes, world frame, origin, resolution, and
  dimensions.
- `occupancy_2d.png` is a lightweight preview of the static occupancy map.
- `summary.json` provides generation metadata and occupancy counts.

The three scenario identifiers are `office`, `hospital`, and `warehouse`.
Coordinates use `isaac_world_enu`; array shape is expressed as rows then
columns. Preserve Congestion, Intention, and Freshness as separate UI fields.
Do not expose privileged Ground Truth through policy-facing views or APIs.

Large runtime streams should be generated locally with the commands in the
stage reports. Do not design a production API around absolute paths found in
local run manifests.
