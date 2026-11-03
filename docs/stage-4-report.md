# Stage 4 Implementation Report — In Progress

Stage 4 is not complete. Host-side mapping, observation, action, reward, and
baseline components pass their tests, but the teleoperated depth maps and
physical Isaac Lab/Pegasus integration episodes have not yet been executed.

## 1. Implemented components

- `VoxelMap`: uint8 Free/Occupied/Unknown 3D array, `zyx` storage order,
  `isaac_world_enu` origin, resolution, bounds, source asset, collision metadata,
  deterministic compressed NPZ and JSON export.
- Automatic Isaac Sim mapping using the installed official
  `isaacsim.asset.gen.omap.Generator`. Isaac Sim 5.1 reports 3D dimensions from
  `generate3d()` but exposes only a 2D-sized `get_buffer()`. The implementation
  therefore calls the official `generate2d()` at successive 0.2 m Z planes and
  stacks the validated dense slices into the 3D map. It retains the navigation-
  height native slice for audit and projects the stack onto a dedicated 0.2 m
  static occupancy grid with an explicit robot-height interval.
- Height-filtered 3D-to-2D projection reusing the exact Stage 3
  `GridGeometry`. Raw occupancy, unknown mask, inflated obstacle map, and
  conservative policy obstacle map remain separate.
- Euclidean configuration-space inflation with a 0.38 m circular Nova Carter
  radius plus a 0.12 m safety margin.
- Robot-centered observation builder with independently stacked static obstacle
  and dynamic crowd channels. Outside-map and unknown cells are unsafe in the
  obstacle channel. Freshness and privileged Ground Truth are not inputs.
- Normalized action adapter for `[linear_velocity, angular_velocity]`, reusing
  Stage 2 velocity and acceleration limiting.
- Independently logged navigation reward terms and an A* engineering baseline.
- Independent 3D voxel and 2D occupancy visualization scripts.
- A pedestrian-free, visible Nova Carter teleop mapper. It fuses the virtual
  camera's `distance_to_image_plane` frames by marking ray interiors Free,
  measured endpoints Occupied, and unobserved voxels Unknown. It also exports
  the accumulated point cloud and every raw depth frame.

The real-time teleop defaults use a 160×90 depth render at 2 Hz, a 20 Hz GUI
viewport, endpoint-voxel deduplication, and one free-space sample per 0.2 m.
Raw depth-frame compression is disabled by default to avoid blocking control;
it can be enabled with `save_depth_frames: true` in the scenario YAML. The
accumulated 3D voxel depth map and point cloud are always saved.

## 2. Mapping architecture and APIs

Official environment USD assets are spawned with Isaac Lab `UsdFileCfg` into an
isolated Isaac Sim process. The installed NVIDIA occupancy-map generator reads
the live PhysX scene and handles composed/instanced collision geometry without
manual USD prim traversal. No robot or pedestrians are spawned. The 3D voxel
resolution is 0.2 m, and generated maps are cached as NPZ/JSON artifacts.

The installed Isaac Lab version is 2.3.2. Its current `DirectRLEnv` and
`DirectRLEnvCfg` lifecycle and hook methods were inspected from the installed
source. Physical DRL environment integration remains pending until the collision
maps below have been generated and inspected.

## 3. 2D map and observation specification

Per explicit user direction, the static obstacle grid is independent of the
coarse Stage 3 crowd grid. This supersedes the original Stage 4 B2 same-grid
requirement. Static occupancy uses 0.2 m cells so walls and furniture do not
invalidate an entire 4 m crowd cell.

| Scenario | Static obstacle shape (0.2 m) | Crowd shape (4 m) | World extent |
|---|---:|---:|---:|
| Office | 120×120 | 6×6 | 24×24 m |
| Hospital | 80×160 | 4×8 | 16×32 m |
| Warehouse | 80×100 | 4×5 | 16×20 m |

The policy observation is a dictionary suitable for an initial recurrent
multi-input policy:

- `map`: float32 `[2, 61, 61]`, channel order `static_obstacle`,
  `dynamic_crowd`, values normalized to `[0, 1]`;
- `robot`: float32 `[8]`: normalized goal-relative x/y, goal distance, goal
  angle, linear/angular velocity, and previous linear/angular action.

The 61×61 crop covers 12.2×12.2 m at the static 0.2 m resolution, is robot-
centered, and rotates with robot heading. The 4 m crowd layer is nearest-neighbor
resampled into this crop while remaining independently identifiable. Outside-map
padding is obstacle value 1, never known-free value 0.

## 4. Files

- `configs/stage4/{office,hospital,warehouse}.yaml`
- `src/vacan/mapping/static_map.py`
- `src/vacan/rl/config.py`
- `src/vacan/rl/stage4.py`
- `scripts/isaacsim/generate_stage4_static_map.py`
- `scripts/run_stage4_mapping.py`
- `scripts/visualize_stage4_maps.py`
- `scripts/validate_stage4.py`
- `tests/test_stage4_mapping.py`

## 5. Commands and actual results

Executed without starting Isaac Sim:

```bash
.venv/bin/pytest -q
# 52 passed in 1.25s

.venv/bin/python scripts/run_stage4_mapping.py \
  --scenario all --output artifacts/stage4 --headless --dry-run
# emitted three isolated Isaac Sim commands successfully
```

The first manual collision-prim traversal exposed that the composed Office asset
does not enumerate `UsdPhysics.CollisionAPI` prims below `/World/Environment`.
It was replaced with NVIDIA's official occupancy generator. The automatic,
headless, no-people command is:

```bash
.venv/bin/python scripts/run_stage4_mapping.py \
  --scenario all --output artifacts/stage4 --gpu 1 --headless

.venv/bin/python scripts/validate_stage4.py artifacts/stage4
```

The depth teleop mapper remains available as an optional observed-map workflow:

```bash
.venv/bin/python scripts/run_stage4_teleop_mapping.py \
  --scenario office --output artifacts/stage4-teleop --gpu 1

# Drive with W/S and A/D; Space stops; P saves and exits.

.venv/bin/python scripts/validate_stage4_teleop.py \
  --scenario office --output artifacts/stage4-teleop
```

## 6. Pending acceptance work

- Run and inspect official-generator 3D/2D maps for all three official assets.
- Integrate the verified maps, Stage 3 live costmap, Nova Carter wheel targets,
  and Pegasus controllers into the installed Isaac Lab 2.3.2 environment.
- Run random and goal-directed physical episodes in all three environments.
- Record trajectory, collision, reward, synchronization, reset, and timing
  evidence and generate navigation/reward visualizations.

No DRL training has been performed. Stage 4 is not marked complete and Stage 5
has not been started.
