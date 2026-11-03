#!/usr/bin/env python3
"""Generate cached 3D/2D maps with Isaac Sim's official occupancy generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--headless", action="store_true")
ARGS = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": bool(ARGS.headless), "active_gpu": int(ARGS.gpu),
    "physics_gpu": int(ARGS.gpu), "multi_gpu": False, "max_gpu_count": 1,
    "sync_loads": True, "fast_shutdown": True,
})

import numpy as np
import omni.client
import omni.kit.app
import omni.physx
import omni.usd

import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path
from vacan.mapping import VoxelMap, project_voxel_map
from vacan.rl import load_stage4_config


def wait_for_loading(timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        if time.monotonic() >= deadline:
            raise TimeoutError("USD composition exceeded timeout")
        simulation_app.update()


def run() -> dict:
    started = time.perf_counter()
    config = load_stage4_config(ARGS.config.resolve())
    stage1_path = (ARGS.config.resolve().parent / config["stage1_config"]).resolve()
    stage2_path = (ARGS.config.resolve().parent / config["stage2_config"]).resolve()
    stage3_path = Path(config["resolved_stage3_config"])
    stage1 = yaml.safe_load(stage1_path.read_text(encoding="utf-8"))
    stage2 = yaml.safe_load(stage2_path.read_text(encoding="utf-8"))
    input_hash = hashlib.sha256()
    repository_root = Path(__file__).resolve().parents[2]
    for path in (ARGS.config.resolve(), stage1_path, stage2_path, stage3_path,
                 Path(__file__).resolve(), repository_root / "src/vacan/mapping/static_map.py"):
        input_hash.update(path.read_bytes())
    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac Sim asset root is unavailable")
    environment_url = f"{root.rstrip('/')}/{stage1['environment']['asset']}"
    status, _ = omni.client.stat(environment_url)
    if status != omni.client.Result.OK:
        raise FileNotFoundError(f"official environment asset unavailable: {environment_url}")
    if not omni.usd.get_context().new_stage():
        raise RuntimeError("could not create Stage 4 mapping stage")
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    spawn = sim_utils.UsdFileCfg(usd_path=environment_url)
    spawn.func(stage1["environment"]["prim_path"], spawn)
    wait_for_loading()
    for _ in range(3):
        simulation_app.update()
    world.reset()
    world.play()
    for _ in range(3):
        world.step(render=not ARGS.headless)

    mapping = config["mapping"]
    x0, y0, z0, x1, y1, z1 = map(float, mapping["bounds_xyz_m"])
    resolution = float(mapping["voxel_resolution_m"])
    nx, ny, nz = (math.ceil((x1 - x0) / resolution),
                  math.ceil((y1 - y0) / resolution),
                  math.ceil((z1 - z0) / resolution))
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("isaacsim.asset.gen.omap", True)
    simulation_app.update()
    from isaacsim.asset.gen.omap.bindings import _omap

    context = omni.usd.get_context()
    generator = _omap.Generator(omni.physx.get_physx_interface(), context.get_stage_id())
    generator.update_settings(resolution, 1.0, 0.0, 0.5)
    start = stage2["robot"]["start_xyz"]
    origin = (float(start[0]), float(start[1]), z0)
    projection_min, projection_max = map(float, mapping["projection_height_m"])
    projection_plane_z = 0.5 * (projection_min + projection_max)
    occupancy = np.full((nz, ny, nx), 2, dtype=np.uint8)
    generator_2d_buffer = None
    generator_2d_dimensions = None
    closest_projection_distance = math.inf
    generator_xy_bounds = None
    for z_index in range(nz):
        slice_z = z0 + (z_index + 0.5) * resolution
        generator.set_transform(
            (origin[0], origin[1], slice_z),
            (x0 - origin[0], y0 - origin[1], 0.0),
            (x1 - origin[0], y1 - origin[1], 0.0),
        )
        generator.generate2d()
        dimensions = tuple(map(int, generator.get_dimensions()))
        buffer = np.asarray(generator.get_buffer(), dtype=np.float32)
        if dimensions[:2] != (nx, ny) or buffer.size != nx * ny:
            raise RuntimeError(
                f"official 2D slice z={slice_z:.3f} returned dims={dimensions}, "
                f"values={buffer.size}; expected ({nx}, {ny}, 1) and {nx * ny}"
            )
        # NVIDIA's documented index is x + y*width, so C-order gives row=y,column=x.
        raw_slice = buffer.reshape(ny, nx)
        occupancy[z_index][np.isclose(raw_slice, 0.0)] = 0
        occupancy[z_index][np.isclose(raw_slice, 1.0)] = 1
        if generator_xy_bounds is None:
            generator_xy_bounds = (
                tuple(map(float, generator.get_min_bound())),
                tuple(map(float, generator.get_max_bound())),
            )
        distance_to_projection = abs(slice_z - projection_plane_z)
        if distance_to_projection < closest_projection_distance:
            closest_projection_distance = distance_to_projection
            generator_2d_buffer = buffer.copy()
            generator_2d_dimensions = dimensions
        print(f"[Stage 4 mapping] {config['scenario']}: slice {z_index + 1}/{nz} "
              f"at z={slice_z:.2f} m", flush=True)
    if not np.any(occupancy == 1):
        raise RuntimeError(
            "official occupancy generator returned no in-bounds occupied voxels; "
            "verify PhysX collisions, free origin, and mapping bounds"
        )

    ARGS.output.mkdir(parents=True, exist_ok=True)
    voxel_path = ARGS.output / "voxel_map.npz"
    generator_3d_dimensions = (nx, ny, nz)
    generator_3d_bounds = ((x0, y0, z0), (x1, y1, z1))
    actual_origin = (x0, y0, z0)
    voxel = VoxelMap(
        occupancy, actual_origin, resolution, config["scenario"], environment_url,
        {"method": "stacked isaacsim.asset.gen.omap.Generator.generate2d slices",
         "generator_dimensions": list(generator_3d_dimensions),
         "generator_bounds": [list(value) for value in generator_3d_bounds],
         "reported_xy_bounds": [list(value) for value in generator_xy_bounds],
         "generator_origin_xy": list(origin[:2]), "buffer_axis_order": "zyx",
         "pedestrian_count": 0, "robot_spawned": False},
    )
    voxel.save(voxel_path)
    np.savez_compressed(
        ARGS.output / "official_omap_2d_raw.npz", buffer=generator_2d_buffer,
        dimensions=np.asarray(generator_2d_dimensions, dtype=np.int32),
        resolution_m=np.asarray(resolution),
        origin_xyz_m=np.asarray((origin[0], origin[1], projection_plane_z)),
        values_json=np.asarray(json.dumps({"occupied": 1.0, "free": 0.0, "unknown": 0.5})),
    )
    inflation = float(mapping["robot_radius_m"]) + float(mapping["safety_margin_m"])
    static2d = project_voxel_map(
        voxel, config["static_grid"], *map(float, mapping["projection_height_m"]),
        inflation, voxel_path.name,
    )
    static2d.save(ARGS.output / "occupancy_2d.npz")
    elapsed = time.perf_counter() - started
    summary = {
        "status": "pass", "created_utc": datetime.now(UTC).isoformat(),
        "scenario": config["scenario"], "environment_asset": environment_url,
        "input_sha256": input_hash.hexdigest(),
        "method": "official omap generate2d stacked at 0.2 m Z intervals",
        "pedestrian_count": 0, "robot_spawned": False,
        "voxel_shape_zyx": list(occupancy.shape), "voxel_resolution_m": resolution,
        "occupied_voxels": int(np.count_nonzero(occupancy == 1)),
        "free_voxels": int(np.count_nonzero(occupancy == 0)),
        "unknown_voxels": int(np.count_nonzero(occupancy == 2)),
        "official_3d_dimensions": list(generator_3d_dimensions),
        "official_2d_dimensions": list(generator_2d_dimensions),
        "grid_shape_rows_columns": list(config["static_grid"].shape),
        "grid_resolution_m": config["static_grid"].resolution_m,
        "crowd_grid_shape_rows_columns": list(config["shared_grid"].shape),
        "crowd_grid_resolution_m": config["shared_grid"].resolution_m,
        "raw_occupied_cells": int(static2d.raw.sum()),
        "inflated_occupied_cells": int(static2d.inflated.sum()),
        "unknown_cells": int(static2d.unknown.sum()), "elapsed_s": elapsed,
        "isaac_lab_api": "2.3.2 DirectRLEnv inspected; mapping uses Isaac Lab UsdFileCfg",
    }
    (ARGS.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


try:
    print(json.dumps(run(), indent=2, sort_keys=True))
except Exception as error:
    ARGS.output.mkdir(parents=True, exist_ok=True)
    failure = {"status": "fail", "error_type": type(error).__name__, "error": str(error),
               "traceback": traceback.format_exc(), "created_utc": datetime.now(UTC).isoformat()}
    (ARGS.output / "summary.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
    raise
finally:
    simulation_app.close()
