#!/usr/bin/env python3
"""Validate one completed no-people teleoperated depth mapping session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacan.mapping import OccupancyMap2D, VoxelMap
from vacan.rl import load_stage4_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("office", "hospital", "warehouse"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stage4-teleop")
    args = parser.parse_args()
    directory = args.output / args.scenario
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    voxel = VoxelMap.load(directory / "voxel_map.npz")
    occupancy = OccupancyMap2D.load(directory / "occupancy_2d.npz")
    config = load_stage4_config(ROOT / "configs/stage4" / f"{args.scenario}.yaml")
    errors = []
    if summary.get("status") != "pass" or summary.get("mapping_mode") != "teleop_depth":
        errors.append("session did not complete in teleop_depth mode")
    if summary.get("pedestrian_count") != 0:
        errors.append("mapping session was not pedestrian-free")
    if summary.get("depth_frames", 0) < 1 or summary.get("depth_rays_integrated", 0) < 1:
        errors.append("no valid depth data was integrated")
    if not np.any(voxel.occupancy == 1) or not np.any(voxel.occupancy == 0):
        errors.append("3D map must contain both occupied and observed-free voxels")
    if not np.any(voxel.occupancy == 2):
        errors.append("unobserved 3D space must remain Unknown")
    if occupancy.geometry != config["static_grid"]:
        errors.append("2D occupancy does not match the configured static grid")
    if np.any(occupancy.raw > occupancy.inflated):
        errors.append("inflation removed a raw obstacle")
    for name in ("depth_pointcloud.npz", "teleop_trajectory.jsonl",
                 "voxel_map_3d.png", "occupancy_2d.png", "isaac-sim.log"):
        if not (directory / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        raise AssertionError("; ".join(errors))
    result = {
        "status": "PASS", "scenario": args.scenario,
        "pedestrian_count": 0, "depth_frames": summary["depth_frames"],
        "depth_rays_integrated": summary["depth_rays_integrated"],
        "voxel_shape_zyx": list(voxel.occupancy.shape),
        "occupied_voxels": int(np.count_nonzero(voxel.occupancy == 1)),
        "free_voxels": int(np.count_nonzero(voxel.occupancy == 0)),
        "unknown_voxels": int(np.count_nonzero(voxel.occupancy == 2)),
        "occupancy_shape": list(occupancy.geometry.shape),
        "raw_occupied_cells": int(occupancy.raw.sum()),
        "inflated_occupied_cells": int(occupancy.inflated.sum()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
