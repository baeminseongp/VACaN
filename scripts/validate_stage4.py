#!/usr/bin/env python3
"""Validate generated Stage 4 static maps and Stage 3 grid alignment."""

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
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "artifacts/stage4")
    parser.add_argument("--scenario", choices=("all", "office", "hospital", "warehouse"), default="all")
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    selected = manifest["episodes"] if args.scenario == "all" else [
        item for item in manifest["episodes"] if item["scenario"] == args.scenario
    ]
    for item in selected:
        scenario = item["scenario"]
        directory = args.output / item["path"]
        config = load_stage4_config(ROOT / "configs/stage4" / f"{scenario}.yaml")
        voxel = VoxelMap.load(directory / "voxel_map.npz")
        static = OccupancyMap2D.load(directory / "occupancy_2d.npz")
        if voxel.environment != scenario or not np.any(voxel.occupancy == 1):
            raise AssertionError(f"{scenario}: empty or mismatched 3D map")
        if static.geometry != config["static_grid"]:
            raise AssertionError(f"{scenario}: static occupancy geometry mismatch")
        if np.any(static.raw > static.inflated):
            raise AssertionError(f"{scenario}: inflation removed obstacles")
        for name in ("voxel_map_3d.png", "occupancy_2d.png", "official_omap_2d_raw.png",
                     "summary.json", "isaac-sim.log"):
            if not (directory / name).is_file():
                raise AssertionError(f"{scenario}: missing {name}")
        results[scenario] = {"voxel_shape": list(voxel.occupancy.shape),
                             "occupied_voxels": int(np.count_nonzero(voxel.occupancy == 1)),
                             "grid_shape": list(static.geometry.shape),
                             "grid_resolution_m": static.geometry.resolution_m,
                             "crowd_grid_shape": list(config["shared_grid"].shape),
                             "crowd_grid_resolution_m": config["shared_grid"].resolution_m,
                             "raw_occupied": int(static.raw.sum()),
                             "inflated_occupied": int(static.inflated.sum())}
    expected = {"office", "hospital", "warehouse"} if args.scenario == "all" else {args.scenario}
    if set(results) != expected:
        raise AssertionError(f"expected {sorted(expected)}, got {sorted(results)}")
    print(json.dumps({"status": "PASS", "environments": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
