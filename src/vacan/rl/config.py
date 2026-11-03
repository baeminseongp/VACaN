"""Strict Stage 4 YAML loading."""

from pathlib import Path
from typing import Any

import yaml

from vacan.mapping.dynamic_costmap import GridGeometry, load_stage3_config


def load_stage4_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected Stage 4 schema_version 1")
    if data.get("scenario") not in {"office", "hospital", "warehouse"}:
        raise ValueError(f"{path}: invalid scenario")
    if data.get("world_frame") != "isaac_world_enu":
        raise ValueError(f"{path}: inconsistent world frame")
    mapping = data.get("mapping", {})
    if float(mapping.get("voxel_resolution_m", 0)) <= 0:
        raise ValueError(f"{path}: invalid voxel resolution")
    if float(mapping.get("occupancy_2d_resolution_m", 0)) <= 0:
        raise ValueError(f"{path}: invalid 2D occupancy resolution")
    bounds = mapping.get("bounds_xyz_m")
    if not isinstance(bounds, list) or len(bounds) != 6 or any(bounds[i] >= bounds[i + 3] for i in range(3)):
        raise ValueError(f"{path}: invalid 3D bounds")
    frequencies = data.get("frequencies", {})
    if float(frequencies.get("physics_hz", 0)) % float(frequencies.get("control_hz", 1)) != 0:
        raise ValueError(f"{path}: control frequency must divide physics frequency")
    stage3_path = (path.parent / data["stage3_config"]).resolve()
    stage3 = load_stage3_config(stage3_path)
    grid = stage3["grid"]
    data["shared_grid"] = GridGeometry(tuple(map(float, grid["origin_xy"])),
                                       tuple(map(float, grid["size_xy"])),
                                       float(grid["resolution_m"]))
    data["resolved_stage3_config"] = str(stage3_path)
    bounds = tuple(map(float, mapping["bounds_xyz_m"]))
    data["static_grid"] = GridGeometry(
        (bounds[0], bounds[1]), (bounds[3] - bounds[0], bounds[4] - bounds[1]),
        float(mapping["occupancy_2d_resolution_m"]),
    )
    return data
