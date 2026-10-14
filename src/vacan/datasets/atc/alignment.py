"""Coordinate-system checks between normalized trajectories and the ROS map."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .map import RosMap


def _world_to_local(ros_map: RosMap, x: float, y: float) -> tuple[float, float]:
    return ros_map.local_point(x, y)


def alignment_report(
    ros_map: RosMap,
    trajectory_path: Path,
    *,
    warning_threshold: float = 0.95,
) -> dict[str, object]:
    total = inside = 0
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    with trajectory_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            x, y = float(row["x_m"]), float(row["y_m"])
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            local_x, local_y = _world_to_local(ros_map, x, y)
            if 0 <= local_x <= ros_map.width * ros_map.resolution and 0 <= local_y <= ros_map.height * ros_map.resolution:
                inside += 1
            total += 1
    fraction = inside / total if total else 0.0
    return {
        "sample_count": total,
        "inside_map_count": inside,
        "inside_map_fraction": fraction,
        "warning_threshold": warning_threshold,
        "status": "PASS" if total and fraction >= warning_threshold else "REVIEW",
        "trajectory_bounds": None if not total else {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
        "map_bounds_polygon": ros_map.bounds_polygon(),
    }


def write_alignment_report(ros_map: RosMap, trajectory_path: Path, output_path: Path) -> dict[str, object]:
    report = alignment_report(ros_map, trajectory_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
