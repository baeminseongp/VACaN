#!/usr/bin/env python3
"""Validate Stage 1 Isaac/Pegasus recordings and acceptance evidence."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

REQUIRED = {
    "timestamp", "simulation_step", "simulation_time_s", "seed", "world_frame",
    "environment_state", "pedestrians", "ground_truth_density_grid", "congestion_zones",
}


def validate_episode(directory: Path) -> dict[str, object]:
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "pass" or summary.get("backend") != "isaac-sim-pegasus":
        raise AssertionError(f"{directory}: not a successful Isaac/Pegasus episode")
    records = [json.loads(line) for line in (directory / "ground_truth.jsonl").read_text().splitlines()]
    if len(records) < 2:
        raise AssertionError(f"{directory}: incomplete recording")
    previous_time = None
    starts = {}
    max_navmesh_distance = 0.0
    zone_values: dict[str, list[int]] = {}
    for expected_step, record in enumerate(records):
        missing = REQUIRED.difference(record)
        if missing:
            raise AssertionError(f"{directory}: missing {sorted(missing)}")
        if record["simulation_step"] != expected_step:
            raise AssertionError(f"{directory}: non-contiguous steps")
        timestamp = datetime.fromisoformat(record["timestamp"])
        if previous_time is not None and timestamp <= previous_time:
            raise AssertionError(f"{directory}: timestamps are not increasing")
        previous_time = timestamp
        if not record["environment_state"].get("robot_independent"):
            raise AssertionError(f"{directory}: environmental GT is not robot-independent")
        if "/Assets/Isaac/5.1/Isaac/Environments/" not in record["environment_state"]["official_isaac_asset"]:
            raise AssertionError(f"{directory}: unofficial or wrong-version asset")
        grid = record["ground_truth_density_grid"]
        rows, columns = grid["shape"]
        if len(grid["counts"]) != rows or any(len(row) != columns for row in grid["counts"]):
            raise AssertionError(f"{directory}: invalid density grid shape")
        if sum(map(sum, grid["counts"])) != len(record["pedestrians"]):
            raise AssertionError(f"{directory}: density counts do not match pedestrian GT")
        for pedestrian in record["pedestrians"]:
            starts.setdefault(pedestrian["id"], pedestrian["position_xyz_m"])
            for field in (
                "position_xyz_m", "orientation_xyzw", "velocity_xyz_m_s",
                "current_destination_xyz_m", "navmesh_distance_m",
            ):
                if field not in pedestrian:
                    raise AssertionError(f"{directory}: pedestrian missing {field}")
            max_navmesh_distance = max(max_navmesh_distance, pedestrian["navmesh_distance_m"])
        for name, zone in record["congestion_zones"].items():
            zone_values.setdefault(name, []).append(zone["pedestrian_count"])
    movements = {}
    for pedestrian in records[-1]["pedestrians"]:
        start = starts[pedestrian["id"]]
        end = pedestrian["position_xyz_m"]
        movements[pedestrian["id"]] = math.hypot(end[0] - start[0], end[1] - start[1])
    if max(movements.values()) <= 0.25:
        raise AssertionError(f"{directory}: pedestrians did not move")
    if max_navmesh_distance > 0.55:
        raise AssertionError(
            f"{directory}: pedestrian left walkable NavMesh by {max_navmesh_distance:.3f} m"
        )
    zone_ranges = {name: [min(values), max(values)] for name, values in zone_values.items()}
    congestion_rises_and_falls = any(
        any(next_value > value for value, next_value in zip(values, values[1:]))
        and any(next_value < value for value, next_value in zip(values, values[1:]))
        for values in zone_values.values()
    )
    if not congestion_rises_and_falls:
        raise AssertionError(f"{directory}: local congestion did not both increase and decrease")
    if not (directory / "top_down.png").is_file():
        raise AssertionError(f"{directory}: missing top-down visualization")
    if not (directory / "resolved_routes.json").is_file():
        raise AssertionError(f"{directory}: missing resolved NavMesh routes")
    return {
        "states": len(records), "pedestrians": len(records[0]["pedestrians"]),
        "max_movement_m": max(movements.values()),
        "max_navmesh_distance_m": max_navmesh_distance, "zone_count_ranges": zone_ranges,
        "top_down": str(directory / "top_down.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("artifacts/stage1-isaac"))
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    if manifest.get("backend") != "isaac-sim-pegasus":
        raise AssertionError("manifest does not identify the mandatory backend")
    results = {
        episode: validate_episode(args.output / episode)
        for episode in manifest.get("episodes", [])
    }
    scenarios = {Path(episode).parts[-2] for episode in results}
    if scenarios != {"office", "hospital", "warehouse"}:
        raise AssertionError(f"required scenario coverage missing: {sorted(scenarios)}")
    for repeat in manifest.get("reproducibility", []):
        if not (
            repeat.get("digest_match")
            and repeat.get("within_tolerance")
            and repeat.get("density_reproducible")
        ):
            raise AssertionError(f"repeat-seed mismatch: {repeat['episode']}")
    print(json.dumps({"status": "PASS", "episodes": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
