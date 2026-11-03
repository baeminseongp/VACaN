#!/usr/bin/env python3
"""Validate Stage 2 physical navigation recordings and interface checks."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

REQUIRED_RECORD_FIELDS = {
    "timestamp", "simulation_step", "simulation_time_s", "seed", "scenario",
    "world_frame", "environment_state", "observation", "privileged_log",
    "reward", "terminated", "truncated", "info",
}


def planar_distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def validate_episode(directory: Path) -> dict[str, object]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "pass":
        raise AssertionError(f"{directory}: Isaac episode did not pass")
    if summary.get("backend") != "isaac-sim-pegasus-nova-carter":
        raise AssertionError(f"{directory}: wrong backend")
    expected_assets = {
        "office": "Isaac/Environments/Office/office.usd",
        "hospital": "Isaac/Environments/Hospital/hospital.usd",
        "warehouse": "Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    }
    if not summary["environment_asset"].endswith(expected_assets[summary["scenario"]]):
        raise AssertionError(f"{directory}: unexpected official environment asset")
    articulation = summary["articulation"]
    if articulation["wheel_joint_names"] != ["joint_wheel_left", "joint_wheel_right"]:
        raise AssertionError(f"{directory}: Nova Carter wheel joints not verified")
    if articulation["collision_prim_count"] <= 0 or not articulation["articulation_roots"]:
        raise AssertionError(f"{directory}: articulation or collision geometry missing")
    if summary["controller"]["motion_model"] != \
            "physical wheel velocity targets; no pose writes during step":
        raise AssertionError(f"{directory}: physical drive evidence missing")

    interface = summary["interface_checks"]
    if interface["yaw_delta_rad"] < 0.25:
        raise AssertionError(f"{directory}: rotation check failed")
    if interface["stopped_angular_speed_rad_s"] > 0.15:
        raise AssertionError(f"{directory}: angular stop check failed")
    if interface["reset_position_error_m"] > 0.08:
        raise AssertionError(f"{directory}: reset check failed")

    records = [
        json.loads(line)
        for line in (directory / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(records) < 2:
        raise AssertionError(f"{directory}: trajectory recording is incomplete")
    previous_timestamp = None
    previous_step = None
    pedestrian_ids = None
    maximum_wheel_target = 0.0
    maximum_wheel_speed = 0.0
    for record in records:
        missing = REQUIRED_RECORD_FIELDS.difference(record)
        if missing:
            raise AssertionError(f"{directory}: missing fields {sorted(missing)}")
        if record["world_frame"] != "isaac_world_enu":
            raise AssertionError(f"{directory}: inconsistent world frame")
        if not record["environment_state"].get("robot_independent"):
            raise AssertionError(f"{directory}: environmental state is not robot-independent")
        timestamp = datetime.fromisoformat(record["timestamp"])
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise AssertionError(f"{directory}: timestamps are not increasing")
        if previous_step is not None and record["simulation_step"] != previous_step + 1:
            raise AssertionError(f"{directory}: simulation steps are not contiguous")
        previous_timestamp, previous_step = timestamp, record["simulation_step"]
        people = record["privileged_log"]["pedestrians"]
        ids = tuple(item["id"] for item in people)
        pedestrian_ids = ids if pedestrian_ids is None else pedestrian_ids
        if ids != pedestrian_ids:
            raise AssertionError(f"{directory}: pedestrian synchronization changed IDs")
        for person in people:
            for field in ("position_xyz_m", "orientation_xyzw", "velocity_xyz_m_s",
                          "current_destination_xyz_m"):
                if field not in person:
                    raise AssertionError(f"{directory}: pedestrian missing {field}")
        maximum_wheel_target = max(
            maximum_wheel_target,
            *(abs(float(value)) for value in record["info"]["wheel_velocity_targets_rad_s"]),
        )
        maximum_wheel_speed = max(
            maximum_wheel_speed,
            *(abs(float(value)) for value in
              record["observation"]["robot"]["wheel_velocities_rad_s"]),
        )

    first, final = records[0], records[-1]
    first_robot = first["observation"]["robot"]
    final_robot = final["observation"]["robot"]
    displacement = planar_distance(first_robot["position_xyz_m"], final_robot["position_xyz_m"])
    initial_goal_distance = float(first["observation"]["goal_distance_m"])
    best_goal_distance = min(float(item["observation"]["goal_distance_m"]) for item in records)
    if displacement < 0.25 or maximum_wheel_target < 0.1 or maximum_wheel_speed < 0.1:
        raise AssertionError(f"{directory}: no evidence of physical wheel-driven translation")
    if best_goal_distance > initial_goal_distance - 0.25:
        raise AssertionError(f"{directory}: robot did not approach the goal")
    if summary["trajectory"]["stopped_speed_m_s"] > 0.15:
        raise AssertionError(f"{directory}: stop command did not stop the base")
    if not (directory / "top_down.png").is_file():
        raise AssertionError(f"{directory}: top-down visualization missing")

    return {
        "states": len(records),
        "pedestrians_per_state": len(pedestrian_ids or ()),
        "displacement_m": displacement,
        "initial_goal_distance_m": initial_goal_distance,
        "best_goal_distance_m": best_goal_distance,
        "max_wheel_target_rad_s": maximum_wheel_target,
        "max_wheel_speed_rad_s": maximum_wheel_speed,
        "goal_reached": summary["trajectory"]["goal_reached"],
        "collision_probe": summary.get("collision_probe"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("artifacts/stage2-isaac"))
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("backend") != "isaac-sim-pegasus-nova-carter":
        raise AssertionError("manifest does not identify the mandatory Stage 2 backend")
    if manifest.get("isaaclab_tree_modified") is not False:
        raise AssertionError("run does not declare Isaac Lab tree read-only")
    results = {}
    for item in manifest.get("episodes", []):
        results[item["path"]] = validate_episode(args.output / item["path"])
    scenarios = {Path(path).parts[-2] for path in results}
    if scenarios != {"office", "hospital", "warehouse"}:
        raise AssertionError(f"required scenario coverage missing: {sorted(scenarios)}")
    office = next(value for path, value in results.items() if Path(path).parts[-2] == "office")
    probe = office.get("collision_probe")
    if not probe or not probe.get("detected"):
        raise AssertionError("Office controlled collision probe did not trigger termination detection")
    print(json.dumps({"status": "PASS", "episodes": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
