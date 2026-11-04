#!/usr/bin/env python3
"""Build simulator-independent Office Stage 7 attribution and query indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    episode = json.loads((run / "episode/episode-results.json").read_text())
    decision = json.loads((run / "decision/decision-results.json").read_text())
    branches = {item["branch_id"]: item for item in episode["branches"]}
    attribution = dict(episode["factor_attribution"])
    congestion_delta = attribution["delta_j_congestion_error"]
    intention_delta = attribution["delta_j_intention_error"]
    summary = {
        "schema_version": 1, "scenario": "office", "seed": episode["seed"],
        "policy_checkpoint": episode["policy_checkpoint"], "policy_frozen": True,
        "replay_semantics": episode["replay_semantics"],
        "primary_comparison_baseline": "full_information_drl",
        "stage6_reference": episode["stage6_auxiliary_reference"],
        "matched_randomness": "same seed and deterministic reset for every branch",
        "documented_nondeterminism": "none observed within the verified no-op tolerance",
        "pedestrians_react_to_robot": episode["pedestrians_react_to_robot"],
        "factor_attribution": attribution,
        "branches": episode["branches"],
        "decision_results": [{
            "experiment_id": item["experiment_id"],
            "input_costmap_l2": item["differences"]["input_costmap_l2"],
            "feature_l2": item["differences"]["features"],
            "actor_hidden_l2": item["differences"]["actor_hidden_l2"],
            "action_l2": item["differences"]["action_l2"],
            "critic_value_difference": item["differences"]["critic_value"],
        } for item in decision["results"]],
        "no_op_pass": branches["no_op"].get("full_trajectory_equal", False)
        and branches["no_op"]["delta_j_vs_actual"] == 0,
        "case_studies": episode["case_studies"],
        "case_study_status": {
            identifier: ("candidate" if value["navigation_failure_candidate"]
                         else "negative_result")
            for identifier, value in episode["case_studies"].items()
            if identifier.startswith("case_")
        },
        "legacy_five_way_deltas": {
            "congestion_error_vs_full_information": congestion_delta,
            "intention_error_vs_full_information": intention_delta,
        },
        "causal_claim": "controlled execution-specific association only",
    }
    write_json(run / "analysis-summary.json", summary)

    index = []
    branch_root = run / "episode/branches"
    for branch_id in branches:
        trajectory_path = branch_root / branch_id / "trajectory.jsonl"
        for line_number, record in enumerate(read_jsonl(trajectory_path), 1):
            index.append({
                "entity": "decision", "scenario_id": "office",
                "episode_id": f"office-{episode['seed']}",
                "counterfactual_branch": branch_id,
                "timestamp_s": record["simulation_time_s"],
                "robot_position_xy_m": record["position_xy_m"], "robot_yaw_rad": record["yaw_rad"],
                "action": record["action"],
                "critic_value": (record.get("policy_diagnostics") or {}).get("value_estimate"),
                "trajectory_record": f"episode/branches/{branch_id}/trajectory.jsonl:{line_number}",
                "ground_truth_record": f"episode/branches/{branch_id}/ground-truth.jsonl:{line_number}",
            })
        for event in read_jsonl(branch_root / branch_id / "intervention-events.jsonl"):
            index.append({
                "entity": "intervention", "scenario_id": "office",
                "episode_id": event["episode_id"], "counterfactual_branch": branch_id,
                "timestamp_s": event["simulation_time_s"], "factor": event["factor"],
                "intervention_id": event["intervention_id"],
                "grid_cells": ([event["grid_cell"]] if "grid_cell" in event
                               else event.get("affected_grid_cells", [])),
                "pedestrian_id": event.get("pedestrian_id"),
            })
    index.sort(key=lambda item: (item["timestamp_s"], item["counterfactual_branch"], item["entity"]))
    (run / "spatiotemporal-index.jsonl").write_text(
        "".join(json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n" for item in index),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1, "scenario": "office", "world_frame": "isaac_world_enu",
        "static_map_reference": "artifacts/stage4/office/occupancy_2d.npz",
        "policy_checkpoint": episode["policy_checkpoint"],
        "files": {
            "analysis": "analysis-summary.json", "query_index": "spatiotemporal-index.jsonl",
            "decision_metadata": "decision/decision-results.json",
            "policy_representations": "decision/policy-representations.npz",
            "episode_metadata": "episode/episode-results.json",
            "branch_trajectories": "episode/branch-trajectories.npz",
        },
        "large_static_maps_duplicated": False,
        "simulator_required_to_load": False,
        "tabular_format": "JSONL (PyArrow is not installed in the verified VACaN environment)",
    }
    write_json(run / "dataset-manifest.json", manifest)
    print(json.dumps({"status": "pass", "records": len(index), "no_op_pass": summary["no_op_pass"],
                      "output": str(run)}, sort_keys=True))


if __name__ == "__main__":
    main()
