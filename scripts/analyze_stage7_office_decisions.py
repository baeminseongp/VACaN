#!/usr/bin/env python3
"""Evaluate Office counterfactuals at fixed recorded Stage 6 decision states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vacan-stage7-matplotlib")
sys.path.insert(0, str(ROOT / "src"))

from vacan.counterfactual.config import load_stage7_config
from vacan.counterfactual.dataset import (
    render_decision_figure, write_json, write_representation_arrays,
)
from vacan.counterfactual.intervention import decision_costmap_intervention
from vacan.counterfactual.policy import (
    evaluate_decision_pair, replace_dynamic_policy_channel, snapshot_from_record,
)
from vacan.mapping.dynamic_costmap import DynamicCostmap, load_stage3_config
from vacan.mapping.static_map import OccupancyMap2D
from vacan.reference.config import load_stage6_config
from vacan.rl.config import load_stage4_config
from vacan.rl.stage4 import ObservationBuilder, ObservationSpec
from vacan.robot.stage2 import load_stage2_config, velocity_limits


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage7/office.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stage7/office/decision")
    args = parser.parse_args()
    config = load_stage7_config(args.config.resolve())
    stage6 = load_stage6_config(Path(config["resolved_stage6_config"]))
    stage4 = load_stage4_config(Path(stage6["resolved_stage4_config"]))
    stage3 = load_stage3_config(Path(stage4["resolved_stage3_config"]))
    stage3_probe = DynamicCostmap(stage3, int(config["seed"]))
    stage2 = load_stage2_config(Path(stage6["resolved_stage2_config"]))
    limits = velocity_limits(stage2)
    observation_cfg = stage4["observation"]
    builder = ObservationBuilder(
        OccupancyMap2D.load(Path(stage6["resolved_static_map"])),
        ObservationSpec(
            int(observation_cfg["crop_rows"]), int(observation_cfg["crop_columns"]),
            bool(observation_cfg["rotate_with_robot"]), float(observation_cfg["crowd_clip"]),
            float(observation_cfg["max_goal_distance_m"]), limits.max_linear_m_s,
            limits.max_angular_rad_s,
        ), dynamic_geometry=config["geometry"],
    )
    artifact = Path(config["resolved_stage6_artifact"])
    trajectories = _records(artifact / "drl-trajectory.jsonl")
    truths = _records(artifact / "ground-truth-drl.jsonl")
    model = RecurrentPPO.load(config["resolved_policy_checkpoint"], device="cpu")
    by_id = {spec.intervention_id: spec for spec in config["parsed_interventions"]}
    all_arrays: dict[str, np.ndarray] = {}
    arrays_by_experiment: dict[str, dict[str, np.ndarray]] = {}
    observations_by_experiment: dict[str, dict[str, np.ndarray]] = {}
    snapshots_by_experiment = {}
    results = []
    for experiment in config["experiments"]:
        specs = [by_id[identifier] for identifier in experiment["intervention_ids"]]
        target_time = min(spec.start_time_s for spec in specs)
        event_index = min(range(len(trajectories) - 1),
                          key=lambda item: abs(
                              trajectories[item]["simulation_time_s"] - target_time))
        decision_index = event_index + 1
        event_record, record, truth = (
            trajectories[event_index], trajectories[decision_index], truths[event_index]
        )
        snapshot = snapshot_from_record(record)
        diagnostics = event_record["costmap_diagnostics"]
        modified = diagnostics["cost_after"]
        logs = []
        for spec in specs:
            modified, changes = decision_costmap_intervention(
                modified, diagnostics, truth["ground_truth_density"], spec,
                ground_truth_pedestrians=truth["pedestrians"],
                cctv_cells=stage3_probe.cctv_cells, geometry=config["geometry"],
            )
            logs.extend(changes)
        counterfactual_observation = replace_dynamic_policy_channel(
            snapshot.observation, np.asarray(modified, dtype=np.float32), builder,
            tuple(event_record["position_xy_m"]), float(event_record["yaw_rad"]),
        )
        result, arrays = evaluate_decision_pair(model, snapshot, counterfactual_observation)
        result.update({
            "experiment_id": experiment["id"],
            "intervention_ids": experiment["intervention_ids"],
            "intervention_logs": logs,
            "grid_costmap_l2": float(np.linalg.norm(
                np.asarray(modified) - np.asarray(diagnostics["cost_after"]))),
            "replay_semantics": "decision_level_fixed_state",
            "source_event_time_s": float(event_record["simulation_time_s"]),
            "decision_time_s": float(record["simulation_time_s"]),
        })
        results.append(result)
        arrays_by_experiment[experiment["id"]] = arrays
        observations_by_experiment[experiment["id"]] = counterfactual_observation
        snapshots_by_experiment[experiment["id"]] = snapshot
        for key, value in arrays.items():
            all_arrays[f"{experiment['id']}::{key}"] = value
    results_by_id = {item["experiment_id"]: item for item in results}
    full_result = results_by_id["full_information"]
    full_arrays = arrays_by_experiment["full_information"]
    full_observation = observations_by_experiment["full_information"]
    for result in results:
        identifier = result["experiment_id"]
        if identifier in {"full_information", "no_op"}:
            baseline_output = result["original"]
            baseline_observation = snapshots_by_experiment[identifier].observation
            primary = result["differences"]
            baseline_name = "actual_drl"
        else:
            baseline_output = full_result["counterfactual"]
            baseline_observation = full_observation
            branch_output = result["counterfactual"]
            first_action = np.asarray(baseline_output["action"], dtype=float)
            second_action = np.asarray(branch_output["action"], dtype=float)
            first_hidden = np.asarray(baseline_output["next_actor_hidden"], dtype=float)
            second_hidden = np.asarray(branch_output["next_actor_hidden"], dtype=float)
            feature_metrics = {}
            branch_arrays = arrays_by_experiment[identifier]
            for key in sorted(item.removeprefix("counterfactual::") for item in full_arrays
                              if item.startswith("counterfactual::")):
                first = full_arrays[f"counterfactual::{key}"]
                second = branch_arrays[f"counterfactual::{key}"]
                difference = second.astype(np.float64) - first.astype(np.float64)
                denominator = max(float(np.linalg.norm(first)), 1e-12)
                feature_metrics[key] = {
                    "shape": list(first.shape), "l2": float(np.linalg.norm(difference)),
                    "normalized_l2": float(np.linalg.norm(difference) / denominator),
                }
            primary = {
                "action_l2": float(np.linalg.norm(second_action - first_action)),
                "linear_action": float(second_action[0] - first_action[0]),
                "angular_action": float(second_action[1] - first_action[1]),
                "critic_value": float(branch_output["critic_value"]
                                      - baseline_output["critic_value"]),
                "actor_hidden_l2": float(np.linalg.norm(second_hidden - first_hidden)),
                "input_costmap_l2": float(np.linalg.norm(
                    observations_by_experiment[identifier]["map"] - full_observation["map"])),
                "features": feature_metrics,
            }
            baseline_name = "full_information_drl"
        result["differences_vs_actual"] = result["differences"]
        result["differences"] = primary
        result["primary_comparison_baseline"] = baseline_name
        result["baseline_policy_output"] = baseline_output
        render_payload = {"original": baseline_output,
                          "counterfactual": result["counterfactual"],
                          "differences": primary}
        render_decision_figure(
            args.output / f"decision-{identifier}.png",
            baseline_observation["map"][1], observations_by_experiment[identifier]["map"][1],
            render_payload, f"Office {identifier}: {baseline_name}",
        )
    metadata = {
        "schema_version": 1, "scenario": "office", "seed": config["seed"],
        "policy_checkpoint": config["resolved_policy_checkpoint"],
        "stage6_artifact": str(artifact), "results": results,
        "primary_comparison_baseline": "full_information_drl",
        "stage6_reference_role": "auxiliary_efficiency_comparator",
        "case_study_findings_fabricated": False,
    }
    write_json(args.output / "decision-results.json", metadata)
    write_representation_arrays(args.output / "policy-representations.npz", all_arrays)
    print(json.dumps({"status": "pass", "experiments": len(results),
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
