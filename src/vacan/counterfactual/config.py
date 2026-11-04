"""Strict Office-only Stage 7 configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vacan.mapping.dynamic_costmap import GridGeometry
from .intervention import parse_intervention


def load_stage7_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected Stage 7 schema_version 1")
    if data.get("scenario") != "office":
        raise ValueError("this Stage 7 implementation is intentionally Office-only")
    if data.get("policy", {}).get("frozen") is not True:
        raise ValueError("Stage 7 requires a frozen policy")
    if "freshness" in {str(item.get("factor")) for item in data.get("interventions", [])}:
        raise ValueError("Freshness is not an independent Stage 7 intervention")
    base = path.resolve().parent
    for key in ("stage6_config", "stage6_artifact", "policy_checkpoint"):
        resolved = (base / data[key]).resolve()
        if not resolved.exists():
            raise ValueError(f"{path}: missing {key}: {resolved}")
        data[f"resolved_{key}"] = str(resolved)
    grid = data["grid"]
    geometry = GridGeometry(tuple(map(float, grid["origin_xy"])),
                            tuple(map(float, grid["size_xy"])), float(grid["resolution_m"]))
    specs = [parse_intervention(item, geometry) for item in data.get("interventions", [])]
    if not specs:
        raise ValueError("at least one Stage 7 intervention is required")
    data["geometry"] = geometry
    data["parsed_interventions"] = specs
    spec_ids = {item.intervention_id for item in specs}
    experiments = data.get("experiments", [])
    experiment_ids = [str(item.get("id")) for item in experiments]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Stage 7 experiment identifiers must be unique")
    available_branches = {"original", *experiment_ids}
    for experiment in experiments:
        unknown = set(experiment.get("intervention_ids", [])) - spec_ids
        if unknown:
            raise ValueError(f"{experiment['id']}: unknown interventions {sorted(unknown)}")
        baseline = str(experiment.get("baseline_id", "original"))
        if baseline not in available_branches or baseline == experiment["id"]:
            raise ValueError(f"{experiment['id']}: invalid baseline_id {baseline}")
    cases = data.get("case_studies", {})
    required_cases = {"case_1_1", "case_1_2", "case_2", "case_3"}
    if set(cases) & required_cases != required_cases:
        raise ValueError(
            "Stage 7 requires case_1_1, case_1_2, case_2, and case_3"
        )
    referenced = {
        cases["case_1_1"].get("correct_branch"), cases["case_1_1"].get("error_branch"),
        cases["case_1_2"].get("correct_branch"), cases["case_1_2"].get("error_branch"),
        cases["case_2"].get("correct_branch"), cases["case_2"].get("direct_only_branch"),
        cases["case_2"].get("error_branch"),
        cases["case_3"].get("normal_branch"), cases["case_3"].get("failing_branch"),
    }
    if not referenced <= available_branches:
        raise ValueError(f"case studies reference unknown branches {sorted(referenced - available_branches)}")
    behavior = cases["case_3"].get("behavior_detection", {})
    positive_behavior_fields = {
        "window_s", "stuck_max_displacement_m", "stuck_max_goal_progress_m",
        "oscillation_max_displacement_m", "oscillation_min_yaw_travel_rad",
        "angular_deadband_rad_s",
    }
    if any(float(behavior.get(key, 0)) <= 0 for key in positive_behavior_fields):
        raise ValueError("case_3 behavior detection thresholds must be positive")
    if int(behavior.get("oscillation_min_angular_reversals", 0)) <= 0:
        raise ValueError("case_3 oscillation reversal threshold must be positive")
    if float(data.get("comparison_tolerance", -1)) < 0:
        raise ValueError("comparison_tolerance must be non-negative")
    return data
