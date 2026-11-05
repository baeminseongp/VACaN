"""Strict Stage 6 configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_stage6_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected Stage 6 schema_version 1")
    if data.get("scenario") not in {"office", "hospital", "warehouse"}:
        raise ValueError(f"{path}: invalid Stage 6 scenario")
    if data.get("reference", {}).get("algorithm") != "time_expanded_astar_constant_velocity":
        raise ValueError(f"{path}: unsupported reference planner")
    reference = data["reference"]
    if float(reference.get("time_step_s", 0)) <= 0 or float(reference.get("horizon_s", 0)) <= 0:
        raise ValueError(f"{path}: invalid planning time configuration")
    if bool(reference.get("uses_actual_future_trajectory", True)):
        raise ValueError(f"{path}: primary reference must not use oracle future trajectories")
    cost = data.get("shared_cost", {})
    if cost.get("objective", "composite_navigation") not in {
            "composite_navigation", "trajectory_efficiency"}:
        raise ValueError(f"{path}: unsupported shared cost objective")
    required_weights = {"time", "length", "collision", "social", "smoothness",
                        "terminal_failure"}
    if set(cost.get("weights", {})) != required_weights:
        raise ValueError(f"{path}: shared cost weights are incomplete")
    if float(data.get("comparison", {}).get("inefficiency_tolerance", -1)) < 0:
        raise ValueError(f"{path}: invalid inefficiency tolerance")
    evaluation = data.get("evaluation", {})
    if int(evaluation.get("pedestrian_count", 0)) <= 0:
        raise ValueError(f"{path}: evaluation.pedestrian_count must be positive")
    for split in ("validation_seeds", "test_seeds"):
        if not isinstance(evaluation.get(split), list) or not evaluation[split]:
            raise ValueError(f"{path}: evaluation.{split} is required")
    base = path.resolve().parent
    for key in ("stage2_config", "stage4_config", "stage5_config", "static_map",
                "policy_checkpoint"):
        resolved = (base / data[key]).resolve()
        if not resolved.is_file():
            raise ValueError(f"{path}: missing {key}: {resolved}")
        data[f"resolved_{key}"] = str(resolved)
    return data
