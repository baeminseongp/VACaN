"""Navigation inefficiency classification and timestamp-aligned diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .cost import CostBreakdown


@dataclass(frozen=True)
class ComparisonResult:
    delta_j: float
    tolerance: float
    inefficient: bool
    outcome_class: str
    component_differences: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_costs(drl: CostBreakdown, reference: CostBreakdown, *,
                  drl_success: bool, reference_success: bool,
                  tolerance: float) -> ComparisonResult:
    if drl_success and reference_success:
        outcome = "both_successful"
    elif drl_success:
        outcome = "drl_success_reference_unsuccessful"
    elif reference_success:
        outcome = "drl_unsuccessful_reference_success"
    else:
        outcome = "both_unsuccessful"
    fields = ("travel_time_s", "path_length_m", "collision_events",
              "social_violation_integral_s", "smoothness", "terminal_failure", "total")
    differences = {name: float(getattr(drl, name) - getattr(reference, name)) for name in fields}
    delta = differences["total"]
    return ComparisonResult(delta, float(tolerance), delta > tolerance, outcome, differences)


def trajectory_divergence(drl: list[dict[str, Any]], reference: list[dict[str, Any]], *,
                          distance_threshold_m: float,
                          heading_change_threshold_rad_s: float) -> list[dict[str, float]]:
    """Interpolate reference positions at DRL timestamps; never align by index."""
    if not drl or not reference:
        return []
    reference_times = np.asarray([record["simulation_time_s"] for record in reference], dtype=float)
    reference_xy = np.asarray([record["position_xy_m"] for record in reference], dtype=float)
    candidates, previous_angular = [], None
    for sample in drl:
        time_s = float(sample["simulation_time_s"])
        if time_s < reference_times[0] or time_s > reference_times[-1]:
            continue
        aligned = np.asarray([np.interp(time_s, reference_times, reference_xy[:, axis])
                              for axis in (0, 1)])
        distance = float(np.linalg.norm(np.asarray(sample["position_xy_m"]) - aligned))
        angular = float(sample.get("action", (0.0, 0.0))[1])
        heading_change = 0.0 if previous_angular is None else abs(angular - previous_angular)
        previous_angular = angular
        if distance >= distance_threshold_m or heading_change >= heading_change_threshold_rad_s:
            candidates.append({"simulation_time_s": time_s,
                               "spatial_divergence_m": distance,
                               "angular_action_change_rad_s": heading_change})
    return candidates


def information_error_candidates(
    drl: list[dict[str, Any]], ground_truth: list[dict[str, Any]], *,
    crowd_error_threshold: float, intention_error_threshold: float,
    speed_change_threshold_m_s: float, angular_change_threshold_rad_s: float,
    local_penalty_threshold: float,
) -> list[dict[str, Any]]:
    """Flag traceable analysis candidates without assigning causal labels."""
    if len(drl) != len(ground_truth):
        raise ValueError("DRL and Ground Truth records must be timestep aligned")
    candidates = []
    previous_action = np.zeros(2, dtype=float)
    for index, (sample, truth) in enumerate(zip(drl, ground_truth)):
        reasons, cells = [], []
        diagnostics = sample.get("costmap_diagnostics") or {}
        observed = diagnostics.get("cost_after")
        actual = truth.get("ground_truth_density")
        max_crowd_error = intention_error = 0.0
        if observed is not None and actual is not None:
            observed_array = np.asarray(observed, dtype=float)
            actual_array = np.asarray(actual, dtype=float)
            if observed_array.shape == actual_array.shape:
                error = np.abs(observed_array - actual_array)
                max_crowd_error = float(error.max(initial=0.0))
                cells = [list(map(int, cell)) for cell in np.argwhere(
                    error >= crowd_error_threshold)]
                if max_crowd_error >= crowd_error_threshold:
                    reasons.append("crowd_estimation_error")
                incoming = diagnostics.get("intention_incoming_count")
                if incoming is not None and index + 1 < len(ground_truth):
                    next_actual = ground_truth[index + 1].get("ground_truth_density")
                    if next_actual is not None:
                        realized_incoming = np.maximum(
                            0.0, np.asarray(next_actual, dtype=float) - actual_array)
                        intention_error = float(np.abs(
                            np.asarray(incoming, dtype=float) - realized_incoming).sum())
                        if intention_error >= intention_error_threshold:
                            reasons.append("intention_discrepancy")
        action = np.asarray(sample.get("action", (0.0, 0.0)), dtype=float)
        speed_change, angular_change = map(float, np.abs(action - previous_action))
        previous_action = action
        if speed_change >= speed_change_threshold_m_s:
            reasons.append("large_speed_change")
        if angular_change >= angular_change_threshold_rad_s:
            reasons.append("large_heading_command_change")
        reward_terms = sample.get("reward_terms") or {}
        local_penalty = float(sum(max(0.0, -float(reward_terms.get(key, 0.0)))
                                  for key in ("collision", "social", "time", "smooth")))
        if local_penalty >= local_penalty_threshold:
            reasons.append("high_local_navigation_penalty")
        if reasons:
            candidates.append({
                "simulation_time_s": float(sample["simulation_time_s"]),
                "reasons": reasons, "grid_cells": cells,
                "max_crowd_estimation_error": max_crowd_error,
                "intention_discrepancy_l1": intention_error,
                "linear_action_change_m_s": speed_change,
                "angular_action_change_rad_s": angular_change,
                "local_navigation_penalty": local_penalty,
            })
    return candidates


def group_candidate_intervals(candidates: list[dict[str, Any]], *,
                              maximum_gap_s: float) -> list[dict[str, Any]]:
    """Merge adjacent candidate points into traceable temporal intervals."""
    if maximum_gap_s <= 0:
        raise ValueError("maximum candidate interval gap must be positive")
    intervals: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["simulation_time_s"]):
        time_s = float(candidate["simulation_time_s"])
        if not intervals or time_s - intervals[-1]["end_time_s"] > maximum_gap_s + 1e-9:
            intervals.append({
                "start_time_s": time_s, "end_time_s": time_s, "sample_count": 1,
                "reasons": sorted(set(candidate["reasons"])),
                "grid_cells": candidate["grid_cells"],
                "max_crowd_estimation_error": candidate["max_crowd_estimation_error"],
                "max_intention_discrepancy_l1": candidate["intention_discrepancy_l1"],
                "max_local_navigation_penalty": candidate["local_navigation_penalty"],
            })
            continue
        interval = intervals[-1]
        interval["end_time_s"] = time_s
        interval["sample_count"] += 1
        interval["reasons"] = sorted(set(interval["reasons"]) | set(candidate["reasons"]))
        interval["grid_cells"] = sorted({tuple(cell) for cell in interval["grid_cells"]
                                           + candidate["grid_cells"]})
        interval["grid_cells"] = [list(cell) for cell in interval["grid_cells"]]
        interval["max_crowd_estimation_error"] = max(
            interval["max_crowd_estimation_error"], candidate["max_crowd_estimation_error"])
        interval["max_intention_discrepancy_l1"] = max(
            interval["max_intention_discrepancy_l1"], candidate["intention_discrepancy_l1"])
        interval["max_local_navigation_penalty"] = max(
            interval["max_local_navigation_penalty"], candidate["local_navigation_penalty"])
    return intervals
