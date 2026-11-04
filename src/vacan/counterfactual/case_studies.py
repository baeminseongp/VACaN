"""Metrics for the three Office Stage 7 case studies.

The functions in this module never alter a policy or simulator state. They
compare already executed, timestamped branches and keep navigation cost,
trajectory divergence, and information error as separate evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _path_length(records: list[dict[str, Any]]) -> float:
    if len(records) < 2:
        return 0.0
    positions = np.asarray([item["position_xy_m"] for item in records], dtype=float)
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def paired_trajectory_metrics(
    correct: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    *,
    trajectory_threshold_m: float,
    action_threshold_l2: float,
) -> dict[str, Any]:
    """Compare two closed-loop executions by timestamp, not record index."""
    if not correct or not comparison:
        raise ValueError("paired trajectory comparison requires two non-empty trajectories")
    correct_time = np.asarray([item["simulation_time_s"] for item in correct], dtype=float)
    correct_xy = np.asarray([item["position_xy_m"] for item in correct], dtype=float)
    correct_action = np.asarray([item.get("action", (0.0, 0.0)) for item in correct], dtype=float)
    spatial_samples: list[tuple[float, float]] = []
    action_samples: list[tuple[float, float]] = []
    for item in comparison:
        now = float(item["simulation_time_s"])
        if now < correct_time[0] - 1e-9 or now > correct_time[-1] + 1e-9:
            continue
        baseline_xy = np.asarray([
            np.interp(now, correct_time, correct_xy[:, axis]) for axis in (0, 1)
        ])
        baseline_action = np.asarray([
            np.interp(now, correct_time, correct_action[:, axis]) for axis in (0, 1)
        ])
        spatial_samples.append((now, float(np.linalg.norm(
            np.asarray(item["position_xy_m"], dtype=float) - baseline_xy
        ))))
        action_samples.append((now, float(np.linalg.norm(
            np.asarray(item.get("action", (0.0, 0.0)), dtype=float) - baseline_action
        ))))
    max_spatial = max((value for _, value in spatial_samples), default=0.0)
    max_action = max((value for _, value in action_samples), default=0.0)
    first_spatial = next((time for time, value in spatial_samples
                          if value >= trajectory_threshold_m), None)
    first_action = next((time for time, value in action_samples
                         if value >= action_threshold_l2), None)
    correct_goal = bool(correct[-1].get("goal_reached", False))
    comparison_goal = bool(comparison[-1].get("goal_reached", False))
    correct_collision = any(bool(item.get("collision", False)) for item in correct)
    comparison_collision = any(bool(item.get("collision", False)) for item in comparison)
    correct_length, comparison_length = _path_length(correct), _path_length(comparison)
    correct_duration = float(correct[-1]["simulation_time_s"] - correct[0]["simulation_time_s"])
    comparison_duration = float(
        comparison[-1]["simulation_time_s"] - comparison[0]["simulation_time_s"]
    )
    outcome_changed = (correct_goal, correct_collision) != (
        comparison_goal, comparison_collision
    )
    return {
        "maximum_spatial_divergence_m": max_spatial,
        "maximum_action_divergence_l2": max_action,
        "first_spatial_divergence_time_s": first_spatial,
        "first_action_divergence_time_s": first_action,
        "trajectory_divergence_threshold_m": float(trajectory_threshold_m),
        "action_divergence_threshold_l2": float(action_threshold_l2),
        "path_difference_observed": bool(
            max_spatial >= trajectory_threshold_m or outcome_changed
        ),
        "action_difference_observed": bool(max_action >= action_threshold_l2),
        "outcome_changed": outcome_changed,
        "correct": {
            "goal_reached": correct_goal,
            "collision": correct_collision,
            "travel_time_s": correct_duration,
            "path_length_m": correct_length,
        },
        "comparison": {
            "goal_reached": comparison_goal,
            "collision": comparison_collision,
            "travel_time_s": comparison_duration,
            "path_length_m": comparison_length,
        },
        "travel_time_difference_s": comparison_duration - correct_duration,
        "path_length_difference_m": comparison_length - correct_length,
    }


def _truth_at(
    truth_times: np.ndarray, truth_layers: np.ndarray, timestamp: float
) -> np.ndarray:
    index = int(np.searchsorted(truth_times, timestamp + 1e-9, side="right") - 1)
    return truth_layers[max(0, min(index, len(truth_layers) - 1))]


def freshness_validity_metrics(
    trajectory: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    crowd_clip: float,
    cells: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Compare timestamp freshness with GT density stability.

    GT validity is one minus the clipped change between the exact density at
    the cell's anchor timestamp and its exact density at the decision time.
    It measures whether the anchored information still describes the current
    environment; it is not exposed to the policy.
    """
    if crowd_clip <= 0:
        raise ValueError("crowd_clip must be positive")
    if not ground_truth:
        raise ValueError("freshness validity needs Ground Truth records")
    truth_times = np.asarray([item["simulation_time_s"] for item in ground_truth], dtype=float)
    truth_layers = np.asarray([item["ground_truth_density"] for item in ground_truth], dtype=float)
    selected = None if cells is None else {tuple(map(int, cell)) for cell in cells}
    samples: list[dict[str, Any]] = []
    for item in trajectory:
        diagnostics = item.get("costmap_diagnostics") or {}
        freshness = diagnostics.get("freshness")
        anchors = diagnostics.get("anchor_timestamp")
        if freshness is None or anchors is None:
            continue
        now = float(item["simulation_time_s"])
        current_truth = _truth_at(truth_times, truth_layers, now)
        freshness_array = np.asarray(freshness, dtype=float)
        anchor_array = np.asarray(anchors, dtype=float)
        for row in range(freshness_array.shape[0]):
            for column in range(freshness_array.shape[1]):
                if selected is not None and (row, column) not in selected:
                    continue
                anchor_time = float(anchor_array[row, column])
                anchor_truth = float(_truth_at(
                    truth_times, truth_layers, anchor_time
                )[row, column])
                current = float(current_truth[row, column])
                validity = 1.0 - min(abs(current - anchor_truth) / crowd_clip, 1.0)
                estimate = float(freshness_array[row, column])
                samples.append({
                    "simulation_time_s": now,
                    "grid_cell": [row, column],
                    "anchor_timestamp_s": anchor_time,
                    "estimated_freshness": estimate,
                    "ground_truth_validity": validity,
                    "signed_error": estimate - validity,
                    "absolute_error": abs(estimate - validity),
                    "anchor_ground_truth_count": anchor_truth,
                    "current_ground_truth_count": current,
                })
    if not samples:
        return {
            "sample_count": 0, "maximum_absolute_error": 0.0,
            "maximum_overestimation": 0.0, "maximum_underestimation": 0.0,
            "samples": [],
        }
    return {
        "definition": "1-min(abs(N_gt(t)-N_gt(anchor_t))/crowd_clip,1)",
        "sample_count": len(samples),
        "maximum_absolute_error": max(item["absolute_error"] for item in samples),
        "maximum_overestimation": max(max(item["signed_error"], 0.0) for item in samples),
        "maximum_underestimation": max(max(-item["signed_error"], 0.0) for item in samples),
        "samples": samples,
    }


def paired_freshness_metrics(
    correct: list[dict[str, Any]],
    erroneous: list[dict[str, Any]],
    *,
    cells: list[list[int]],
) -> dict[str, Any]:
    """Measure the direct freshness-map effect of an intention intervention."""
    selected = [tuple(map(int, cell)) for cell in cells]
    correct_by_time = {
        round(float(item["simulation_time_s"]), 6): item for item in correct
        if (item.get("costmap_diagnostics") or {}).get("freshness") is not None
    }
    samples = []
    for item in erroneous:
        now = round(float(item["simulation_time_s"]), 6)
        baseline = correct_by_time.get(now)
        diagnostics = item.get("costmap_diagnostics") or {}
        if baseline is None or diagnostics.get("freshness") is None:
            continue
        first = np.asarray(baseline["costmap_diagnostics"]["freshness"], dtype=float)
        second = np.asarray(diagnostics["freshness"], dtype=float)
        differences = [float(second[cell] - first[cell]) for cell in selected]
        samples.append({
            "simulation_time_s": float(item["simulation_time_s"]),
            "cell_signed_differences": {
                f"{cell[0]},{cell[1]}": difference
                for cell, difference in zip(selected, differences)
            },
            "l2": float(np.linalg.norm(differences)),
        })
    return {
        "affected_cells": [list(cell) for cell in selected],
        "sample_count": len(samples),
        "maximum_l2": max((item["l2"] for item in samples), default=0.0),
        "first_difference_time_s": next(
            (item["simulation_time_s"] for item in samples if item["l2"] > 1e-9), None
        ),
        "samples": samples,
    }
