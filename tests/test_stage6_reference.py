from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vacan.mapping.dynamic_costmap import GridGeometry
from vacan.mapping.static_map import OccupancyMap2D
from vacan.reference.comparison import (
    compare_costs, group_candidate_intervals, information_error_candidates,
    trajectory_divergence,
)
from vacan.reference.aggregate import aggregate_runs
from vacan.reference.config import load_stage6_config
from vacan.reference.cost import SharedNavigationCost
from vacan.reference.dataset import write_matched_dataset
from vacan.reference.planner import (
    PedestrianState, ReferenceWaypointController, TimeExpandedAStar,
)
from vacan.robot.stage2 import VelocityLimits


ROOT = Path(__file__).resolve().parents[1]


def _map(obstacles: list[tuple[int, int]] = []) -> OccupancyMap2D:
    geometry = GridGeometry((0.0, 0.0), (4.0, 4.0), 0.2)
    raw = np.zeros(geometry.shape, dtype=np.uint8)
    for cell in obstacles:
        raw[cell] = 1
    return OccupancyMap2D(raw, np.zeros_like(raw, dtype=bool), raw.copy(), geometry,
                          "test", 0.0, "fixture")


def _limits() -> VelocityLimits:
    return VelocityLimits(0.6, 1.0, 0.5, 1.5)


def test_stage6_config_freezes_office_policy_and_no_oracle_future():
    config = load_stage6_config(ROOT / "configs/stage6/office.yaml")
    assert config["scenario"] == "office"
    assert config["policy"] == {
        "inference": "deterministic", "frozen": True, "phase": "partial_coverage_noise"
    }
    assert config["reference"]["uses_ground_truth_current_state"] is True
    assert config["reference"]["uses_actual_future_trajectory"] is False
    assert config["evaluation"]["pedestrian_count"] == 18
    assert config["evaluation"]["validation_seeds"] == [15101]
    assert config["evaluation"]["test_seeds"] == [25101]


def test_time_expanded_planner_avoids_static_and_dynamic_occupancy():
    static_map = _map([(10, 10)])
    planner = TimeExpandedAStar(static_map, _limits(), time_step_s=0.8,
                                horizon_s=30.0, pedestrian_radius_m=0.5,
                                goal_tolerance_m=0.2)
    person = PedestrianState("p0", (2.1, 1.1), (0.0, 0.0))
    plan = planner.plan((0.5, 1.1), 0.0, (3.5, 1.1), [person])
    assert plan.feasible
    assert (10, 10) not in [(state[1], state[2]) for state in plan.grid_states]
    assert all(np.linalg.norm(np.asarray(point) - np.asarray(person.position_xy_m)) >= 0.5
               for point in plan.world_xy_m[1:])
    assert all(b > a for a, b in zip(plan.timestamps_s, plan.timestamps_s[1:]))


def test_planner_reports_infeasible_without_straight_line_fallback():
    static_map = _map([(row, 10) for row in range(20)])
    planner = TimeExpandedAStar(static_map, _limits(), time_step_s=0.8,
                                horizon_s=10.0, pedestrian_radius_m=0.5,
                                goal_tolerance_m=0.2)
    plan = planner.plan((0.5, 1.0), 0.0, (3.5, 1.0), [])
    assert not plan.feasible
    assert plan.world_xy_m == []


def test_reference_controller_respects_velocity_limits():
    controller = ReferenceWaypointController(_limits(), 0.5, 2.0, 0.2)
    controller.set_path([(0.0, 0.0), (0.0, 2.0)])
    linear, angular = controller.action((0.0, 0.0), 0.0)
    assert abs(linear) <= 0.6
    assert abs(angular) <= 1.0


def test_shared_cost_and_delta_j_use_identical_evaluator():
    config = load_stage6_config(ROOT / "configs/stage6/office.yaml")
    assert config["shared_cost"]["objective"] == "trajectory_efficiency"
    evaluator = SharedNavigationCost(config["shared_cost"])
    reference_samples = [
        {"simulation_time_s": 0.0, "position_xy_m": [0.0, 0.0], "action": [0.0, 0.0]},
        {"simulation_time_s": 2.0, "position_xy_m": [1.0, 0.0], "action": [0.5, 0.0]},
    ]
    drl_samples = reference_samples + [
        {"simulation_time_s": 4.0, "position_xy_m": [1.5, 0.0], "action": [0.2, 0.4],
         "collision": True, "nearest_person_distance_m": 0.4},
    ]
    reference = evaluator.evaluate(reference_samples, goal_reached=True)
    drl = evaluator.evaluate(drl_samples, goal_reached=False)
    assert reference.trajectory_efficiency == pytest.approx(
        reference.weighted_time + reference.weighted_length
        + reference.weighted_smoothness
    )
    assert drl.total == pytest.approx(
        drl.trajectory_efficiency + drl.context_penalty + drl.safety_penalty
    )
    comparison = compare_costs(drl, reference, drl_success=False, reference_success=True,
                               tolerance=1e-6)
    assert comparison.delta_j == drl.total - reference.total
    assert comparison.inefficient
    assert comparison.outcome_class == "drl_unsuccessful_reference_success"


def test_divergence_is_timestamp_aligned_and_dataset_is_machine_readable(tmp_path):
    reference = [
        {"simulation_time_s": 0.0, "position_xy_m": [0.0, 0.0]},
        {"simulation_time_s": 2.0, "position_xy_m": [2.0, 0.0]},
    ]
    drl = [{"simulation_time_s": 1.0, "position_xy_m": [1.0, 1.0], "action": [0.2, 0.0]}]
    candidates = trajectory_divergence(drl, reference, distance_threshold_m=0.75,
                                       heading_change_threshold_rad_s=0.5)
    assert candidates[0]["spatial_divergence_m"] == 1.0
    comparison = {"delta_j": 1.0, "component_differences": {"total": 1.0}}
    write_matched_dataset(tmp_path, {"scenario": "office"}, drl, reference, comparison)
    assert json.loads((tmp_path / "metadata.json").read_text())["scenario"] == "office"
    with np.load(tmp_path / "aligned-trajectories.npz") as archive:
        assert archive["drl_xy_m"].shape == (1, 2)


def test_aggregate_deduplicates_repeated_seed(tmp_path):
    def write(name, seed, delta):
        directory = tmp_path / name
        directory.mkdir()
        cost = {"collision_events": 0, "travel_time_s": 2.0, "path_length_m": 1.0,
                "social_violation_integral_s": 0.0, "smoothness": 0.5, "total": 3.0}
        (directory / "summary.json").write_text(json.dumps({
            "status": "pass", "seed": seed, "drl_success": True,
            "reference_success": True, "planning_calls": 4,
            "infeasible_planning_calls": 1,
            "comparison": {"drl_cost": cost, "reference_cost": cost,
                           "delta_j": delta, "inefficient": delta > 0,
                           "outcome_class": "both_successful"},
        }), encoding="utf-8")
    write("matched-seed1-a", 1, 1.0)
    write("matched-seed1-b", 1, 0.0)
    write("matched-seed2-c", 2, -1.0)
    result = aggregate_runs(tmp_path)
    assert result["unique_episode_count"] == 2
    assert result["seeds"] == [1, 2]
    assert result["delta_j"]["mean"] == -0.5
    assert result["reference_infeasible_planning_rate"] == 0.25
    assert len(result["excluded_runs"]) == 1


def test_information_diagnostics_are_candidates_not_causal_labels():
    diagnostics = {"cost_after": [[0.0, 0.0], [0.0, 0.0]],
                   "intention_incoming_count": [[0, 0], [0, 0]]}
    drl = [{"simulation_time_s": 0.0, "action": [0.0, 0.0],
            "costmap_diagnostics": diagnostics, "reward_terms": {}},
           {"simulation_time_s": 0.2, "action": [0.3, 0.6],
            "costmap_diagnostics": diagnostics,
            "reward_terms": {"social": -0.7}}]
    ground_truth = [{"ground_truth_density": [[0, 2], [0, 0]]},
                    {"ground_truth_density": [[0, 2], [0, 0]]}]
    candidates = information_error_candidates(
        drl, ground_truth, crowd_error_threshold=1.0, intention_error_threshold=1.0,
        speed_change_threshold_m_s=0.2, angular_change_threshold_rad_s=0.5,
        local_penalty_threshold=0.5,
    )
    assert "crowd_estimation_error" in candidates[0]["reasons"]
    assert set(candidates[1]["reasons"]) >= {
        "large_speed_change", "large_heading_command_change", "high_local_navigation_penalty"
    }
    assert all("cause" not in candidate for candidate in candidates)
    intervals = group_candidate_intervals(candidates, maximum_gap_s=0.25)
    assert len(intervals) == 1
    assert intervals[0]["sample_count"] == 2
