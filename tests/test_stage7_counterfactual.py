from __future__ import annotations

import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from sb3_contrib import RecurrentPPO

from vacan.counterfactual.config import load_stage7_config
from vacan.counterfactual.case_studies import (
    freshness_validity_metrics, paired_freshness_metrics, paired_trajectory_metrics,
)
from vacan.counterfactual.intervention import (
    InterventionManager, InterventionSpec, decision_costmap_intervention, resolve_scope,
)
from vacan.counterfactual.policy import DecisionSnapshot, evaluate_decision_pair
from vacan.mapping.dynamic_costmap import DynamicCostmap, GridGeometry
from vacan.reference.cost import SharedNavigationCost
from vacan.rl.stage5 import recurrent_policy_kwargs
from vacan.rl.stage5_config import load_stage5_config


ROOT = Path(__file__).resolve().parents[1]


def config():
    return {
        "schema_version": 1, "backend": "stage2-state-replay", "scenario": "test",
        "world_frame": "isaac_world_enu", "seed": 7, "update_frequency_hz": 5.0,
        "decay_time_s": 300.0,
        "grid": {"origin_xy": [0.0, 0.0], "size_xy": [4.0, 4.0], "resolution_m": 1.0,
                 "walkable_regions": [[0.0, 0.0, 4.0, 4.0]], "static_obstacle_regions": []},
        "cctv": {"regions": [[0.0, 0.0, 1.0, 1.0]]},
        "robot_observation": {
            "horizontal_fov_deg": 90.0, "max_range_m": 4.0,
            "camera_offset_xy_m": [0.0, 0.0], "camera_yaw_offset_rad": 0.0,
            "position_noise_std_m": 0.0, "velocity_noise_std_m_s": 0.0,
            "geometric_occlusion": False, "minimum_cell_overlap_ratio": 0.2,
            "overlap_samples_per_axis": 5,
        },
        "intention": {"prediction_horizon_s": 1.0, "transition_confirmation_ticks": 2},
    }


def person(pid="p", x=2.2, y=1.5, vx=0.0, vy=0.0):
    return {"id": pid, "position_xyz_m": [x, y, 0.0],
            "velocity_xyz_m_s": [vx, vy, 0.0]}


def robot():
    return {"position_xyz_m": [0.5, 1.5, 0.0], "yaw_rad": 0.0}


def spec(identifier, factor, kind, *, cells=(), destination=None, pedestrian="p",
         value=None, delta=None, convention=None, event_id=None):
    return InterventionSpec(identifier, factor, kind, 0.0, 0.0, tuple(cells), pedestrian,
                            value, delta, destination, convention, None, event_id)


def test_stage7_office_config_has_exactly_two_factors_and_frozen_policy():
    value = load_stage7_config(ROOT / "configs/stage7/office.yaml")
    assert value["scenario"] == "office"
    assert value["policy"]["frozen"] is True
    assert {item.factor for item in value["parsed_interventions"]} == {"congestion", "intention"}
    assert {"case_1", "case_2", "case_3"} <= set(value["case_studies"])


def test_scope_supports_cells_rectangle_and_polygon():
    geometry = GridGeometry((0, 0), (4, 4), 1)
    cells = resolve_scope({"cells": [[3, 3]], "rectangle_xy": [0, 0, 2, 1],
                           "polygon_xy": [[2, 2], [4, 2], [4, 4], [2, 4]]}, geometry)
    assert (3, 3) in cells and (0, 0) in cells and (2, 2) in cells


def test_no_op_manager_reproduces_stage3_costmap_exactly():
    base = DynamicCostmap(config(), 11)
    probe = DynamicCostmap(copy.deepcopy(config()), 11)
    manager = InterventionManager([spec("noop", "congestion", "no_op")],
                                  probe.geometry, probe.cctv_cells)
    probe.intervention_manager = manager
    first = base.tick(0.0, robot(), [person(vx=0.2)])
    second = probe.tick(0.0, robot(), [person(vx=0.2)])
    assert first == second
    assert manager.logs == []


@pytest.mark.parametrize("kind,original,value,delta,expected", [
    ("overestimate", 2, None, 3, 5),
    ("underestimate", 2, None, 3, 0),
    ("overestimate", 2, 7, None, 7),
])
def test_congestion_over_and_under_estimation(kind, original, value, delta, expected):
    geometry = GridGeometry((0, 0), (4, 4), 1)
    item = spec("c", "congestion", kind, cells=((1, 2),), value=value, delta=delta)
    manager = InterventionManager([item], geometry, {(0, 0)})
    observations = [{"id": "a", "observed_grid": [1, 2]},
                    {"id": "b", "observed_grid": [1, 2]}]
    visible = [[True] * 4 for _ in range(4)]
    overrides, logs = manager.modify_congestion(0, observations, visible, [[0] * 4 for _ in range(4)])
    assert overrides[(1, 2)] == expected
    assert logs[0]["original_observed_count"] == original


def test_ground_truth_correction_is_explicitly_full_grid_and_resets_anchor():
    value = config()
    geometry = GridGeometry((0, 0), (4, 4), 1)
    item = spec("gt", "congestion", "ground_truth_correction", cells=((1, 2),),
                convention="full_grid_count")
    manager = InterventionManager([item], geometry, {(0, 0)})
    costmap = DynamicCostmap(value, 1, manager)
    result = costmap.tick(0.0, robot(), [person(), person("q", 2.3, 1.4)])
    assert result["diagnostics"]["intervention_events"][0]["measurement_convention"] == "full_grid_count"
    assert costmap.anchor_timestamp[1][2] == 0.0


def test_full_information_congestion_replaces_every_non_cctv_cell_even_outside_fov():
    geometry = GridGeometry((0, 0), (4, 4), 1)
    item = spec("full", "congestion", "full_information")
    manager = InterventionManager([item], geometry, {(0, 0)})
    truth = [[0] * 4 for _ in range(4)]
    truth[3][3] = 2
    invisible = [[False] * 4 for _ in range(4)]
    overrides, logs = manager.modify_congestion(0, [], invisible, truth)
    assert (0, 0) not in overrides
    assert overrides[(3, 3)] == 2
    assert len(overrides) == 15
    assert logs[0]["observation_source"] == "privileged_full_grid_count"


def test_full_information_intention_uses_controller_destination_and_deduplicates():
    geometry = GridGeometry((0, 0), (4, 4), 1)
    item = spec("full", "intention", "full_information")
    manager = InterventionManager([item], geometry, {(0, 0)})
    pedestrian = person()
    pedestrian["current_destination_xyz_m"] = [3.2, 2.2, 0.0]
    events, logs = manager.modify_intentions(0, [], [], [pedestrian])
    assert events[0]["destination_grid"] == [2, 3]
    assert events[0]["source"] == "pegasus_controller_destination"
    assert logs[0]["ground_truth_reference"] == "pegasus_controller_current_destination"
    assert manager.modify_intentions(0.2, [], [], [pedestrian]) == ([], [])


def test_intervention_cannot_overwrite_cctv():
    geometry = GridGeometry((0, 0), (4, 4), 1)
    with pytest.raises(ValueError, match="CCTV"):
        InterventionManager([spec("bad", "congestion", "overestimate", cells=((0, 0),))],
                            geometry, {(0, 0)})


def test_explicit_counterfactual_cctv_corruption_is_isolated_from_ground_truth():
    geometry = GridGeometry((0, 0), (4, 4), 1)
    item = InterventionSpec(
        "cctv", "congestion", "cctv_corruption", 0, 1,
        cells=((0, 0),), value=5,
        measurement_convention="counterfactual_cctv_corruption",
    )
    manager = InterventionManager([item], geometry, {(0, 0)})
    exact = [[0] * 4 for _ in range(4)]
    exact[0][0] = 1
    visible, logs = manager.modify_cctv(0, exact)
    assert exact[0][0] == 1
    assert visible[0][0] == 5
    assert logs[0]["exact_cctv_count"] == 1
    assert logs[0]["base_research_assumption_preserved"] is True


def test_cctv_corruption_requires_cctv_only_scope_and_explicit_label():
    geometry = GridGeometry((0, 0), (4, 4), 1)
    with pytest.raises(ValueError, match="only CCTV"):
        InterventionManager([
            InterventionSpec(
                "bad", "congestion", "cctv_corruption", 0, 1,
                cells=((1, 1),), value=3,
                measurement_convention="counterfactual_cctv_corruption",
            )
        ], geometry, {(0, 0)})
    with pytest.raises(ValueError, match="explicitly labeled"):
        InterventionManager([
            InterventionSpec(
                "bad", "congestion", "cctv_corruption", 0, 1,
                cells=((0, 0),), value=3,
            )
        ], geometry, {(0, 0)})


def test_intention_replacement_missing_and_spurious_are_event_aware():
    geometry = GridGeometry((0, 0), (4, 4), 1)
    original = [{"event_id": "p:1,2:0", "pedestrian_id": "p",
                 "destination_grid": [1, 2], "contribution": 1}]
    observation = [{"id": "p", "position_xy_m": [1.2, 1.2],
                    "velocity_xy_m_s": [1, 0]}]
    replace = InterventionManager([
        spec("replace", "intention", "incorrect_destination", destination=(2, 2))
    ], geometry, {(0, 0)})
    changed, logs = replace.modify_intentions(0, original, observation)
    assert changed[0]["destination_grid"] == [2, 2]
    assert logs[0]["resulting_costmap_delta_by_cell"] == {"1,2": -1, "2,2": 1}

    missing = InterventionManager([spec("missing", "intention", "missing")], geometry, {(0, 0)})
    assert missing.modify_intentions(0, original, observation)[0] == []

    spurious_spec = spec("spurious", "intention", "spurious", destination=(3, 3),
                         event_id="injected-1")
    spurious = InterventionManager([spurious_spec], geometry, {(0, 0)})
    assert len(spurious.modify_intentions(0, [], observation)[0]) == 1
    assert spurious.modify_intentions(0, [], observation)[0] == []


def test_decision_no_op_does_not_mutate_original():
    cost = [[0.0, 1.0], [2.0, 3.0]]
    item = InterventionSpec("noop", "congestion", "no_op", 0, 0)
    modified, logs = decision_costmap_intervention(cost, {}, [[0, 0], [0, 0]], item)
    assert modified == cost and modified is not cost and logs == []


def test_decision_congestion_reconstruction_preserves_later_intention_addition():
    # Final 6 = direct observation 4 followed by two intention additions.
    cost = [[0.0] * 4 for _ in range(4)]
    cost[1][2] = 6.0
    diagnostics = {"observations": [
        {"id": str(index), "observed_grid": [1, 2]} for index in range(4)
    ]}
    truth = [[0] * 4 for _ in range(4)]
    truth[1][2] = 3
    item = spec("gt", "congestion", "ground_truth_correction", cells=((1, 2),),
                convention="full_grid_count")
    modified, _ = decision_costmap_intervention(cost, diagnostics, truth, item)
    assert modified[1][2] == 5.0


def test_case_study_trajectory_and_freshness_metrics_are_timestamp_aligned():
    correct = [
        {"simulation_time_s": 0.0, "position_xy_m": [0, 0], "action": [0, 0]},
        {"simulation_time_s": 1.0, "position_xy_m": [1, 0], "action": [1, 0],
         "goal_reached": True},
    ]
    error = [
        {"simulation_time_s": 0.0, "position_xy_m": [0, 0], "action": [0, 0]},
        {"simulation_time_s": 0.5, "position_xy_m": [0, 1], "action": [0, 1]},
        {"simulation_time_s": 1.0, "position_xy_m": [0, 2], "action": [0, 1],
         "goal_reached": False},
    ]
    metrics = paired_trajectory_metrics(
        correct, error, trajectory_threshold_m=0.75, action_threshold_l2=0.05
    )
    assert metrics["path_difference_observed"] is True
    assert metrics["first_spatial_divergence_time_s"] == 0.5
    assert metrics["outcome_changed"] is True

    def record(time, freshness):
        return {
            "simulation_time_s": time, "position_xy_m": [0, 0], "action": [0, 0],
            "costmap_diagnostics": {
                "freshness": [[freshness, 1.0], [1.0, 1.0]],
                "anchor_timestamp": [[0.0, 0.0], [0.0, 0.0]],
            },
        }
    correct_fresh = [record(0.0, 1.0), record(1.0, 0.5)]
    error_fresh = [record(0.0, 1.0), record(1.0, 1.0)]
    truth = [
        {"simulation_time_s": 0.0, "ground_truth_density": [[0, 0], [0, 0]]},
        {"simulation_time_s": 1.0, "ground_truth_density": [[2, 0], [0, 0]]},
    ]
    validity = freshness_validity_metrics(
        error_fresh, truth, crowd_clip=2.0, cells=[[0, 0]]
    )
    assert validity["maximum_overestimation"] == 1.0
    paired = paired_freshness_metrics(
        correct_fresh, error_fresh, cells=[[0, 0]]
    )
    assert paired["maximum_l2"] == 0.5
    assert paired["first_difference_time_s"] == 1.0


class TinyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "map": gym.spaces.Box(0, 1, (2, 61, 61), np.float32),
            "robot": gym.spaces.Box(-1, 1, (8,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        return {"map": np.zeros((2, 61, 61), np.float32),
                "robot": np.zeros(8, np.float32)}, {}

    def step(self, action):
        return self.reset()[0], 0.0, False, False, {}


def test_frozen_policy_no_op_has_identical_action_features_and_value():
    env = TinyEnv()
    training = load_stage5_config(ROOT / "configs/stage5/recurrent_ppo.yaml")
    model = RecurrentPPO("MultiInputLstmPolicy", env, n_steps=8, batch_size=4, n_epochs=1,
                         policy_kwargs=recurrent_policy_kwargs(training), seed=31,
                         device="cpu", verbose=0)
    observation, _ = env.reset()
    snapshot = DecisionSnapshot(0.0, observation, None, True)
    result, arrays = evaluate_decision_pair(model, snapshot, copy.deepcopy(observation))
    assert result["differences"]["action_l2"] == 0
    assert result["differences"]["critic_value"] == 0
    assert result["differences"]["actor_hidden_l2"] == 0
    assert all(item["l2"] == 0 for item in result["differences"]["features"].values())
    assert arrays
    assert result["policy_parameter_sha256_before"] == result["policy_parameter_sha256_after"]


def test_stage7_reuses_stage6_cost_and_delta_sign():
    evaluator = SharedNavigationCost({
        "weights": {"time": 1, "length": 2, "collision": 50, "social": 10,
                    "smoothness": 0.1, "terminal_failure": 1},
        "personal_space_m": 0.8, "terminal_failure_penalty": 120,
        "normalization": {"max_linear_m_s": 0.6, "max_angular_rad_s": 1.0},
    })
    original = [{"simulation_time_s": 0, "position_xy_m": [0, 0], "action": [0, 0]},
                {"simulation_time_s": 1, "position_xy_m": [1, 0], "action": [0.5, 0]}]
    counterfactual = original + [
        {"simulation_time_s": 2, "position_xy_m": [2, 0], "action": [0.5, 0]}
    ]
    first = evaluator.evaluate(original, goal_reached=True)
    second = evaluator.evaluate(counterfactual, goal_reached=True)
    assert second.total - first.total > 0
