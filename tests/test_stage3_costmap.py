import copy
from pathlib import Path

import pytest

from vacan.mapping.dynamic_costmap import DynamicCostmap, GridGeometry, load_stage3_config


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


def person(pid, x, y, vx=0.0, vy=0.0):
    return {"id": pid, "position_xyz_m": [x, y, 0.0], "velocity_xyz_m_s": [vx, vy, 0.0]}


def robot(x=0.5, y=1.5, yaw=0.0):
    return {"position_xyz_m": [x, y, 0.0], "yaw_rad": yaw}


@pytest.mark.parametrize("scenario, expected_shape", [
    ("office", (6, 6)), ("hospital", (4, 8)), ("warehouse", (4, 5)),
])
def test_stage3_configs(scenario, expected_shape):
    value = load_stage3_config(Path(f"configs/stage3/{scenario}.yaml"))
    assert value["update_frequency_hz"] == 5.0
    assert value["decay_time_s"] == 300.0
    assert value["grid"]["resolution_m"] == 4.0
    assert DynamicCostmap(value, value["seed"]).geometry.shape == expected_shape


def test_no_cctv_fails_clearly():
    value = config()
    value["cctv"]["regions"] = []
    with pytest.raises(ValueError, match="CCTV"):
        DynamicCostmap(value, 1)


def test_exact_cctv_counts_mean_and_initialization_include_zeros():
    value = config()
    value["cctv"]["regions"] = [[0, 0, 2, 2]]
    manager = DynamicCostmap(value, 1)
    result = manager.tick(0.0, robot(), [person("a", 0.2, 0.2), person("b", 0.3, 0.4)])
    assert sorted(result["diagnostics"]["cctv_counts"].values()) == [0, 0, 0, 2]
    assert result["diagnostics"]["cctv_mean"] == 0.5
    assert manager.current_cost[3][0] == 0.5


def test_user_supplied_cctv_values_average_is_five():
    values = [7, 7, 5, 1]
    assert sum(values) / len(values) == 5


def test_robot_fov_filters_behind_and_out_of_range():
    manager = DynamicCostmap(config(), 1)
    observed = manager._observe(robot(), [
        person("front", 2.0, 1.5), person("behind", 0.0, 1.5), person("far", 8.0, 1.5)
    ])
    assert [item["id"] for item in observed] == ["front"]


def test_measurement_noise_is_reproducible():
    value = config()
    value["robot_observation"]["position_noise_std_m"] = 0.1
    value["robot_observation"]["velocity_noise_std_m_s"] = 0.1
    first, second = DynamicCostmap(value, 99), DynamicCostmap(copy.deepcopy(value), 99)
    assert first._observe(robot(), [person("p", 2, 1.5, 1, 0)]) == \
        second._observe(robot(), [person("p", 2, 1.5, 1, 0)])


def test_robot_observation_replaces_cost_and_records_direct_time():
    manager = DynamicCostmap(config(), 1)
    manager.tick(0.0, robot(), [person("p", 2.2, 1.5, 1.0, 0.0)])
    cell = manager.geometry.world_to_grid(2.2, 1.5)
    assert manager.current_cost[cell[0]][cell[1]] == 1.0
    assert manager.anchor_cost[cell[0]][cell[1]] == 1.0
    assert manager.last_direct_observation_time[cell[0]][cell[1]] == 0.0


def test_required_linear_decay_and_intention_example():
    manager = DynamicCostmap(config(), 1)
    cell = (2, 2)
    manager.cctv_mean = 3.0
    manager.initialized = True
    manager.current_cost[2][2] = manager.anchor_cost[2][2] = 5.0
    manager.anchor_timestamp[2][2] = 0.0
    manager._decay(0.0)
    assert manager.current_cost[2][2] == 5.0
    manager._decay(150.0)
    assert manager.current_cost[2][2] == 4.0
    manager.current_cost[2][2] += 2.0
    manager.anchor_cost[2][2] = manager.current_cost[2][2]
    manager.anchor_timestamp[2][2] = 150.0
    manager._decay(300.0)
    assert manager.current_cost[2][2] == 4.5
    manager._decay(450.0)
    assert manager.current_cost[2][2] == 3.0


def test_dynamic_cctv_mean_changes():
    manager = DynamicCostmap(config(), 1)
    first = manager._ground_truth_counts([])
    manager._update_cctv(first, 0)
    assert manager.cctv_mean == 0
    second = manager._ground_truth_counts([person("p", 0.2, 0.2)])
    manager._update_cctv(second, 1)
    assert manager.cctv_mean == 1


def test_prediction_intention_addition_and_duplicate_prevention():
    manager = DynamicCostmap(config(), 1)
    observation = {"id": "p", "position_xy_m": [1.2, 1.2],
                   "velocity_xy_m_s": [1.0, 0.0], "observed_grid": [1, 1]}
    events = manager._intentions([copy.deepcopy(observation)], 0.0)
    assert events[0]["destination_grid"] == [1, 2]
    assert manager.current_cost[1][2] == 1.0
    assert manager._intentions([copy.deepcopy(observation)], 0.2) == []
    assert manager.current_cost[1][2] == 1.0


def test_transition_hysteresis_requires_confirmation():
    manager = DynamicCostmap(config(), 1)
    initial = {"id": "p", "position_xy_m": [1.2, 1.2], "velocity_xy_m_s": [1, 0],
               "observed_grid": [1, 1]}
    changed = {"id": "p", "position_xy_m": [1.2, 1.2], "velocity_xy_m_s": [2, 0],
               "observed_grid": [1, 1]}
    manager._intentions([initial], 0)
    assert manager._intentions([changed], 1) == []
    assert len(manager._intentions([changed], 2)) == 1


def test_intention_never_overwrites_cctv_and_does_not_change_direct_time():
    manager = DynamicCostmap(config(), 1)
    manager.current_cost[0][0] = 4.0
    observation = {"id": "p", "position_xy_m": [1.2, 0.2], "velocity_xy_m_s": [-1, 0],
                   "observed_grid": [0, 1]}
    assert manager._intentions([observation], 2) == []
    assert manager.current_cost[0][0] == 4.0
    assert manager.last_direct_observation_time[0][0] is None


def test_same_tick_direct_update_precedes_intention_addition():
    manager = DynamicCostmap(config(), 1)
    manager.initialized = True
    manager.cctv_mean = 0
    manager.visibility_mask[1][2] = True
    observation = {"id": "p", "position_xy_m": [2.2, 1.2], "velocity_xy_m_s": [0, 0],
                   "observed_grid": [1, 2]}
    manager._direct_updates([observation], 3)
    manager._intentions([observation], 3)
    assert manager.current_cost[1][2] == 2.0
    assert manager.last_update_type[1][2] == "robot_observation+intention"
    assert manager.anchor_timestamp[1][2] == 3


def test_fixed_frequency_uses_simulation_seconds():
    manager = DynamicCostmap(config(), 1)
    assert manager.tick(0.0, robot(), []) is not None
    assert manager.tick(0.1, robot(), []) is None
    assert manager.tick(0.2, robot(), []) is not None


def test_policy_output_excludes_ground_truth():
    result = DynamicCostmap(config(), 1).tick(0.0, robot(), [person("p", 2, 1.5)])
    assert "ground_truth_count" not in result["policy"]
    assert "ground_truth_count" in result["privileged"]


def test_consistent_world_grid_coordinates():
    geometry = GridGeometry((-2.0, -1.0), (4.0, 2.0), 1.0)
    assert geometry.world_to_grid(-1.5, -0.5) == (0, 0)
    assert geometry.cell_center((0, 0)) == (-1.5, -0.5)
    assert geometry.world_to_grid(2.0, 0.0) is None
