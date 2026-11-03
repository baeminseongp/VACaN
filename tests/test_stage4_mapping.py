import json
from pathlib import Path

import numpy as np
import pytest

from vacan.mapping import (
    DepthVoxelIntegrator, GridGeometry, OccupancyMap2D, VoxelMap,
    inflate_obstacles, project_voxel_map,
)
from vacan.rl import (
    NavigationReward, NormalizedActionAdapter, ObservationBuilder, ObservationSpec,
    astar_path, load_stage4_config,
)
from vacan.robot.stage2 import VelocityLimits


@pytest.mark.parametrize("scenario,crowd_shape,static_shape", [
    ("office", (6, 6), (120, 120)),
    ("hospital", (4, 8), (80, 160)),
    ("warehouse", (4, 5), (80, 100)),
])
def test_stage4_config_separates_crowd_and_static_grids(scenario, crowd_shape, static_shape):
    config = load_stage4_config(Path(f"configs/stage4/{scenario}.yaml"))
    assert config["shared_grid"].shape == crowd_shape
    assert config["shared_grid"].resolution_m == 4.0
    assert config["static_grid"].shape == static_shape
    assert config["static_grid"].resolution_m == 0.2
    assert config["observation"]["crop_rows"] == 61
    assert config["frequencies"]["physics_hz"] == 60.0
    assert config["frequencies"]["costmap_hz"] == 5.0
    assert config["teleop_mapping"]["depth_resolution"] == [160, 90]
    assert config["teleop_mapping"]["depth_frequency_hz"] == 2.0
    assert config["teleop_mapping"]["viewport_frequency_hz"] == 20.0
    assert config["teleop_mapping"]["save_depth_frames"] is False


def test_voxel_coordinates_export_and_deterministic_load(tmp_path):
    occupancy = np.zeros((2, 3, 4), dtype=np.uint8)
    occupancy[1, 2, 3] = 1
    voxel = VoxelMap(occupancy, (-1.0, -2.0, 0.1), 0.2, "test", "asset.usd",
                     {"collision_prims": ["/World/wall"]})
    assert voxel.voxel_center(1, 2, 3) == pytest.approx((-0.3, -1.5, 0.4))
    path = tmp_path / "voxel.npz"
    voxel.save(path)
    loaded = VoxelMap.load(path)
    assert np.array_equal(loaded.occupancy, occupancy)
    assert loaded.bounds_xyz_m == pytest.approx((-1, -2, 0.1, -0.2, -1.4, 0.5))
    assert json.loads(path.with_suffix(".json").read_text())["axis_order"] == "zyx"


def test_projection_height_filter_and_static_dynamic_alignment(tmp_path):
    occupancy = np.zeros((3, 2, 2), dtype=np.uint8)
    occupancy[0, 0, 0] = 1  # below collision-relevant body interval
    occupancy[2, 1, 1] = 1
    voxel = VoxelMap(occupancy, (0, 0, 0), 1.0, "test", "asset", {})
    geometry = GridGeometry((0, 0), (2, 2), 1.0)
    result = project_voxel_map(voxel, geometry, 1.0, 3.0, 0.0, "voxel.npz")
    assert result.raw.tolist() == [[0, 0], [0, 1]]
    assert result.geometry is geometry
    path = tmp_path / "map.npz"
    result.save(path)
    assert np.array_equal(OccupancyMap2D.load(path).raw, result.raw)


def test_projection_known_free_takes_precedence_over_unknown_subvoxels():
    occupancy = np.full((2, 1, 1), 2, dtype=np.uint8)
    occupancy[0, 0, 0] = 0
    voxel = VoxelMap(occupancy, (0, 0, 0), 1.0, "test", "asset", {})
    result = project_voxel_map(voxel, GridGeometry((0, 0), (1, 1), 1),
                               0.0, 2.0, 0.0, "voxel.npz")
    assert not result.unknown[0, 0]
    assert result.policy_obstacle[0, 0] == 0


def test_inflation_and_unknown_are_unsafe():
    raw = np.zeros((5, 5), dtype=np.uint8)
    raw[2, 2] = 1
    inflated = inflate_obstacles(raw, 1.0, 1.01)
    assert inflated.sum() == 5
    static = OccupancyMap2D(raw, np.eye(5, dtype=bool), inflated,
                            GridGeometry((0, 0), (5, 5), 1), "test", 1.01, "voxel")
    assert np.all(static.policy_obstacle.diagonal() == 1)


def make_static():
    geometry = GridGeometry((0, 0), (4, 4), 1)
    raw = np.zeros((4, 4), dtype=np.uint8)
    return OccupancyMap2D(raw, np.zeros_like(raw, dtype=bool), raw.copy(), geometry,
                          "test", 0.5, "voxel")


def test_two_channel_robot_centered_observation_is_finite_and_no_gt():
    spec = ObservationSpec(5, 5, False, 4.0, 10.0, 1.0, 2.0)
    builder = ObservationBuilder(make_static(), spec)
    robot = {"position_xyz_m": [0.5, 0.5, 0], "yaw_rad": 0,
             "linear_velocity_xyz_m_s": [0.2, 0, 0],
             "angular_velocity_xyz_rad_s": [0, 0, 0.4]}
    observation = builder.build(np.ones((4, 4), dtype=float), robot, (3.5, 3.5), (0, 0))
    assert observation["map"].shape == (2, 5, 5)
    assert observation["robot"].shape == (8,)
    assert observation["map"].dtype == observation["robot"].dtype == np.float32
    assert np.isfinite(observation["map"]).all() and np.isfinite(observation["robot"]).all()
    assert observation["map"][0, 0, 0] == 1  # outside map is unsafe, not free
    assert set(observation) == {"map", "robot"}  # no freshness or privileged GT channel


def test_coarse_crowd_grid_is_resampled_onto_fine_static_crop():
    static = make_static()
    dynamic_geometry = GridGeometry((0, 0), (4, 4), 2.0)
    builder = ObservationBuilder(
        static, ObservationSpec(3, 3, False, 4.0, 10.0, 1.0, 2.0),
        dynamic_geometry=dynamic_geometry,
    )
    robot = {"position_xyz_m": [1.5, 1.5, 0], "yaw_rad": 0,
             "linear_velocity_xyz_m_s": [0, 0, 0],
             "angular_velocity_xyz_rad_s": [0, 0, 0]}
    crowd = np.asarray([[4.0, 0.0], [0.0, 0.0]])
    observation = builder.build(crowd, robot, (3, 3), (0, 0))
    assert observation["map"].shape == (2, 3, 3)
    assert observation["map"][1, 1, 1] == 1.0


def test_normalized_action_limits_velocity_and_acceleration():
    limits = VelocityLimits(1.0, 2.0, 0.5, 1.0)
    adapter = NormalizedActionAdapter(limits, 0.2)
    assert adapter.apply(np.asarray([2.0, -2.0])) == pytest.approx((0.1, -0.2))
    adapter.reset()
    assert adapter.apply(np.asarray([0.0, 0.0])) == (0.0, 0.0)


def test_reward_components_termination_inputs_are_separate():
    reward = NavigationReward({"progress": 2, "goal": 10, "collision": 5, "social": 1,
                               "time": 0.1, "smooth": 0.5, "personal_space_m": 1})
    total, terms = reward.compute(5, 4, True, False, 0.5, (0.2, 0.1), (0, 0))
    assert set(terms) == {"progress", "goal", "collision", "social", "time", "smooth"}
    assert total == pytest.approx(sum(terms.values()))
    assert terms["progress"] == 2 and terms["goal"] == 10 and terms["social"] == -0.5


def test_astar_baseline_avoids_static_obstacles():
    obstacles = np.zeros((5, 5), dtype=np.uint8)
    obstacles[2, :4] = 1
    path = astar_path(obstacles, (0, 0), (4, 0))
    assert path[0] == (0, 0) and path[-1] == (4, 0)
    assert all(not obstacles[cell] for cell in path)


def test_depth_rays_carve_free_and_preserve_occupied_endpoints():
    integrator = DepthVoxelIntegrator((0, 0, 0, 4, 2, 2), 1.0)
    accepted = integrator.integrate(np.asarray([0.5, 0.5, 0.5]),
                                    np.asarray([[3.5, 0.5, 0.5]]), 5.0)
    assert accepted == 1
    assert integrator.occupancy[0, 0, 0] == 0
    assert integrator.occupancy[0, 0, 3] == 1
    assert integrator.occupancy[1, 1, 3] == 2
    # A later free ray must not erase an already occupied voxel.
    integrator.integrate(np.asarray([0.5, 0.5, 0.5]),
                         np.asarray([[3.5, 1.5, 0.5]]), 5.0)
    assert integrator.occupancy[0, 0, 3] == 1
