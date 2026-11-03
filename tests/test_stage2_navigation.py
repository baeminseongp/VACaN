from pathlib import Path

import pytest

from vacan.robot.stage2 import ActionLimiter, VelocityLimits, WaypointFollower, load_stage2_config


@pytest.mark.parametrize("scenario", ["office", "hospital", "warehouse"])
def test_stage2_configs_use_verified_nova_carter(scenario):
    config = load_stage2_config(Path(f"configs/stage2/{scenario}.yaml"))
    assert config["robot"]["wheel_radius_m"] == pytest.approx(0.14)
    assert config["robot"]["wheel_base_m"] == pytest.approx(0.4132)
    assert config["robot"]["contact_scope"] == "all_rigid_bodies"
    assert config["robot"]["start_xyz"][2] == pytest.approx(0.25)
    assert Path("configs/stage2", config["stage1_config"]).resolve().is_file()
    assert Path("configs/stage2", config["pedestrian_routes"]).resolve().is_file()
    assert config["episode"]["collision_termination"] is True
    assert config["episode"]["timeout"] is True
    assert config["episode"]["reset_settle_steps"] == 30


def test_action_limiter_clips_velocity_and_acceleration():
    limiter = ActionLimiter(VelocityLimits(0.6, 1.0, 0.5, 1.5))
    assert limiter.apply((3.0, -4.0), 0.1) == pytest.approx((0.05, -0.15))
    assert limiter.apply((3.0, -4.0), 0.1) == pytest.approx((0.10, -0.30))
    limiter.reset()
    assert limiter.previous == (0.0, 0.0)


def test_waypoint_follower_rotates_before_forward_motion():
    controller = WaypointFollower([(0.0, 2.0)], 0.5, 2.0, 0.2, 1.0)
    linear, angular = controller.action(0.0, 0.0, 0.0)
    assert linear == 0.0
    assert angular > 0.0
    linear, angular = controller.action(0.0, 0.0, 1.55)
    assert linear > 0.45
    assert abs(angular) < 0.1
