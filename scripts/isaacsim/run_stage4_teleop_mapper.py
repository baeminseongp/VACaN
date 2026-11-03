#!/usr/bin/env python3
"""Drive Nova Carter manually and fuse virtual depth into 3D/2D maps (no people)."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--gpu", type=int, default=1)
ARGS = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False, "active_gpu": int(ARGS.gpu), "physics_gpu": int(ARGS.gpu),
    "multi_gpu": False, "max_gpu_count": 1, "sync_loads": True, "fast_shutdown": True,
})

import carb
import carb.input
import numpy as np
import omni.appwindow
import omni.client
import omni.usd

import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from vacan.mapping import DepthVoxelIntegrator, project_voxel_map
from vacan.rl import load_stage4_config
from vacan.robot.stage2 import ActionLimiter, load_stage2_config, velocity_limits


def wait_for_loading(timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        if time.monotonic() >= deadline:
            raise TimeoutError("USD asset composition exceeded timeout")
        simulation_app.update()


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=float)


def quaternion_yaw(quaternion_wxyz) -> float:
    w, x, y, z = map(float, quaternion_wxyz)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class KeyboardTeleop:
    def __init__(self) -> None:
        appwindow = omni.appwindow.get_default_app_window()
        if appwindow is None:
            raise RuntimeError("teleop requires an Isaac Sim window; do not use --headless")
        self.interface = carb.input.acquire_input_interface()
        self.keyboard = appwindow.get_keyboard()
        self.pressed: set[str] = set()
        self.finish = False
        self.handle = self.interface.subscribe_to_keyboard_events(self.keyboard, self._event)

    def _event(self, event, *_) -> bool:
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.pressed.add(name)
            if name == "P":
                self.finish = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(name)
        return True

    def command(self, max_linear: float, max_angular: float) -> tuple[float, float]:
        if "SPACE" in self.pressed:
            return 0.0, 0.0
        linear = max_linear * ((1 if "W" in self.pressed else 0) - (1 if "S" in self.pressed else 0))
        angular = max_angular * ((1 if "A" in self.pressed else 0) - (1 if "D" in self.pressed else 0))
        return linear, angular

    def close(self) -> None:
        self.interface.unsubscribe_to_keyboard_events(self.keyboard, self.handle)


def run() -> dict:
    started = time.perf_counter()
    config_path = ARGS.config.resolve()
    config = load_stage4_config(config_path)
    stage1_path = (config_path.parent / config["stage1_config"]).resolve()
    stage2_path = (config_path.parent / config["stage2_config"]).resolve()
    stage1 = yaml.safe_load(stage1_path.read_text(encoding="utf-8"))
    stage2 = load_stage2_config(stage2_path)
    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac Sim asset root is unavailable")
    environment_url = f"{root.rstrip('/')}/{stage1['environment']['asset']}"
    robot_url = f"{root.rstrip('/')}/{stage2['robot']['asset']}"
    for url in (environment_url, robot_url):
        status, _ = omni.client.stat(url)
        if status != omni.client.Result.OK:
            raise FileNotFoundError(f"official asset unavailable: {url}")

    if not omni.usd.get_context().new_stage():
        raise RuntimeError("could not create teleop mapping stage")
    physics_hz = float(config["frequencies"]["physics_hz"])
    dt = 1.0 / physics_hz
    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0, device="cpu")
    environment = sim_utils.UsdFileCfg(usd_path=environment_url)
    environment.func(stage1["environment"]["prim_path"], environment)
    robot_cfg = stage2["robot"]
    robot = world.scene.add(WheeledRobot(
        prim_path=robot_cfg["prim_path"], name="nova_carter",
        wheel_dof_names=robot_cfg["wheel_joints"], create_robot=True, usd_path=robot_url,
        position=np.asarray(robot_cfg["start_xyz"], dtype=float),
        orientation=yaw_quaternion(float(robot_cfg["start_yaw_rad"])),
    ))
    wait_for_loading()
    for _ in range(3):
        simulation_app.update()
    world.reset()
    controller = DifferentialController(
        name="vacan_teleop_differential", wheel_radius=float(robot_cfg["wheel_radius_m"]),
        wheel_base=float(robot_cfg["wheel_base_m"]),
        max_linear_speed=float(stage2["action_limits"]["max_linear_m_s"]),
        max_angular_speed=float(stage2["action_limits"]["max_angular_rad_s"]),
        max_wheel_speed=float(stage2["action_limits"]["max_wheel_rad_s"]),
    )
    limiter = ActionLimiter(velocity_limits(stage2))

    teleop_cfg = config["teleop_mapping"]
    resolution = tuple(map(int, teleop_cfg["depth_resolution"]))
    camera = Camera(prim_path="/World/VACaNMappingDepthCamera", name="vacan_mapping_depth",
                    frequency=int(teleop_cfg["depth_frequency_hz"]), resolution=resolution)
    camera.initialize()
    camera.set_clipping_range(near_distance=0.25,
                              far_distance=float(teleop_cfg["depth_max_range_m"]))
    camera.add_distance_to_image_plane_to_frame()
    camera.add_pointcloud_to_frame(include_unlabelled=True)
    for _ in range(12):
        world.step(render=True)

    integrator = DepthVoxelIntegrator(tuple(config["mapping"]["bounds_xyz_m"]),
                                      float(config["mapping"]["voxel_resolution_m"]))
    keyboard = KeyboardTeleop()
    ARGS.output.mkdir(parents=True, exist_ok=True)
    depth_directory = ARGS.output / "depth_frames"
    if teleop_cfg["save_depth_frames"]:
        depth_directory.mkdir(parents=True, exist_ok=True)
    trajectory = (ARGS.output / "teleop_trajectory.jsonl").open("w", encoding="utf-8")
    capture_every = max(1, round(physics_hz / float(teleop_cfg["depth_frequency_hz"])))
    render_every = max(1, round(physics_hz / float(teleop_cfg["viewport_frequency_hz"])))
    max_steps = round(float(teleop_cfg["max_duration_s"]) * physics_hz)
    camera_offset = np.asarray(teleop_cfg["camera_offset_xyz_m"], dtype=float)
    captured = 0
    previous_position = np.asarray(robot.get_world_pose()[0], dtype=float)
    distance_driven = 0.0
    capture_processing_s: list[float] = []
    print("[Stage 4 teleop] W/S forward/reverse, A/D turn, Space stop, P save and quit", flush=True)
    try:
        for step in range(max_steps):
            if not simulation_app.is_running() or keyboard.finish:
                break
            wall_step_started = time.perf_counter()
            position, orientation = robot.get_world_pose()
            position = np.asarray(position, dtype=float)
            yaw = quaternion_yaw(orientation)
            camera_position = position + np.asarray([
                math.cos(yaw) * camera_offset[0] - math.sin(yaw) * camera_offset[1],
                math.sin(yaw) * camera_offset[0] + math.cos(yaw) * camera_offset[1],
                camera_offset[2],
            ])
            camera.set_world_pose(camera_position, yaw_quaternion(yaw), camera_axes="world")
            requested = keyboard.command(float(stage2["action_limits"]["max_linear_m_s"]),
                                         float(stage2["action_limits"]["max_angular_rad_s"]))
            applied = limiter.apply(requested, dt)
            robot.apply_wheel_actions(controller.forward(np.asarray(applied)))
            render_frame = step % render_every == 0 or step % capture_every == 0
            world.step(render=render_frame)
            distance_driven += float(np.linalg.norm(position[:2] - previous_position[:2]))
            previous_position = position

            if render_frame:
                eye = position - 2.5 * np.asarray([math.cos(yaw), math.sin(yaw), 0]) + np.asarray([0, 0, 1.6])
                target = position + 1.2 * np.asarray([math.cos(yaw), math.sin(yaw), 0]) + np.asarray([0, 0, 0.4])
                set_camera_view(eye=eye.tolist(), target=target.tolist())
            if step % capture_every != 0:
                remaining = dt - (time.perf_counter() - wall_step_started)
                if remaining > 0:
                    time.sleep(remaining)
                continue
            capture_started = time.perf_counter()
            points = np.asarray(camera.get_pointcloud(device="cpu", world_frame=True), dtype=np.float32).reshape(-1, 3)
            valid = np.isfinite(points).all(axis=1)
            points = points[valid]
            accepted = integrator.integrate(
                camera_position, points, float(teleop_cfg["depth_max_range_m"]),
                int(teleop_cfg["depth_ray_stride"]),
            )
            if teleop_cfg["save_depth_frames"]:
                depth = np.asarray(camera.get_depth(device="cpu"), dtype=np.float32)
                np.savez_compressed(depth_directory / f"depth-{captured:06d}.npz", depth_m=depth,
                                    camera_position_xyz_m=camera_position,
                                    camera_orientation_wxyz=yaw_quaternion(yaw),
                                    simulation_time_s=step * dt)
            capture_processing_s.append(time.perf_counter() - capture_started)
            trajectory.write(json.dumps({
                "simulation_time_s": step * dt, "position_xyz_m": position.tolist(),
                "yaw_rad": yaw, "requested_action": list(requested), "applied_action": list(applied),
                "depth_frame": captured, "accepted_depth_rays": accepted,
            }, separators=(",", ":")) + "\n")
            captured += 1
            if captured == 1 or captured % 25 == 0:
                print(f"[Stage 4 teleop] {captured} depth frames, "
                      f"{int(np.count_nonzero(integrator.occupancy == 1))} occupied voxels", flush=True)
            remaining = dt - (time.perf_counter() - wall_step_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        trajectory.close()
        keyboard.close()
        robot.apply_wheel_actions(controller.forward(np.zeros(2)))

    voxel = integrator.voxel_map(config["scenario"], environment_url, {
        "method": "teleoperated Nova Carter virtual depth camera ray fusion",
        "sensor": "isaacsim.sensors.camera.Camera distance_to_image_plane",
        "pedestrian_count": 0, "depth_resolution": list(resolution),
        "depth_frequency_hz": teleop_cfg["depth_frequency_hz"],
        "max_range_m": teleop_cfg["depth_max_range_m"],
        "frames_integrated": integrator.frames_integrated,
        "rays_integrated": integrator.rays_integrated,
    })
    voxel.save(ARGS.output / "voxel_map.npz")
    occupied_indices = np.argwhere(voxel.occupancy == 1)
    occupied_points = np.asarray([voxel.voxel_center(*index) for index in occupied_indices], dtype=np.float32)
    np.savez_compressed(ARGS.output / "depth_pointcloud.npz",
                        points_xyz_m=occupied_points, world_frame=np.asarray("isaac_world_enu"))
    mapping = config["mapping"]
    static2d = project_voxel_map(
        voxel, config["static_grid"], *map(float, mapping["projection_height_m"]),
        float(mapping["robot_radius_m"]) + float(mapping["safety_margin_m"]),
        "voxel_map.npz",
    )
    static2d.save(ARGS.output / "occupancy_2d.npz")
    summary = {
        "status": "pass", "created_utc": datetime.now(UTC).isoformat(),
        "scenario": config["scenario"], "mapping_mode": "teleop_depth", "pedestrian_count": 0,
        "depth_frames": captured, "depth_rays_integrated": integrator.rays_integrated,
        "occupied_voxels": int(np.count_nonzero(voxel.occupancy == 1)),
        "free_voxels": int(np.count_nonzero(voxel.occupancy == 0)),
        "unknown_voxels": int(np.count_nonzero(voxel.occupancy == 2)),
        "voxel_shape_zyx": list(voxel.occupancy.shape),
        "grid_shape_rows_columns": list(static2d.geometry.shape),
        "raw_occupied_cells": int(static2d.raw.sum()),
        "inflated_occupied_cells": int(static2d.inflated.sum()),
        "unknown_cells": int(static2d.unknown.sum()), "distance_driven_m": distance_driven,
        "mean_depth_processing_ms": (1000.0 * float(np.mean(capture_processing_s))
                                      if capture_processing_s else 0.0),
        "p95_depth_processing_ms": (1000.0 * float(np.percentile(capture_processing_s, 95))
                                     if capture_processing_s else 0.0),
        "elapsed_wall_s": time.perf_counter() - started,
        "controls": {"forward": "W", "reverse": "S", "left": "A", "right": "D",
                     "stop": "Space", "save_and_quit": "P"},
    }
    (ARGS.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


try:
    print(json.dumps(run(), indent=2, sort_keys=True))
except Exception as error:
    ARGS.output.mkdir(parents=True, exist_ok=True)
    failure = {"status": "fail", "error_type": type(error).__name__, "error": str(error),
               "traceback": traceback.format_exc(), "created_utc": datetime.now(UTC).isoformat()}
    (ARGS.output / "summary.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
    raise
finally:
    simulation_app.close()
