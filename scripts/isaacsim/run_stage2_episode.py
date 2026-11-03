#!/usr/bin/env python3
"""Run one physical Nova Carter + Pegasus Stage 2 navigation episode."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--headless", action="store_true")
# Stage 5 reuses the validated physical environment in-process and supplies
# additional training arguments. Unknown arguments remain owned by that caller.
ARGS, _UNKNOWN_ARGS = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": bool(ARGS.headless), "active_gpu": int(ARGS.gpu),
        "physics_gpu": int(ARGS.gpu), "multi_gpu": False, "max_gpu_count": 1,
        "sync_loads": True, "fast_shutdown": True,
    }
)

import carb
import numpy as np
import omni.anim.graph.core as animation_graph_core
import omni.client
import omni.timeline
import omni.usd
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.sensors import RigidContactView
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from vacan.robot.stage2 import ActionLimiter, WaypointFollower, load_stage2_config, velocity_limits, wrap_angle


def vector3(value) -> np.ndarray:
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def quaternion_yaw(quaternion_wxyz) -> float:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wait_for_asset_composition(timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    context = omni.usd.get_context()
    while context.get_stage_loading_status()[2] > 0:
        if time.monotonic() >= deadline:
            raise TimeoutError("USD asset composition exceeded timeout")
        simulation_app.update()


def require_asset(path: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    result = None
    while time.monotonic() < deadline:
        result, _ = omni.client.stat(path)
        if result == omni.client.Result.OK:
            return
        simulation_app.update()
        time.sleep(0.25)
    raise FileNotFoundError(f"Official asset unavailable: {path} ({result})")


class ResolvedRouteController(PersonController):
    """Replay a Stage 1 obstacle-validated pedestrian route in fixed time."""

    instances: list["ResolvedRouteController"] = []

    def __init__(self, pedestrian_id: str, speed: float, cached: dict) -> None:
        super().__init__()
        self.pedestrian_id = pedestrian_id
        self.speed = float(speed)
        self.spawn = vector3(cached["spawn_xyz_m"])
        self.destinations = [vector3(point) for point in cached["destinations_xyz_m"]]
        self.route_offset = int(cached["route_offset"])
        self.destination_index = self.route_offset
        self.initial_path = [vector3(point) for point in cached["initial_path_xyz_m"]]
        self.cycle_paths = {
            int(index): [vector3(point) for point in points]
            for index, points in cached["cycle_paths_xyz_m"].items()
        }
        self.path_points: list[np.ndarray] = []
        self.position = self.spawn.copy()
        self.previous_position = self.position.copy()
        self.used_initial = False
        self.commanded_target = self.spawn.copy()
        self.instances.append(self)

    @property
    def current_destination(self) -> np.ndarray:
        return self.destinations[self.destination_index]

    def _plan(self) -> None:
        points = self.initial_path if not self.used_initial else self.cycle_paths[self.destination_index]
        self.used_initial = True
        self.path_points = [point.copy() for point in (points[1:] if len(points) > 1 else points)]

    def _yield_scale(self) -> float:
        location = np.round(self.position[:2] * 10.0) / 10.0
        for other in self.instances:
            if other is self or self.pedestrian_id < other.pedestrian_id:
                continue
            other_location = np.round(other.position[:2] * 10.0) / 10.0
            if np.linalg.norm(location - other_location) < 0.9:
                return 0.5
        return 1.0

    def update(self, dt: float) -> None:
        if not self.path_points:
            self._plan()
        while self.path_points and np.linalg.norm(self.path_points[0][:2] - self.position[:2]) < 0.25:
            self.path_points.pop(0)
        if not self.path_points:
            self.destination_index = (self.destination_index + 1) % len(self.destinations)
            self._plan()
        target = self.path_points[0]
        delta = target - self.position
        distance = float(np.linalg.norm(delta[:2]))
        self.previous_position = self.position.copy()
        if distance > 1e-9:
            step = min(self.speed * self._yield_scale() * float(dt), distance)
            self.position = self.position + delta / distance * step
        self.commanded_target = target.copy()
        self._person.update_target_position(target.tolist(), walk_speed=self.speed)

    def enforce_pose(self) -> None:
        if self._person.character_graph is None:
            return
        delta = self.position - self.previous_position
        yaw = 0.0 if np.linalg.norm(delta[:2]) < 1e-9 else math.atan2(float(delta[1]), float(delta[0]))
        orientation_xyzw = np.asarray([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
        self._person.character_graph.set_world_transform(
            carb.Float3(*self.position.tolist()), carb.Float4(*orientation_xyzw.tolist())
        )
        self._person.state.position = self.position.copy()
        self._person.state.orientation = orientation_xyzw.copy()

    def reset_route(self) -> None:
        self.destination_index = self.route_offset
        self.path_points.clear()
        self.position = self.spawn.copy()
        self.previous_position = self.spawn.copy()
        self.used_initial = False
        self.enforce_pose()


class NovaCarterNavigationEnv:
    """Minimal reset/step interface backed by Isaac physics."""

    def __init__(self, config: dict, output: Path, seed: int) -> None:
        self.config = config
        self.output = output
        self.seed = seed
        self.dt = float(config["episode"]["physics_dt_s"])
        self.step_index = 0
        self.previous_goal_distance = 0.0
        self.path_length = 0.0
        self.previous_position = None
        self.collision_pairs: set[tuple[str, str]] = set()
        self.output.mkdir(parents=True, exist_ok=True)
        self.stream = None
        self._build()

    def _resolve_relative(self, value: str) -> Path:
        return (ARGS.config.resolve().parent / value).resolve()

    def _build(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        ResolvedRouteController.instances.clear()
        root = get_assets_root_path()
        if not root:
            raise RuntimeError("Isaac Sim asset root is unavailable")
        stage1_path = self._resolve_relative(self.config["stage1_config"])
        self.stage1 = yaml.safe_load(stage1_path.read_text(encoding="utf-8"))
        route_path = self._resolve_relative(self.config["pedestrian_routes"])
        routes = json.loads(route_path.read_text(encoding="utf-8"))
        environment_url = f"{root.rstrip('/')}/{self.stage1['environment']['asset']}"
        robot_url = f"{root.rstrip('/')}/{self.config['robot']['asset']}"
        require_asset(environment_url)
        require_asset(robot_url)
        if not omni.usd.get_context().new_stage():
            raise RuntimeError("Could not create Stage 2 stage")
        simulation_app.update()
        pg = PegasusInterface()
        pg._world_settings = {
            "physics_dt": self.dt,
            "rendering_dt": float(self.config["episode"]["rendering_dt_s"]),
            "stage_units_in_meters": 1.0, "device": "cpu",
        }
        pg._world = World(**pg._world_settings)
        self.world = pg.world
        spawn = sim_utils.UsdFileCfg(usd_path=environment_url)
        spawn.func(self.stage1["environment"]["prim_path"], spawn)
        start = vector3(self.config["robot"]["start_xyz"])
        orientation = yaw_quaternion(float(self.config["robot"]["start_yaw_rad"]))
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path=self.config["robot"]["prim_path"], name="nova_carter",
                wheel_dof_names=self.config["robot"]["wheel_joints"], create_robot=True,
                usd_path=robot_url, position=start, orientation=orientation,
            )
        )
        probe = self.config.get("collision_probe", {})
        self.probe = None
        if probe.get("enabled"):
            self.probe = self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/Stage2CollisionProbe", name="stage2_collision_probe",
                    position=vector3(probe["position_xyz"]), scale=vector3(probe["size_xyz"]),
                    color=np.asarray([0.8, 0.1, 0.1]),
                )
            )
        self.controllers: list[ResolvedRouteController] = []
        self.people = []
        for character in sorted({person["character"] for person in self.stage1["pedestrians"]}):
            require_asset(f"{Person.assets_root_path.rstrip('/')}/{character}")
        for pedestrian in self.stage1["pedestrians"]:
            cached = routes["pedestrians"][pedestrian["id"]]
            controller = ResolvedRouteController(
                pedestrian["id"], pedestrian["speed_m_s"], cached
            )
            person = Person(
                pedestrian["id"], pedestrian["character"],
                init_pos=controller.spawn.tolist(), controller=controller,
            )
            self.controllers.append(controller)
            self.people.append((pedestrian, person, controller))
        wait_for_asset_composition()
        for _ in range(3):
            simulation_app.update()
        # RigidContactView must attach PhysX ContactReporter APIs before the
        # first hard reset creates the physics scene. Creating it afterward
        # yields an empty tensor contact view in Isaac Sim 5.1.
        robot_root = self.world.stage.GetPrimAtPath(self.config["robot"]["prim_path"])
        self.contact_body_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(robot_root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not self.contact_body_paths:
            raise RuntimeError("Nova Carter has no rigid bodies for contact monitoring")
        self.contact_view = RigidContactView(
            prim_paths_expr=self.contact_body_paths,
            filter_paths_expr=[[] for _ in self.contact_body_paths],
            name="nova_carter_all_body_contacts",
            prepare_contact_sensors=True,
        )
        self.world.reset()
        self.contact_view.initialize(self.world.physics_sim_view)
        if not self.contact_view.is_physics_handle_valid() or self.contact_view.num_shapes < 1:
            raise RuntimeError(
                f"No PhysX contact bodies matched {self.contact_body_paths}"
            )
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        for _ in range(8):
            self.world.step(render=True)
            for controller in self.controllers:
                controller.enforce_pose()
        self._update_viewer_camera()
        for _, person, _ in self.people:
            person.character_graph = animation_graph_core.get_character(person.character_skel_root_stage_path)
            if person.character_graph is None:
                raise RuntimeError(f"No live animation graph for {person._stage_prefix}")
        self.diff_controller = DifferentialController(
            name="vacan_nova_carter_differential",
            wheel_radius=float(self.config["robot"]["wheel_radius_m"]),
            wheel_base=float(self.config["robot"]["wheel_base_m"]),
            max_linear_speed=float(self.config["action_limits"]["max_linear_m_s"]),
            max_angular_speed=float(self.config["action_limits"]["max_angular_rad_s"]),
            max_wheel_speed=float(self.config["action_limits"]["max_wheel_rad_s"]),
        )
        self.limiter = ActionLimiter(velocity_limits(self.config))
        baseline = self.config["baseline"]
        self.baseline = WaypointFollower(
            [tuple(point) for point in self.config["robot"]["waypoints_xy"]],
            float(baseline["cruise_speed_m_s"]), float(baseline["angular_gain"]),
            float(baseline["waypoint_tolerance_m"]), float(baseline["forward_heading_limit_rad"]),
        )
        self.articulation = self._inspect_robot(robot_url)
        self.environment_url = environment_url
        self.stage1_config_path = str(stage1_path)
        self.route_cache_path = str(route_path)

    def _inspect_robot(self, robot_url: str) -> dict:
        stage = self.world.stage
        root_prim = stage.GetPrimAtPath(self.config["robot"]["prim_path"])
        articulation_roots, collision_prims, rigid_bodies = [], [], []
        for prim in Usd.PrimRange(root_prim):
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_prims.append(path)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(path)
        dof_names = list(self.robot.dof_names)
        wheel_indices = [self.robot.get_dof_index(name) for name in self.config["robot"]["wheel_joints"]]
        if any(index < 0 for index in wheel_indices):
            raise AssertionError(f"Nova Carter wheel joints missing: {dof_names}")
        if not articulation_roots or not collision_prims:
            raise AssertionError("Nova Carter articulation/collision geometry is incomplete")
        return {
            "asset": robot_url, "articulation_roots": articulation_roots,
            "dof_names": dof_names, "num_dof": int(self.robot.num_dof),
            "wheel_joint_names": self.config["robot"]["wheel_joints"],
            "wheel_joint_indices": wheel_indices,
            "collision_prim_count": len(collision_prims),
            "rigid_body_count": len(rigid_bodies),
            "contact_body_paths": self.contact_body_paths,
            "collision_prims": collision_prims,
        }

    def _contact_force(self) -> tuple[float, list[float]]:
        value = self.contact_view.get_net_contact_forces(dt=self.dt)
        if value is None:
            return 0.0, [0.0, 0.0, 0.0]
        force = np.asarray(value, dtype=float).reshape(-1, 3).sum(axis=0)
        return float(np.linalg.norm(force[:2])), force.tolist()

    def _update_viewer_camera(self) -> None:
        """Keep the visible viewport inside the scene behind Nova Carter."""
        if ARGS.headless:
            return
        position, orientation = self.robot.get_world_pose()
        position = vector3(position)
        yaw = quaternion_yaw(orientation)
        forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
        eye = position - 2.2 * forward + np.asarray([0.0, 0.0, 1.35])
        target = position + 0.8 * forward + np.asarray([0.0, 0.0, 0.35])
        set_camera_view(eye=eye.tolist(), target=target.tolist())

    def _robot_state(self) -> dict:
        position, orientation = self.robot.get_world_pose()
        linear = np.asarray(self.robot.get_linear_velocity(), dtype=float)
        angular = np.asarray(self.robot.get_angular_velocity(), dtype=float)
        return {
            "position_xyz_m": np.asarray(position, dtype=float).tolist(),
            "orientation_wxyz": np.asarray(orientation, dtype=float).tolist(),
            "yaw_rad": quaternion_yaw(orientation),
            "linear_velocity_xyz_m_s": linear.tolist(),
            "angular_velocity_xyz_rad_s": angular.tolist(),
            "wheel_positions_rad": np.asarray(self.robot.get_wheel_positions(), dtype=float).tolist(),
            "wheel_velocities_rad_s": np.asarray(self.robot.get_wheel_velocities(), dtype=float).tolist(),
        }

    def observe(self) -> dict:
        state = self._robot_state()
        goal = vector3(self.config["robot"]["goal_xyz"])
        position = vector3(state["position_xyz_m"])
        delta = goal - position
        return {
            "simulation_step": self.step_index,
            "robot": state,
            "goal_relative_xy_m": [float(delta[0]), float(delta[1])],
            "goal_distance_m": float(np.linalg.norm(delta[:2])),
        }

    def reset(self, seed: int, scenario: str) -> dict:
        if scenario != self.config["scenario"]:
            raise ValueError(f"environment was built for {self.config['scenario']}, not {scenario}")
        random.seed(seed)
        np.random.seed(seed)
        start = vector3(self.config["robot"]["start_xyz"])
        orientation = yaw_quaternion(float(self.config["robot"]["start_yaw_rad"]))
        self.robot.set_world_pose(position=start, orientation=orientation)
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))
        self.robot.set_joint_positions(np.zeros(self.robot.num_dof))
        self.robot.set_joint_velocities(np.zeros(self.robot.num_dof))
        self.limiter.reset()
        self.diff_controller.reset()
        self.robot.apply_wheel_actions(self.diff_controller.forward(np.zeros(2)))
        self.baseline.reset()
        self.step_index = 0
        self.path_length = 0.0
        for controller in self.controllers:
            controller.reset_route()
        for _ in range(int(self.config["episode"]["reset_settle_steps"])):
            self.world.step(render=True)
            for controller in self.controllers:
                controller.enforce_pose()
        self._update_viewer_camera()
        observation = self.observe()
        self.previous_position = vector3(observation["robot"]["position_xyz_m"])
        self.previous_goal_distance = float(observation["goal_distance_m"])
        return observation

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        wall_step_started = time.perf_counter()
        requested = (float(action[0]), float(action[1]))
        applied = self.limiter.apply(requested, self.dt)
        wheel_action = self.diff_controller.forward(np.asarray(applied, dtype=float))
        self.robot.apply_wheel_actions(wheel_action)
        self.world.step(render=True)
        for controller in self.controllers:
            controller.enforce_pose()
        self._update_viewer_camera()
        self.step_index += 1
        observation = self.observe()
        position = vector3(observation["robot"]["position_xyz_m"])
        distance_step = float(np.linalg.norm(position[:2] - self.previous_position[:2]))
        self.path_length += distance_step
        self.previous_position = position
        contact_force, contact_vector = self._contact_force()
        collision = contact_force >= float(self.config["collision"]["planar_force_threshold_n"])
        goal_reached = observation["goal_distance_m"] <= float(self.config["episode"]["goal_tolerance_m"])
        timeout = self.step_index >= int(self.config["episode"]["max_steps"])
        progress = self.previous_goal_distance - observation["goal_distance_m"]
        self.previous_goal_distance = observation["goal_distance_m"]
        costs = self.config["navigation_costs"]
        reward_terms = {
            "progress": float(costs["path_length"]) * progress,
            "elapsed_time": -float(costs["elapsed_time"]) * self.dt,
            "control_effort": -float(costs["control_effort"]) * (abs(applied[0]) + abs(applied[1])) * self.dt,
            "collision": -float(costs["collision"]) if collision else 0.0,
            "goal": float(costs["goal_bonus"]) if goal_reached else 0.0,
        }
        reward = float(sum(reward_terms.values()))
        terminated = bool(goal_reached or (collision and self.config["episode"]["collision_termination"]))
        truncated = bool(timeout and not terminated)
        wheel_targets = np.asarray(wheel_action.joint_velocities, dtype=float).tolist()
        info = {
            "requested_action": list(requested), "applied_action": list(applied),
            "wheel_velocity_targets_rad_s": wheel_targets,
            "reward_terms": reward_terms, "goal_reached": goal_reached,
            "collision": collision, "timeout": timeout,
            "planar_contact_force_n": contact_force, "contact_force_xyz_n": contact_vector,
            "path_length_m": self.path_length,
        }
        if self.stream is not None:
            self._write_log(observation, reward, terminated, truncated, info)
        if not ARGS.headless:
            remaining = self.dt - (time.perf_counter() - wall_step_started)
            if remaining > 0.0:
                time.sleep(remaining)
        return observation, reward, terminated, truncated, info

    def _write_log(self, observation, reward, terminated, truncated, info) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self.seed)
        pedestrians = []
        for cfg, person, controller in self.people:
            velocity = (controller.position - controller.previous_position) / self.dt
            pedestrians.append({
                "id": cfg["id"], "position_xyz_m": vector3(person.state.position).tolist(),
                "orientation_xyzw": np.asarray(person.state.orientation, dtype=float).tolist(),
                "velocity_xyz_m_s": velocity.tolist(),
                "current_destination_xyz_m": controller.current_destination.tolist(),
            })
        payload = {
            "timestamp": (start + timedelta(seconds=self.step_index * self.dt)).isoformat(),
            "simulation_step": self.step_index, "simulation_time_s": self.step_index * self.dt,
            "seed": self.seed, "scenario": self.config["scenario"],
            "world_frame": "isaac_world_enu",
            "environment_state": {
                "robot_independent": True,
                "official_isaac_asset": self.environment_url,
            },
            "observation": observation, "privileged_log": {"pedestrians": pedestrians},
            "reward": reward, "terminated": terminated, "truncated": truncated, "info": info,
        }
        self.stream.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")

    def run_baseline(self) -> dict:
        self.reset(self.seed, self.config["scenario"])
        log_path = self.output / "trajectory.jsonl"
        self.stream = log_path.open("w", encoding="utf-8")
        initial = self.observe()
        max_speed, max_angular, max_contact = 0.0, 0.0, 0.0
        forward_commanded = rotation_commanded = stop_commanded = False
        terminal_result = None
        try:
            for _ in range(int(self.config["episode"]["max_steps"])):
                state = self._robot_state()
                action = self.baseline.action(
                    state["position_xyz_m"][0], state["position_xyz_m"][1], state["yaw_rad"]
                )
                forward_commanded |= action[0] > 0.05
                rotation_commanded |= abs(action[1]) > 0.05
                result = self.step(action)
                observation, _, terminated, truncated, info = result
                speed = math.hypot(*observation["robot"]["linear_velocity_xyz_m_s"][:2])
                max_speed = max(max_speed, speed)
                max_angular = max(max_angular, abs(observation["robot"]["angular_velocity_xyz_rad_s"][2]))
                max_contact = max(max_contact, info["planar_contact_force_n"])
                if terminated or truncated:
                    terminal_result = result
                    break
            for _ in range(90):
                self.step((0.0, 0.0))
                stop_commanded = True
        finally:
            self.stream.close()
            self.stream = None
        final = self.observe()
        displacement = float(np.linalg.norm(
            vector3(final["robot"]["position_xyz_m"])[:2]
            - vector3(initial["robot"]["position_xyz_m"])[:2]
        ))
        stopped_speed = math.hypot(*final["robot"]["linear_velocity_xyz_m_s"][:2])
        return {
            "initial": initial, "final": final, "displacement_m": displacement,
            "path_length_m": self.path_length, "max_linear_speed_m_s": max_speed,
            "max_angular_speed_rad_s": max_angular, "max_planar_contact_force_n": max_contact,
            "forward_commanded": forward_commanded, "rotation_commanded": rotation_commanded,
            "stop_commanded": stop_commanded, "stopped_speed_m_s": stopped_speed,
            "goal_reached": bool(terminal_result and terminal_result[4]["goal_reached"]),
            "collision": bool(terminal_result and terminal_result[4]["collision"]),
            "timeout": bool(terminal_result and terminal_result[4]["timeout"]),
            "steps": self.step_index,
        }

    def validate_reset_rotation_stop(self) -> dict:
        first = self.reset(self.seed, self.config["scenario"])
        for _ in range(90):
            rotated = self.step((0.0, 0.6))[0]
        yaw_delta = abs(wrap_angle(rotated["robot"]["yaw_rad"] - first["robot"]["yaw_rad"]))
        for _ in range(90):
            stopped = self.step((0.0, 0.0))[0]
        angular_stopped = abs(stopped["robot"]["angular_velocity_xyz_rad_s"][2])
        reset = self.reset(self.seed, self.config["scenario"])
        reset_error = float(np.linalg.norm(
            vector3(first["robot"]["position_xyz_m"]) - vector3(reset["robot"]["position_xyz_m"])
        ))
        return {
            "yaw_delta_rad": yaw_delta, "stopped_angular_speed_rad_s": angular_stopped,
            "reset_position_error_m": reset_error,
        }

    def validate_collision_probe(self) -> dict | None:
        probe = self.config.get("collision_probe", {})
        if not probe.get("enabled"):
            return None
        initial = self.reset(self.seed, self.config["scenario"])
        initial_position = vector3(initial["robot"]["position_xyz_m"])
        detected = False
        maximum = 0.0
        detection_step = None
        final_observation = initial
        for _ in range(int(probe["drive_steps"])):
            final_observation, _, terminated, _, info = self.step(probe["command"])
            maximum = max(maximum, info["planar_contact_force_n"])
            if info["collision"]:
                detected, detection_step = True, self.step_index
                break
            if terminated:
                break
        final_position = vector3(final_observation["robot"]["position_xyz_m"])
        return {
            "detected": detected, "detection_step": detection_step,
            "max_planar_contact_force_n": maximum,
            "drive_steps_completed": self.step_index,
            "displacement_m": float(np.linalg.norm(final_position[:2] - initial_position[:2])),
            "initial_position_xyz_m": initial_position.tolist(),
            "final_position_xyz_m": final_position.tolist(),
        }

    def close(self) -> None:
        self.robot.apply_wheel_actions(self.diff_controller.forward(np.zeros(2)))
        self.timeline.stop()


def run() -> dict:
    started = time.perf_counter()
    config = load_stage2_config(ARGS.config.resolve())
    env = NovaCarterNavigationEnv(config, ARGS.output.resolve(), ARGS.seed)
    try:
        interface_checks = env.validate_reset_rotation_stop()
        collision_probe = env.validate_collision_probe()
        trajectory = env.run_baseline()
        runtime_evidence = {
            "interface_checks": interface_checks,
            "collision_probe": collision_probe,
            "trajectory": trajectory,
        }
        (ARGS.output / "runtime-checks.json").write_text(
            json.dumps(runtime_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        runtime_errors = []
        if interface_checks["yaw_delta_rad"] < 0.25:
            runtime_errors.append("rotation response was below 0.25 rad")
        if interface_checks["stopped_angular_speed_rad_s"] > 0.15:
            runtime_errors.append("rotation stop exceeded 0.15 rad/s")
        if interface_checks["reset_position_error_m"] > 0.08:
            runtime_errors.append("reset position error exceeded 0.08 m")
        if trajectory["displacement_m"] < 0.25:
            runtime_errors.append("physical displacement was below 0.25 m")
        if trajectory["stopped_speed_m_s"] > 0.15:
            runtime_errors.append("stop speed exceeded 0.15 m/s")
        initial_goal_distance = float(trajectory["initial"]["goal_distance_m"])
        final_goal_distance = float(trajectory["final"]["goal_distance_m"])
        if final_goal_distance > initial_goal_distance - 0.25:
            runtime_errors.append("baseline did not approach the goal by at least 0.25 m")
        if trajectory["collision"]:
            runtime_errors.append("baseline terminated in collision")
        if collision_probe is not None and not collision_probe["detected"]:
            runtime_errors.append("controlled collision probe was not detected")
        if runtime_errors:
            raise AssertionError("; ".join(runtime_errors))
        result = {
            "status": "pass", "timestamp_utc": datetime.now(UTC).isoformat(),
            "backend": config["backend"], "scenario": config["scenario"], "seed": ARGS.seed,
            "runtime": {"gpu_index": ARGS.gpu, "headless": ARGS.headless},
            "environment_asset": env.environment_url, "articulation": env.articulation,
            "stage1_config": env.stage1_config_path,
            "pedestrian_route_cache": env.route_cache_path,
            "controller": {
                "api": "isaacsim.robot.wheeled_robots.controllers.DifferentialController",
                "wheel_radius_m": config["robot"]["wheel_radius_m"],
                "wheel_base_m": config["robot"]["wheel_base_m"],
                "motion_model": "physical wheel velocity targets; no pose writes during step",
            },
            "interface_checks": interface_checks, "collision_probe": collision_probe,
            "trajectory": trajectory, "elapsed_s": time.perf_counter() - started,
        }
        if not ARGS.headless:
            hold_deadline = time.monotonic() + 3.0
            while time.monotonic() < hold_deadline:
                simulation_app.update()
                time.sleep(1.0 / 60.0)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    result = {}
    exit_code = 0
    try:
        result = run()
    except Exception as error:
        exit_code = 1
        result = {
            "status": "fail", "timestamp_utc": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        ARGS.output.mkdir(parents=True, exist_ok=True)
        (ARGS.output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        simulation_app.close()
    raise SystemExit(exit_code)
