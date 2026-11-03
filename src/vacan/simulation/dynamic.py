"""Deterministic 2D SFM backend matching iCrowdNav's simulator contract.

This lightweight Stage 1 backend follows the public force terms and
differential-drive interface in BRoln7/icrowdnav. It is not Isaac Sim and does
not claim to contain the unpublished Office/Hospital/Warehouse USD files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .grid import GridSpec, _contains
from .ground_truth import GroundTruthRecorder, GroundTruthState, PedestrianState, RobotState
from .observations import PartialObservationProvider
from .scenario import ScenarioDefinition, load_scenario


@dataclass
class _Pedestrian:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    waypoint: int
    speed_limit: float


class DynamicPedestrianSimulation:
    """Policy-safe facade over the reproducible iCrowdNav-style SFM backend."""

    def __init__(self, scenario: str | Path | ScenarioDefinition, recorder: GroundTruthRecorder | None = None) -> None:
        self.scenario = scenario if isinstance(scenario, ScenarioDefinition) else load_scenario(scenario)
        self._recorder = recorder
        self._observer = PartialObservationProvider(self.scenario.sensors)
        self._grid = GridSpec(self.scenario.map_width_m, self.scenario.map_height_m, self.scenario.grid_resolution_m)
        self._step = 0
        self._pedestrians: list[_Pedestrian] = []
        self._robot = np.zeros(5, dtype=np.float64)
        self._last_action = (0.0, 0.0)
        self._collision = False

    @property
    def action_space(self) -> dict[str, Any]:
        return {"type": "differential_drive", "linear_m_s": [0.0, 1.0], "angular_rad_s": [-2.0, 2.0]}

    @property
    def episode_length(self) -> int:
        return self.scenario.episode_length

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        actual_seed = self.scenario.seed if seed is None else int(seed)
        rng = np.random.default_rng(actual_seed)
        self._step = 0
        self._collision = False
        self._robot[:] = (*self.scenario.robot_start, 0.0, 0.0)
        self._pedestrians = []
        for index, flow in enumerate(self.scenario.pedestrian_flows):
            start_x, start_y = (float(v) for v in flow["start"])
            speed_limit = self.scenario.pedestrian_max_speed * float(rng.uniform(0.82, 1.0))
            self._pedestrians.append(_Pedestrian(index, start_x, start_y, 0.0, 0.0, 0, speed_limit))
        state = self._snapshot(False, False)
        if self._recorder:
            self._recorder.record(state)
        return self._observation(state), {"seed": actual_seed, "backend": "icrowdnav-sfm"}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float64).reshape(-1)
        if command.size != 2:
            raise ValueError("iCrowdNav control requires [linear_velocity, angular_velocity]")
        v = float(np.clip(command[0], 0.0, 1.0))
        omega = float(np.clip(command[1], -2.0, 2.0))
        self._last_action = (v, omega)
        dt = self.scenario.time_step_s
        self._robot[2] = _wrap(self._robot[2] + omega * dt)
        candidate = self._robot[:2] + v * dt * np.array([math.cos(self._robot[2]), math.sin(self._robot[2])])
        if self._valid_position(candidate[0], candidate[1], self.scenario.robot_radius):
            self._robot[:2] = candidate
            self._robot[3:5] = (v, omega)
        else:
            self._robot[3:5] = (0.0, omega)
            self._collision = True
        self._advance_pedestrians()
        self._step += 1
        distance = float(np.linalg.norm(self._robot[:2] - np.asarray(self.scenario.robot_goal)))
        terminated = distance <= 0.55
        truncated = self._step >= self.scenario.episode_length
        state = self._snapshot(terminated, truncated)
        if self._recorder:
            self._recorder.record(state)
        reward = -distance - (5.0 if self._collision else 0.0)
        return self._observation(state), reward, terminated, truncated, {
            "collision": self._collision, "success": terminated, "timeout": truncated,
        }

    def close(self) -> None:
        return None

    def render_frame(self) -> np.ndarray:
        import cv2

        scale = 70
        height = int(self.scenario.map_height_m * scale)
        width = int(self.scenario.map_width_m * scale)
        frame = np.full((height, width, 3), 245, dtype=np.uint8)

        def pixel(x: float, y: float) -> tuple[int, int]:
            ox, oy = self._grid.origin
            return int((x - ox) * scale), int(height - (y - oy) * scale)

        for obstacle in self.scenario.obstacles:
            cx, cy = pixel(float(obstacle["x"]), float(obstacle["y"]))
            if obstacle["shape"] == "circle":
                cv2.circle(frame, (cx, cy), int(float(obstacle["radius"]) * scale), (90, 90, 90), -1)
            else:
                rect = ((cx, cy), (float(obstacle["length"]) * scale, float(obstacle["width"]) * scale), -math.degrees(float(obstacle.get("orientation", 0.0))))
                cv2.fillConvexPoly(frame, cv2.boxPoints(rect).astype(np.int32), (90, 90, 90))
        for pedestrian in self._pedestrians:
            cv2.circle(frame, pixel(pedestrian.x, pedestrian.y), int(self.scenario.pedestrian_radius * scale), (40, 80, 220), -1)
        rx, ry = pixel(self._robot[0], self._robot[1])
        cv2.circle(frame, (rx, ry), int(self.scenario.robot_radius * scale), (30, 190, 40), -1)
        cv2.line(frame, (rx, ry), pixel(self._robot[0] + 0.5 * math.cos(self._robot[2]), self._robot[1] + 0.5 * math.sin(self._robot[2])), (0, 80, 0), 3)
        cv2.circle(frame, pixel(*self.scenario.robot_goal), 10, (0, 190, 255), 2)
        cv2.putText(frame, f"{self.scenario.name} | iCrowdNav SFM contract | step {self._step}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        return frame

    def _advance_pedestrians(self) -> None:
        positions = np.array([[p.x, p.y] for p in self._pedestrians], dtype=np.float64)
        velocities = np.array([[p.vx, p.vy] for p in self._pedestrians], dtype=np.float64)
        updates: list[np.ndarray] = []
        dt = self.scenario.time_step_s
        sfm = self.scenario.raw.get("sfm", {})
        gain_k = float(sfm.get("goal_gain", 3.0))
        gain_a = float(sfm.get("agent_repulsion_gain", 4.0))
        gain_b = float(sfm.get("agent_repulsion_range_m", 0.5))
        safe = float(sfm.get("safe_distance_m", 0.75))
        for index, ped in enumerate(self._pedestrians):
            flow = self.scenario.pedestrian_flows[index]
            goal = np.asarray(flow["waypoints"][ped.waypoint], dtype=np.float64)
            pos = positions[index]
            delta = goal - pos
            distance = float(np.linalg.norm(delta))
            desired = np.zeros(2) if distance < 1e-9 else ped.speed_limit * delta / distance
            acceleration = gain_k * (desired - velocities[index])
            for other_index, other in enumerate(positions):
                if other_index == index:
                    continue
                away = pos - other
                separation = float(np.linalg.norm(away))
                if 1e-6 < separation < 3.0:
                    acceleration += gain_a * math.exp((safe - separation) / gain_b) * away / separation
            robot_away = pos - self._robot[:2]
            robot_distance = float(np.linalg.norm(robot_away))
            if 1e-6 < robot_distance < 3.0:
                acceleration += gain_a * math.exp((safe - robot_distance) / gain_b) * robot_away / robot_distance
            acceleration += self._obstacle_force(pos)
            velocity = velocities[index] + acceleration * dt
            speed = float(np.linalg.norm(velocity))
            if speed > ped.speed_limit:
                velocity *= ped.speed_limit / speed
            updates.append(velocity)
        for ped, velocity in zip(self._pedestrians, updates, strict=True):
            candidate = np.array([ped.x, ped.y]) + velocity * dt
            if self._valid_position(candidate[0], candidate[1], self.scenario.pedestrian_radius):
                ped.x, ped.y = map(float, candidate)
                ped.vx, ped.vy = map(float, velocity)
            else:
                ped.vx = ped.vy = 0.0
            goal = np.asarray(self.scenario.pedestrian_flows[ped.id]["waypoints"][ped.waypoint])
            if np.linalg.norm(np.array([ped.x, ped.y]) - goal) <= 0.4:
                ped.waypoint = (ped.waypoint + 1) % len(self.scenario.pedestrian_flows[ped.id]["waypoints"])

    def _obstacle_force(self, position: np.ndarray) -> np.ndarray:
        force = np.zeros(2)
        for obstacle in self.scenario.obstacles:
            nearest = _nearest_point(obstacle, position)
            away = position - nearest
            distance = float(np.linalg.norm(away))
            if 1e-6 < distance < 1.5:
                force += 4.0 * math.exp((0.5 - distance) / 0.25) * away / distance
        return force

    def _valid_position(self, x: float, y: float, radius: float) -> bool:
        if abs(x) > self.scenario.map_width_m / 2 - radius or abs(y) > self.scenario.map_height_m / 2 - radius:
            return False
        return not any(_contains(obstacle, x, y, padding=radius) for obstacle in self.scenario.obstacles)

    def _observation(self, state: GroundTruthState) -> dict[str, Any]:
        dx = self.scenario.robot_goal[0] - self._robot[0]
        dy = self.scenario.robot_goal[1] - self._robot[1]
        c, s = math.cos(self._robot[2]), math.sin(self._robot[2])
        robot_obs = np.array([c * dx + s * dy, -s * dx + c * dy, self._robot[3], self._robot[4]], dtype=np.float32)
        return self._observer.observe(state, robot_obs)

    def _snapshot(self, terminated: bool, truncated: bool) -> GroundTruthState:
        pedestrians = tuple(self._pedestrian_states())
        robot_point = type("RobotPoint", (), {"x": self._robot[0], "y": self._robot[1]})()
        occupancy, counts, density = self._grid.rasterize(self.scenario.obstacles, pedestrians, robot_point)
        timestamp = self.scenario.start_timestamp + timedelta(seconds=self._step * self.scenario.time_step_s)
        zones: dict[str, dict[str, float | int]] = {}
        for name, (min_x, min_y, max_x, max_y) in self.scenario.congestion_zones.items():
            count = sum(min_x <= p.x <= max_x and min_y <= p.y <= max_y for p in pedestrians)
            area = (max_x - min_x) * (max_y - min_y)
            zones[name] = {"pedestrian_count": count, "area_m2": area, "density_per_m2": count / area}
        interactions: list[list[int]] = []
        for i, first in enumerate(pedestrians):
            for second in pedestrians[i + 1:]:
                if math.hypot(first.x - second.x, first.y - second.y) < 1.2:
                    interactions.append([first.id, second.id])
        return GroundTruthState(
            scenario=self.scenario.name, timestamp=timestamp.isoformat(), simulation_step=self._step,
            simulation_time_s=self._step * self.scenario.time_step_s,
            robot=RobotState(x=float(self._robot[0]), y=float(self._robot[1]), orientation=float(self._robot[2]), velocity_x=float(self._robot[3] * math.cos(self._robot[2])), velocity_y=float(self._robot[3] * math.sin(self._robot[2])), angular_velocity=float(self._robot[4]), action=self._last_action, goal_x=self.scenario.robot_goal[0], goal_y=self.scenario.robot_goal[1], goal_orientation=0.0),
            pedestrians=pedestrians, static_obstacles=self.scenario.obstacles,
            interactions={"human-human": interactions, "human-object": []},
            grid={"origin": self._grid.origin, "shape": [self._grid.rows, self._grid.cols], "resolution_m": self._grid.resolution_m, "occupancy": occupancy.tolist(), "pedestrian_counts": counts.tolist(), "pedestrian_density_per_m2": density.tolist()},
            congestion_zones=zones,
            collision={"any": self._collision or self._robot_human_collision(), "human": self._robot_human_collision(), "object": self._collision, "success": terminated, "timeout": truncated},
            terminated=terminated, truncated=truncated,
        )

    def _pedestrian_states(self) -> list[PedestrianState]:
        records = []
        for ped, flow in zip(self._pedestrians, self.scenario.pedestrian_flows, strict=True):
            goal = flow["waypoints"][ped.waypoint]
            orientation = math.atan2(ped.vy, ped.vx) if math.hypot(ped.vx, ped.vy) > 1e-8 else 0.0
            records.append(PedestrianState(ped.id, "dynamic", ped.x, ped.y, orientation, ped.vx, ped.vy, math.hypot(ped.vx, ped.vy), float(goal[0]), float(goal[1]), str(flow.get("direction", "configured"))))
        return records

    def _robot_human_collision(self) -> bool:
        return any(math.hypot(p.x - self._robot[0], p.y - self._robot[1]) <= self.scenario.robot_radius + self.scenario.pedestrian_radius for p in self._pedestrians)


def _nearest_point(obstacle: dict[str, Any], point: np.ndarray) -> np.ndarray:
    center = np.array([float(obstacle["x"]), float(obstacle["y"])])
    if obstacle["shape"] == "circle":
        delta = point - center
        distance = np.linalg.norm(delta)
        return center if distance < 1e-9 else center + delta / distance * float(obstacle["radius"])
    angle = float(obstacle.get("orientation", 0.0))
    rotation = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    local = rotation @ (point - center)
    local[0] = np.clip(local[0], -float(obstacle["length"]) / 2, float(obstacle["length"]) / 2)
    local[1] = np.clip(local[1], -float(obstacle["width"]) / 2, float(obstacle["width"]) / 2)
    return center + rotation.T @ local


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
