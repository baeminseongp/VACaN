"""Host-verifiable Stage 4 observation, action, reward, and baseline logic."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vacan.mapping.dynamic_costmap import GridGeometry
from vacan.mapping.static_map import OccupancyMap2D
from vacan.robot.stage2 import ActionLimiter, VelocityLimits, wrap_angle


@dataclass(frozen=True)
class ObservationSpec:
    crop_rows: int
    crop_columns: int
    rotate_with_robot: bool
    crowd_clip: float
    max_goal_distance_m: float
    max_linear_m_s: float
    max_angular_rad_s: float


class ObservationBuilder:
    """Build two policy map channels plus normalized robot state, with no GT input."""

    channel_order = ("static_obstacle", "dynamic_crowd")
    robot_state_order = (
        "goal_x", "goal_y", "goal_distance", "goal_angle",
        "linear_velocity", "angular_velocity", "previous_linear", "previous_angular",
    )

    def __init__(self, static_map: OccupancyMap2D, spec: ObservationSpec,
                 dynamic_geometry: GridGeometry | None = None) -> None:
        if spec.crop_rows % 2 != 1 or spec.crop_columns % 2 != 1:
            raise ValueError("robot-centered crop dimensions must be odd")
        self.static_map = static_map
        self.geometry = static_map.geometry
        self.dynamic_geometry = dynamic_geometry or static_map.geometry
        self.spec = spec

    def _sample(self, layer: np.ndarray, layer_geometry: GridGeometry,
                robot_xy: tuple[float, float], yaw: float, outside_value: float) -> np.ndarray:
        result = np.full((self.spec.crop_rows, self.spec.crop_columns),
                         outside_value, dtype=np.float32)
        center_row, center_column = self.spec.crop_rows // 2, self.spec.crop_columns // 2
        resolution = self.geometry.resolution_m
        for row in range(self.spec.crop_rows):
            for column in range(self.spec.crop_columns):
                dx = (column - center_column) * resolution
                dy = (row - center_row) * resolution
                if self.spec.rotate_with_robot:
                    dx, dy = (math.cos(yaw) * dx - math.sin(yaw) * dy,
                              math.sin(yaw) * dx + math.cos(yaw) * dy)
                cell = layer_geometry.world_to_grid(robot_xy[0] + dx, robot_xy[1] + dy)
                if cell is not None:
                    result[row, column] = float(layer[cell])
        return result

    def build(self, dynamic_crowd: np.ndarray, robot: dict[str, Any],
              goal_xy: tuple[float, float], previous_action: tuple[float, float]) -> dict[str, np.ndarray]:
        if dynamic_crowd.shape != self.dynamic_geometry.shape:
            raise ValueError("dynamic crowd array does not match its configured geometry")
        position = tuple(map(float, robot["position_xyz_m"][:2]))
        yaw = float(robot["yaw_rad"])
        static = self._sample(self.static_map.policy_obstacle, self.geometry,
                              position, yaw, 1.0)
        crowd = self._sample(dynamic_crowd, self.dynamic_geometry,
                             position, yaw, 0.0)
        crowd = np.clip(crowd / self.spec.crowd_clip, 0.0, 1.0)
        maps = np.stack((static, crowd), axis=0).astype(np.float32)
        dx, dy = goal_xy[0] - position[0], goal_xy[1] - position[1]
        distance = math.hypot(dx, dy)
        angle = wrap_angle(math.atan2(dy, dx) - yaw)
        linear = float(robot["linear_velocity_xyz_m_s"][0])
        angular = float(robot["angular_velocity_xyz_rad_s"][2])
        state = np.asarray([
            np.clip(dx / self.spec.max_goal_distance_m, -1, 1),
            np.clip(dy / self.spec.max_goal_distance_m, -1, 1),
            np.clip(distance / self.spec.max_goal_distance_m, 0, 1), angle / math.pi,
            np.clip(linear / self.spec.max_linear_m_s, -1, 1),
            np.clip(angular / self.spec.max_angular_rad_s, -1, 1),
            np.clip(previous_action[0] / self.spec.max_linear_m_s, -1, 1),
            np.clip(previous_action[1] / self.spec.max_angular_rad_s, -1, 1),
        ], dtype=np.float32)
        return {"map": maps, "robot": state}


class NormalizedActionAdapter:
    """Map [-1,1]^2 policy output to acceleration-limited [v, omega]."""

    def __init__(self, limits: VelocityLimits, control_dt_s: float) -> None:
        self.limits, self.control_dt_s = limits, control_dt_s
        self.limiter = ActionLimiter(limits)

    def reset(self) -> None:
        self.limiter.reset()

    def apply(self, normalized: np.ndarray) -> tuple[float, float]:
        action = np.clip(np.asarray(normalized, dtype=float), -1.0, 1.0)
        requested = (float(action[0] * self.limits.max_linear_m_s),
                     float(action[1] * self.limits.max_angular_rad_s))
        return self.limiter.apply(requested, self.control_dt_s)


class NavigationReward:
    def __init__(self, coefficients: dict[str, float]) -> None:
        self.c = {key: float(value) for key, value in coefficients.items()}

    def compute(self, previous_distance: float, distance: float, goal: bool, collision: bool,
                nearest_person_distance: float | None, action: tuple[float, float],
                previous_action: tuple[float, float]) -> tuple[float, dict[str, float]]:
        personal = self.c["personal_space_m"]
        social_violation = (0.0 if nearest_person_distance is None else
                            max(0.0, personal - nearest_person_distance) / personal)
        terms = {
            "progress": self.c["progress"] * (previous_distance - distance),
            "goal": self.c["goal"] if goal else 0.0,
            "collision": -self.c["collision"] if collision else 0.0,
            "social": -self.c["social"] * social_violation,
            "time": -self.c["time"],
            "smooth": -self.c["smooth"] * (abs(action[0] - previous_action[0])
                                             + abs(action[1] - previous_action[1])),
        }
        return float(sum(terms.values())), terms


def astar_path(obstacles: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Four-connected engineering baseline; not the research reference planner."""
    if obstacles[start] or obstacles[goal]:
        return []
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost = {start: 0.0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return list(reversed(path))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = current[0] + dr, current[1] + dc
            if not (0 <= nxt[0] < obstacles.shape[0] and 0 <= nxt[1] < obstacles.shape[1]):
                continue
            if obstacles[nxt]:
                continue
            new_cost = cost[current] + 1.0
            if new_cost < cost.get(nxt, math.inf):
                cost[nxt], parent[nxt] = new_cost, current
                priority = new_cost + abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                heapq.heappush(frontier, (priority, nxt))
    return []
