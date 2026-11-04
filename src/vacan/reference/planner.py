"""Time-expanded full-information reference planner.

Current privileged pedestrian position and velocity initialize constant-velocity
predictions. Actual future simulator trajectories are never consumed.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vacan.mapping.static_map import OccupancyMap2D
from vacan.robot.stage2 import VelocityLimits, wrap_angle


@dataclass(frozen=True)
class PedestrianState:
    identifier: str
    position_xy_m: tuple[float, float]
    velocity_xy_m_s: tuple[float, float]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "PedestrianState":
        return cls(str(record["id"]), tuple(map(float, record["position_xyz_m"][:2])),
                   tuple(map(float, record["velocity_xyz_m_s"][:2])))


@dataclass(frozen=True)
class PlanResult:
    feasible: bool
    reason: str
    world_xy_m: list[tuple[float, float]]
    timestamps_s: list[float]
    grid_states: list[tuple[int, int, int, int]]
    expanded_states: int


class TimeExpandedAStar:
    """Heading-aware A* over (time,row,column,heading) states."""

    _directions = ((0, 1), (1, 1), (1, 0), (1, -1),
                   (0, -1), (-1, -1), (-1, 0), (-1, 1))

    def __init__(self, static_map: OccupancyMap2D, limits: VelocityLimits, *,
                 time_step_s: float, horizon_s: float, pedestrian_radius_m: float,
                 goal_tolerance_m: float) -> None:
        if time_step_s <= 0 or horizon_s <= time_step_s:
            raise ValueError("invalid reference planning time geometry")
        if pedestrian_radius_m <= 0 or goal_tolerance_m <= 0:
            raise ValueError("reference safety radii must be positive")
        if (math.pi / 4.0) / time_step_s > limits.max_angular_rad_s + 1e-9:
            raise ValueError("planner time step violates angular velocity limit")
        diagonal_speed = static_map.geometry.resolution_m * math.sqrt(2.0) / time_step_s
        if diagonal_speed > limits.max_linear_m_s + 1e-9:
            raise ValueError("planner time step violates linear velocity limit")
        self.map, self.limits = static_map, limits
        self.dt = float(time_step_s)
        self.max_time_index = int(math.floor(horizon_s / time_step_s))
        self.pedestrian_radius = float(pedestrian_radius_m)
        self.goal_tolerance = float(goal_tolerance_m)

    @staticmethod
    def _heading_bin(yaw: float) -> int:
        return int(round((yaw % (2.0 * math.pi)) / (math.pi / 4.0))) % 8

    def _dynamic_blocked(self, cell: tuple[int, int], time_s: float,
                         pedestrians: list[PedestrianState]) -> bool:
        center = np.asarray(self.map.geometry.cell_center(cell), dtype=float)
        return any(float(np.linalg.norm(
            center - (np.asarray(person.position_xy_m)
                      + np.asarray(person.velocity_xy_m_s) * time_s)
        )) < self.pedestrian_radius for person in pedestrians)

    def plan(self, start_xy_m: tuple[float, float], start_yaw_rad: float,
             goal_xy_m: tuple[float, float], pedestrians: list[PedestrianState], *,
             start_time_s: float = 0.0) -> PlanResult:
        geometry = self.map.geometry
        start_cell = geometry.world_to_grid(*start_xy_m)
        goal_cell = geometry.world_to_grid(*goal_xy_m)
        if start_cell is None or goal_cell is None:
            return PlanResult(False, "start_or_goal_outside_map", [], [], [], 0)
        obstacles = self.map.policy_obstacle.astype(bool)
        if obstacles[start_cell] or obstacles[goal_cell]:
            return PlanResult(False, "start_or_goal_in_static_obstacle", [], [], [], 0)
        start = (0, start_cell[0], start_cell[1], self._heading_bin(start_yaw_rad))
        frontier: list[tuple[float, float, tuple[int, int, int, int]]] = [(0.0, 0.0, start)]
        cost = {start: 0.0}
        parent: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None] = {start: None}
        expanded, terminal = 0, None
        tolerance_cells = self.goal_tolerance / geometry.resolution_m
        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current_cost != cost.get(current):
                continue
            t, row, column, heading = current
            expanded += 1
            if math.hypot(goal_cell[0] - row, goal_cell[1] - column) <= tolerance_cells:
                terminal = current
                break
            if t >= self.max_time_index:
                continue
            dr, dc = self._directions[heading]
            candidates = [(row, column, heading), (row, column, (heading - 1) % 8),
                          (row, column, (heading + 1) % 8),
                          (row + dr, column + dc, heading)]
            for next_row, next_column, next_heading in candidates:
                if not (0 <= next_row < obstacles.shape[0] and 0 <= next_column < obstacles.shape[1]):
                    continue
                if obstacles[next_row, next_column]:
                    continue
                if next_row != row and next_column != column:
                    if obstacles[row, next_column] or obstacles[next_row, column]:
                        continue
                next_t = t + 1
                if self._dynamic_blocked((next_row, next_column), next_t * self.dt, pedestrians):
                    continue
                nxt = (next_t, next_row, next_column, next_heading)
                new_cost = current_cost + 1.0
                if new_cost >= cost.get(nxt, math.inf):
                    continue
                cost[nxt], parent[nxt] = new_cost, current
                heuristic = math.hypot(goal_cell[0] - next_row, goal_cell[1] - next_column)
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, nxt))
        if terminal is None:
            return PlanResult(False, "horizon_exhausted_or_blocked", [], [], [], expanded)
        states = []
        node: tuple[int, int, int, int] | None = terminal
        while node is not None:
            states.append(node)
            node = parent[node]
        states.reverse()
        return PlanResult(True, "feasible",
                          [geometry.cell_center((s[1], s[2])) for s in states],
                          [start_time_s + s[0] * self.dt for s in states], states, expanded)


class ReferenceWaypointController:
    """Convert a planned world-frame path into bounded unicycle commands."""

    def __init__(self, limits: VelocityLimits, cruise_speed_m_s: float,
                 angular_gain: float, waypoint_tolerance_m: float) -> None:
        self.limits = limits
        self.cruise = min(float(cruise_speed_m_s), limits.max_linear_m_s)
        self.gain, self.tolerance = float(angular_gain), float(waypoint_tolerance_m)
        self.path: list[tuple[float, float]] = []
        self.index = 0

    def set_path(self, points: list[tuple[float, float]]) -> None:
        self.path, self.index = list(points), (1 if len(points) > 1 else 0)

    def action(self, position_xy_m: tuple[float, float], yaw_rad: float) -> tuple[float, float]:
        if not self.path:
            return 0.0, 0.0
        while self.index < len(self.path) - 1 and math.dist(
                position_xy_m, self.path[self.index]) <= self.tolerance:
            self.index += 1
        target = self.path[self.index]
        error = wrap_angle(math.atan2(target[1] - position_xy_m[1],
                                      target[0] - position_xy_m[0]) - yaw_rad)
        angular = float(np.clip(self.gain * error, -self.limits.max_angular_rad_s,
                                self.limits.max_angular_rad_s))
        alignment = max(0.0, 1.0 - abs(error) / (math.pi / 2.0))
        return self.cruise * alignment, angular
