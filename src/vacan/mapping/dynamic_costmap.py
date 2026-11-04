"""Deterministic Stage 3 CCTV/robot/intention dynamic costmap."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


Cell = tuple[int, int]


@dataclass(frozen=True)
class GridGeometry:
    origin_xy: tuple[float, float]
    size_xy: tuple[float, float]
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return (
            math.ceil(self.size_xy[1] / self.resolution_m),
            math.ceil(self.size_xy[0] / self.resolution_m),
        )

    def world_to_grid(self, x: float, y: float) -> Cell | None:
        column = math.floor((x - self.origin_xy[0]) / self.resolution_m)
        row = math.floor((y - self.origin_xy[1]) / self.resolution_m)
        rows, columns = self.shape
        return (row, column) if 0 <= row < rows and 0 <= column < columns else None

    def cell_center(self, cell: Cell) -> tuple[float, float]:
        row, column = cell
        return (
            self.origin_xy[0] + (column + 0.5) * self.resolution_m,
            self.origin_xy[1] + (row + 0.5) * self.resolution_m,
        )


@dataclass
class IntentionTrack:
    confirmed: Cell | None = None
    candidate: Cell | None = None
    confirmations: int = 0


def _layer(rows: int, columns: int, value: Any) -> list[list[Any]]:
    return [[value for _ in range(columns)] for _ in range(rows)]


def _cells_for_regions(geometry: GridGeometry, regions: list[list[float]]) -> set[Cell]:
    cells: set[Cell] = set()
    rows, columns = geometry.shape
    for row in range(rows):
        for column in range(columns):
            x, y = geometry.cell_center((row, column))
            if any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in regions):
                cells.add((row, column))
    return cells


def load_stage3_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected Stage 3 schema_version 1")
    if data.get("backend") != "stage2-state-replay":
        raise ValueError(f"{path}: invalid backend")
    grid = data.get("grid", {})
    for key in ("origin_xy", "size_xy"):
        if not isinstance(grid.get(key), list) or len(grid[key]) != 2:
            raise ValueError(f"{path}: grid.{key} must contain two values")
    if float(grid.get("resolution_m", 0)) <= 0:
        raise ValueError(f"{path}: invalid grid resolution")
    if not data.get("cctv", {}).get("regions"):
        raise ValueError(f"{path}: at least one CCTV region is required")
    if float(data.get("update_frequency_hz", 0)) <= 0:
        raise ValueError(f"{path}: update_frequency_hz must be positive")
    if float(data.get("decay_time_s", 0)) != 300.0:
        raise ValueError(f"{path}: Stage 3 requires exactly 300-second decay")
    return data


class DynamicCostmap:
    """Grid costmap whose policy view never contains privileged GT layers."""

    def __init__(self, config: dict[str, Any], seed: int,
                 intervention_manager: Any | None = None) -> None:
        grid = config["grid"]
        self.geometry = GridGeometry(
            tuple(map(float, grid["origin_xy"])),
            tuple(map(float, grid["size_xy"])),
            float(grid["resolution_m"]),
        )
        self.config = config
        self.rows, self.columns = self.geometry.shape
        self.decay_time = float(config["decay_time_s"])
        self.update_interval = 1.0 / float(config["update_frequency_hz"])
        self.next_update_time: float | None = None
        self.rng = random.Random(seed)
        self.walkable_cells = _cells_for_regions(
            self.geometry, config["grid"].get("walkable_regions", [])
        )
        if not self.walkable_cells:
            self.walkable_cells = {
                (row, column) for row in range(self.rows) for column in range(self.columns)
            }
        explicit_obstacles = _cells_for_regions(
            self.geometry, config["grid"].get("static_obstacle_regions", [])
        )
        self.obstacle_cells = explicit_obstacles | {
            (row, column)
            for row in range(self.rows)
            for column in range(self.columns)
            if (row, column) not in self.walkable_cells
        }
        self.cctv_cells = _cells_for_regions(self.geometry, config["cctv"]["regions"])
        if not self.cctv_cells:
            raise ValueError("CCTV configuration covers no grid cells")
        self.walkable_mask = self._mask(self.walkable_cells)
        self.static_obstacle_mask = self._mask(self.obstacle_cells)
        self.cctv_mask = self._mask(self.cctv_cells)
        self.current_cost = _layer(self.rows, self.columns, 0.0)
        self.anchor_cost = _layer(self.rows, self.columns, 0.0)
        self.anchor_timestamp = _layer(self.rows, self.columns, 0.0)
        self.last_direct_observation_time = _layer(self.rows, self.columns, None)
        self.freshness = _layer(self.rows, self.columns, 1.0)
        self.last_update_type = _layer(self.rows, self.columns, "uninitialized")
        self.visibility_mask = _layer(self.rows, self.columns, False)
        self.overlap_ratio = _layer(self.rows, self.columns, 0.0)
        self.intention_incoming = _layer(self.rows, self.columns, 0)
        self.tracks: dict[str, IntentionTrack] = {}
        self.initialized = False
        self.cctv_mean = 0.0
        # Optional Stage 7 hook. None preserves the validated Stage 3 path.
        self.intervention_manager = intervention_manager

    def _mask(self, cells: set[Cell]) -> list[list[bool]]:
        return [
            [(row, column) in cells for column in range(self.columns)]
            for row in range(self.rows)
        ]

    def due(self, simulation_time_s: float) -> bool:
        if self.next_update_time is None:
            return True
        return simulation_time_s + 1e-9 >= self.next_update_time

    def _advance_schedule(self, simulation_time_s: float) -> None:
        if self.next_update_time is None:
            self.next_update_time = simulation_time_s + self.update_interval
            return
        while self.next_update_time <= simulation_time_s + 1e-9:
            self.next_update_time += self.update_interval

    def _ground_truth_counts(self, pedestrians: list[dict[str, Any]]) -> list[list[int]]:
        counts = _layer(self.rows, self.columns, 0)
        for person in pedestrians:
            position = person["position_xyz_m"]
            cell = self.geometry.world_to_grid(float(position[0]), float(position[1]))
            if cell is not None:
                counts[cell[0]][cell[1]] += 1
        return counts

    def _update_cctv(self, counts: list[list[float]], now: float) -> dict[str, float]:
        values: dict[str, float] = {}
        total = 0
        for row, column in sorted(self.cctv_cells):
            count = float(counts[row][column])
            values[f"{row},{column}"] = count
            total += count
            self.current_cost[row][column] = float(count)
            self.anchor_cost[row][column] = float(count)
            self.anchor_timestamp[row][column] = now
            self.freshness[row][column] = 1.0
            self.last_update_type[row][column] = "cctv"
        self.cctv_mean = total / len(self.cctv_cells)
        return values

    def _initialize_non_cctv(self, now: float) -> None:
        for row in range(self.rows):
            for column in range(self.columns):
                if (row, column) in self.cctv_cells:
                    continue
                self.current_cost[row][column] = float(self.cctv_mean)
                self.anchor_cost[row][column] = float(self.cctv_mean)
                self.anchor_timestamp[row][column] = now
                self.freshness[row][column] = 1.0
                self.last_update_type[row][column] = "initialized_cctv_mean"
        self.initialized = True

    def _decay(self, now: float) -> None:
        for row in range(self.rows):
            for column in range(self.columns):
                if (row, column) in self.cctv_cells:
                    continue
                elapsed = max(0.0, now - self.anchor_timestamp[row][column])
                freshness = max(0.0, 1.0 - elapsed / self.decay_time)
                self.freshness[row][column] = freshness
                self.current_cost[row][column] = (
                    freshness * self.anchor_cost[row][column]
                    + (1.0 - freshness) * self.cctv_mean
                )

    def _camera_pose(self, robot: dict[str, Any]) -> tuple[float, float, float]:
        fov = self.config["robot_observation"]
        x, y = map(float, robot["position_xyz_m"][:2])
        yaw = float(robot["yaw_rad"])
        forward, lateral = map(float, fov.get("camera_offset_xy_m", [0.0, 0.0]))
        camera_x = x + forward * math.cos(yaw) - lateral * math.sin(yaw)
        camera_y = y + forward * math.sin(yaw) + lateral * math.cos(yaw)
        return camera_x, camera_y, yaw + float(fov.get("camera_yaw_offset_rad", 0.0))

    def _in_fov(self, point: tuple[float, float], camera: tuple[float, float, float]) -> bool:
        dx, dy = point[0] - camera[0], point[1] - camera[1]
        distance = math.hypot(dx, dy)
        cfg = self.config["robot_observation"]
        if distance > float(cfg["max_range_m"]):
            return False
        angle = math.atan2(math.sin(math.atan2(dy, dx) - camera[2]),
                           math.cos(math.atan2(dy, dx) - camera[2]))
        return abs(angle) <= math.radians(float(cfg["horizontal_fov_deg"])) / 2.0

    def _occluded(self, point: tuple[float, float], camera: tuple[float, float, float]) -> bool:
        if not self.config["robot_observation"].get("geometric_occlusion", False):
            return False
        distance = math.hypot(point[0] - camera[0], point[1] - camera[1])
        samples = max(2, math.ceil(distance / (self.geometry.resolution_m * 0.25)))
        for index in range(1, samples):
            fraction = index / samples
            cell = self.geometry.world_to_grid(
                camera[0] + fraction * (point[0] - camera[0]),
                camera[1] + fraction * (point[1] - camera[1]),
            )
            if cell in self.obstacle_cells:
                return True
        return False

    def _visibility(self, camera: tuple[float, float, float]) -> None:
        samples = int(self.config["robot_observation"].get("overlap_samples_per_axis", 5))
        threshold = float(self.config["robot_observation"]["minimum_cell_overlap_ratio"])
        self.visibility_mask = _layer(self.rows, self.columns, False)
        self.overlap_ratio = _layer(self.rows, self.columns, 0.0)
        resolution = self.geometry.resolution_m
        for row in range(self.rows):
            for column in range(self.columns):
                visible = 0
                for subrow in range(samples):
                    for subcolumn in range(samples):
                        x = self.geometry.origin_xy[0] + (
                            column + (subcolumn + 0.5) / samples
                        ) * resolution
                        y = self.geometry.origin_xy[1] + (
                            row + (subrow + 0.5) / samples
                        ) * resolution
                        if self._in_fov((x, y), camera) and not self._occluded((x, y), camera):
                            visible += 1
                ratio = visible / (samples * samples)
                self.overlap_ratio[row][column] = ratio
                self.visibility_mask[row][column] = ratio >= threshold

    def _observe(self, robot: dict[str, Any], pedestrians: list[dict[str, Any]]) -> list[dict[str, Any]]:
        camera = self._camera_pose(robot)
        self._visibility(camera)
        cfg = self.config["robot_observation"]
        position_std = float(cfg["position_noise_std_m"])
        velocity_std = float(cfg["velocity_noise_std_m_s"])
        observations = []
        for person in pedestrians:
            true_position = tuple(map(float, person["position_xyz_m"][:2]))
            if not self._in_fov(true_position, camera) or self._occluded(true_position, camera):
                continue
            noisy_position = [
                true_position[0] + self.rng.gauss(0.0, position_std),
                true_position[1] + self.rng.gauss(0.0, position_std),
            ]
            velocity = person["velocity_xyz_m_s"]
            noisy_velocity = [
                float(velocity[0]) + self.rng.gauss(0.0, velocity_std),
                float(velocity[1]) + self.rng.gauss(0.0, velocity_std),
            ]
            cell = self.geometry.world_to_grid(*noisy_position)
            if cell is None or not self.visibility_mask[cell[0]][cell[1]]:
                continue
            observations.append({
                "id": person["id"], "position_xy_m": noisy_position,
                "velocity_xy_m_s": noisy_velocity, "observed_grid": list(cell),
            })
        return observations

    def _direct_updates(self, observations: list[dict[str, Any]], now: float,
                        count_overrides: dict[Cell, float] | None = None) -> None:
        counts: dict[Cell, int] = {}
        for observation in observations:
            cell = tuple(observation["observed_grid"])
            counts[cell] = counts.get(cell, 0) + 1
        for row in range(self.rows):
            for column in range(self.columns):
                cell = (row, column)
                if cell in self.cctv_cells or (
                        not self.visibility_mask[row][column]
                        and cell not in (count_overrides or {})):
                    continue
                value = float((count_overrides or {}).get(cell, counts.get(cell, 0)))
                self.current_cost[row][column] = value
                self.anchor_cost[row][column] = value
                self.anchor_timestamp[row][column] = now
                self.last_direct_observation_time[row][column] = now
                self.freshness[row][column] = 1.0
                self.last_update_type[row][column] = "robot_observation"

    def _intentions(self, observations: list[dict[str, Any]], now: float,
                    intervention_manager: Any | None = None,
                    pedestrians: list[dict[str, Any]] | None = None) \
            -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | list[dict[str, Any]]:
        horizon = float(self.config["intention"]["prediction_horizon_s"])
        confirmation = int(self.config["intention"]["transition_confirmation_ticks"])
        events = []
        self.intention_incoming = _layer(self.rows, self.columns, 0)
        for observation in observations:
            predicted = [
                observation["position_xy_m"][0] + observation["velocity_xy_m_s"][0] * horizon,
                observation["position_xy_m"][1] + observation["velocity_xy_m_s"][1] * horizon,
            ]
            destination = self.geometry.world_to_grid(*predicted)
            observation["predicted_position_xy_m"] = predicted
            observation["predicted_destination_grid"] = list(destination) if destination else None
            if destination is None:
                continue
            track = self.tracks.setdefault(observation["id"], IntentionTrack())
            emit = False
            if track.confirmed is None:
                track.confirmed = destination
                emit = True
            elif destination == track.confirmed:
                track.candidate, track.confirmations = None, 0
            else:
                if destination == track.candidate:
                    track.confirmations += 1
                else:
                    track.candidate, track.confirmations = destination, 1
                if track.confirmations >= confirmation:
                    track.confirmed = destination
                    track.candidate, track.confirmations = None, 0
                    emit = True
            if not emit or destination in self.cctv_cells:
                continue
            event = {
                "event_id": f"{observation['id']}:{destination[0]},{destination[1]}:{now:.6f}",
                "pedestrian_id": observation["id"], "destination_grid": list(destination),
                "contribution": 1, "source": "stage3_prediction",
            }
            events.append(event)
        intervention_logs: list[dict[str, Any]] = []
        if intervention_manager is not None:
            events, intervention_logs = intervention_manager.modify_intentions(
                now, events, observations, pedestrians
            )
        for event in events:
            destination = tuple(event["destination_grid"])
            if destination in self.cctv_cells:
                continue
            row, column = destination
            self.intention_incoming[row][column] += int(event.get("contribution", 1))
        for row in range(self.rows):
            for column in range(self.columns):
                incoming = self.intention_incoming[row][column]
                if incoming == 0 or (row, column) in self.cctv_cells:
                    continue
                self.current_cost[row][column] += float(incoming)
                self.anchor_cost[row][column] = self.current_cost[row][column]
                self.anchor_timestamp[row][column] = now
                self.freshness[row][column] = 1.0
                previous = self.last_update_type[row][column]
                self.last_update_type[row][column] = (
                    "robot_observation+intention" if previous == "robot_observation" else "intention"
                )
        if intervention_manager is None:
            return events
        return events, intervention_logs

    def tick(
        self, simulation_time_s: float, robot: dict[str, Any], pedestrians: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not self.due(simulation_time_s):
            return None
        before = [row[:] for row in self.current_cost]
        ground_truth = self._ground_truth_counts(pedestrians)
        cctv_input: list[list[float]] = [list(map(float, row)) for row in ground_truth]
        cctv_logs: list[dict[str, Any]] = []
        if self.intervention_manager is not None:
            cctv_input, cctv_logs = self.intervention_manager.modify_cctv(
                simulation_time_s, ground_truth
            )
        cctv_counts = self._update_cctv(cctv_input, simulation_time_s)
        exact_cctv_counts = {
            f"{row},{column}": int(ground_truth[row][column])
            for row, column in sorted(self.cctv_cells)
        }
        exact_cctv_mean = sum(exact_cctv_counts.values()) / len(self.cctv_cells)
        if not self.initialized:
            self._initialize_non_cctv(simulation_time_s)
        self._decay(simulation_time_s)
        observations = self._observe(robot, pedestrians)
        congestion_logs: list[dict[str, Any]] = []
        overrides: dict[Cell, float] = {}
        if self.intervention_manager is not None:
            overrides, congestion_logs = self.intervention_manager.modify_congestion(
                simulation_time_s, observations, self.visibility_mask, ground_truth
            )
        self._direct_updates(observations, simulation_time_s, overrides)
        intention_result = self._intentions(
            observations, simulation_time_s, self.intervention_manager, pedestrians
        )
        if self.intervention_manager is None:
            events = intention_result
            intention_logs: list[dict[str, Any]] = []
        else:
            events, intention_logs = intention_result
        self._advance_schedule(simulation_time_s)
        return {
            "policy": {
                "simulation_time_s": simulation_time_s,
                "costmap": [row[:] for row in self.current_cost],
                "static_obstacle_mask": self.static_obstacle_mask,
                "walkable_mask": self.walkable_mask,
                "cctv_coverage_mask": self.cctv_mask,
            },
            "diagnostics": {
                "cctv_counts": cctv_counts, "cctv_mean": self.cctv_mean,
                "cctv_ground_truth_counts": exact_cctv_counts,
                "cctv_ground_truth_mean": exact_cctv_mean,
                "observations": observations, "intention_events": events,
                "robot_visibility_mask": self.visibility_mask,
                "fov_overlap_ratio": self.overlap_ratio,
                "cost_before": before, "cost_after": [row[:] for row in self.current_cost],
                "anchor_cost": [row[:] for row in self.anchor_cost],
                "anchor_timestamp": [row[:] for row in self.anchor_timestamp],
                "last_direct_observation_time": [row[:] for row in self.last_direct_observation_time],
                "freshness": [row[:] for row in self.freshness],
                "intention_incoming_count": [row[:] for row in self.intention_incoming],
                "last_update_type": [row[:] for row in self.last_update_type],
                "intervention_events": cctv_logs + congestion_logs + intention_logs,
            },
            "privileged": {"ground_truth_count": ground_truth},
        }
