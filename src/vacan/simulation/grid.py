"""Deterministic world/grid conversion and occupancy rasterization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


class GridEntity(Protocol):
    x: float
    y: float


@dataclass(frozen=True)
class GridSpec:
    width_m: float
    height_m: float
    resolution_m: float

    def __post_init__(self) -> None:
        if min(self.width_m, self.height_m, self.resolution_m) <= 0:
            raise ValueError("Grid dimensions and resolution must be positive")
        if not math.isclose(self.width_m / self.resolution_m, self.cols):
            raise ValueError("width_m must be divisible by resolution_m")
        if not math.isclose(self.height_m / self.resolution_m, self.rows):
            raise ValueError("height_m must be divisible by resolution_m")

    @property
    def cols(self) -> int:
        return int(round(self.width_m / self.resolution_m))

    @property
    def rows(self) -> int:
        return int(round(self.height_m / self.resolution_m))

    @property
    def origin(self) -> tuple[float, float]:
        return (-self.width_m / 2.0, -self.height_m / 2.0)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        origin_x, origin_y = self.origin
        col = int(math.floor((x - origin_x) / self.resolution_m))
        row = int(math.floor((y - origin_y) / self.resolution_m))
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"World coordinate outside grid: ({x}, {y})")
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Grid index outside grid: ({row}, {col})")
        origin_x, origin_y = self.origin
        return (
            origin_x + (col + 0.5) * self.resolution_m,
            origin_y + (row + 0.5) * self.resolution_m,
        )

    def rasterize(
        self,
        obstacles: Iterable[dict[str, float | str | int]],
        pedestrians: Iterable[GridEntity],
        robot: GridEntity,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        static = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for row in range(self.rows):
            for col in range(self.cols):
                x, y = self.grid_to_world(row, col)
                if any(
                    _contains(obstacle, x, y, padding=self.resolution_m / 2.0)
                    for obstacle in obstacles
                ):
                    static[row, col] = 1

        counts = np.zeros((self.rows, self.cols), dtype=np.uint16)
        for pedestrian in pedestrians:
            try:
                row, col = self.world_to_grid(pedestrian.x, pedestrian.y)
            except ValueError:
                continue
            counts[row, col] += 1

        occupancy = static.copy()
        occupancy[counts > 0] = 2
        try:
            robot_row, robot_col = self.world_to_grid(robot.x, robot.y)
            occupancy[robot_row, robot_col] = 3
        except ValueError:
            pass
        density = counts.astype(np.float32) / (self.resolution_m**2)
        return occupancy, counts, density


def _contains(
    obstacle: dict[str, float | str | int], x: float, y: float, padding: float = 0.0
) -> bool:
    shape = obstacle["shape"]
    center_x = float(obstacle["x"])
    center_y = float(obstacle["y"])
    if shape == "circle":
        radius = float(obstacle["radius"]) + padding
        return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    orientation = float(obstacle.get("orientation", 0.0))
    dx = x - center_x
    dy = y - center_y
    local_x = dx * math.cos(orientation) + dy * math.sin(orientation)
    local_y = -dx * math.sin(orientation) + dy * math.cos(orientation)
    return abs(local_x) <= float(obstacle["length"]) / 2 + padding and abs(
        local_y
    ) <= float(obstacle["width"]) / 2 + padding
