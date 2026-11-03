"""Host-side Stage 1 configuration, density, and dataset utilities.

This module deliberately has no Isaac Sim imports so configurations and saved
Ground Truth can be validated in VACaN's lightweight development environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


SPLIT_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}
REQUIRED_ENVIRONMENTS = {"office", "hospital", "warehouse"}


@dataclass(frozen=True)
class GridSpec:
    origin_xy: tuple[float, float]
    size_xy: tuple[float, float]
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        columns = math.ceil(self.size_xy[0] / self.resolution_m)
        rows = math.ceil(self.size_xy[1] / self.resolution_m)
        return rows, columns


def load_stage1_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate a Stage 1 Isaac/Pegasus YAML file."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 3:
        raise ValueError(f"{path}: expected Stage 1 schema_version 3")
    if data.get("backend") != "isaac-sim-pegasus":
        raise ValueError(f"{path}: backend must be isaac-sim-pegasus")
    environment = data.get("environment", {})
    if environment.get("kind") not in REQUIRED_ENVIRONMENTS:
        raise ValueError(f"{path}: invalid environment kind")
    asset = environment.get("asset")
    if not isinstance(asset, str) or not asset.startswith("Isaac/Environments/"):
        raise ValueError(f"{path}: environment.asset must be an official Isaac relative path")
    episode = data.get("episode", {})
    if float(episode.get("physics_dt_s", 0)) <= 0 or int(episode.get("steps", 0)) <= 0:
        raise ValueError(f"{path}: invalid episode timing")
    grid = grid_spec(data)
    if min(grid.size_xy) <= 0 or grid.resolution_m <= 0:
        raise ValueError(f"{path}: invalid Ground Truth grid")
    pedestrians = data.get("pedestrians")
    if not isinstance(pedestrians, list) or len(pedestrians) < 2:
        raise ValueError(f"{path}: at least two pedestrians are required")
    names: set[str] = set()
    for pedestrian in pedestrians:
        name = pedestrian.get("id")
        if not isinstance(name, str) or name in names:
            raise ValueError(f"{path}: pedestrian IDs must be unique strings")
        names.add(name)
        for key in ("spawn", "destinations"):
            value = pedestrian.get(key)
            if key == "spawn" and (not isinstance(value, list) or len(value) != 3):
                raise ValueError(f"{path}: {name}.{key} must be [x, y, z]")
            if key == "destinations" and (
                not isinstance(value, list)
                or len(value) < 2
                or any(not isinstance(point, list) or len(point) != 3 for point in value)
            ):
                raise ValueError(f"{path}: {name}.{key} needs at least two 3D points")
        if float(pedestrian.get("speed_m_s", 0)) <= 0:
            raise ValueError(f"{path}: {name}.speed_m_s must be positive")
    seeds = data.get("seeds", {})
    for split in SPLIT_OFFSETS:
        values = seeds.get(split)
        if not isinstance(values, list) or not values or not all(isinstance(v, int) for v in values):
            raise ValueError(f"{path}: seeds.{split} must be a nonempty integer list")
    return data


def grid_spec(config: dict[str, Any]) -> GridSpec:
    value = config["ground_truth"]["grid"]
    return GridSpec(
        origin_xy=(float(value["origin_xy"][0]), float(value["origin_xy"][1])),
        size_xy=(float(value["size_xy"][0]), float(value["size_xy"][1])),
        resolution_m=float(value["resolution_m"]),
    )


def density_grid(
    positions_xy: Iterable[tuple[float, float]], spec: GridSpec
) -> tuple[list[list[int]], list[list[float]]]:
    """Return exact Ground Truth counts and people/m² for a fixed world grid."""

    rows, columns = spec.shape
    counts = [[0 for _ in range(columns)] for _ in range(rows)]
    x0, y0 = spec.origin_xy
    for x, y in positions_xy:
        column = math.floor((x - x0) / spec.resolution_m)
        row = math.floor((y - y0) / spec.resolution_m)
        if 0 <= row < rows and 0 <= column < columns:
            counts[row][column] += 1
    cell_area = spec.resolution_m**2
    density = [[count / cell_area for count in row] for row in counts]
    return counts, density


def episode_seed(config: dict[str, Any], split: str, episode_index: int) -> int:
    """Resolve a configured independent train/validation/test episode seed."""

    if split not in SPLIT_OFFSETS:
        raise ValueError(f"unknown split {split!r}")
    values = config["seeds"][split]
    if not 0 <= episode_index < len(values):
        raise IndexError(f"episode {episode_index} is not configured for {split}")
    return int(values[episode_index])
