"""Regular metric zones derived from walkable ROS occupancy cells."""

from __future__ import annotations

import json
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from vacan.datasets.atc.map import RosMap, occupancy_state


@dataclass(frozen=True)
class Zone:
    zone_id: str
    grid_row: int
    grid_column: int
    polygon: tuple[tuple[float, float], ...]
    polygon_area_m2: float
    walkable_area_m2: float
    walkable_fraction: float

    @property
    def centroid(self) -> tuple[float, float]:
        coordinates = self.polygon[:-1]
        return (
            sum(point[0] for point in coordinates) / len(coordinates),
            sum(point[1] for point in coordinates) / len(coordinates),
        )


@dataclass(frozen=True)
class ZoneIndex:
    ros_map: RosMap
    zone_size_m: float
    zones: tuple[Zone, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_cell", {(zone.grid_row, zone.grid_column): zone for zone in self.zones})

    def zone_for_point(self, world_x: float, world_y: float) -> Zone | None:
        local_x, local_y = self.ros_map.local_point(world_x, world_y)
        if not (0 <= local_x < self.ros_map.width * self.ros_map.resolution):
            return None
        if not (0 <= local_y < self.ros_map.height * self.ros_map.resolution):
            return None
        cell = (int(local_y // self.zone_size_m), int(local_x // self.zone_size_m))
        return self._by_cell.get(cell)  # type: ignore[attr-defined]


def generate_grid_zones(
    ros_map: RosMap,
    *,
    zone_size_m: float,
    minimum_walkable_fraction: float,
    traversable_states: tuple[str, ...] = ("FREE",),
) -> ZoneIndex:
    if zone_size_m <= 0:
        raise ValueError("zone_size_m must be positive")
    if not 0 <= minimum_walkable_fraction <= 1:
        raise ValueError("minimum_walkable_fraction must be in [0, 1]")
    cell_area = ros_map.resolution**2
    free_counts: dict[tuple[int, int], int] = {}
    total_counts: dict[tuple[int, int], int] = {}
    for image_row in range(ros_map.height):
        local_row = ros_map.height - image_row - 1
        local_y = local_row * ros_map.resolution
        zone_row = int(local_y // zone_size_m)
        offset = image_row * ros_map.width
        for column, pixel in enumerate(ros_map.pixels[offset:offset + ros_map.width]):
            zone_column = int((column * ros_map.resolution) // zone_size_m)
            key = (zone_row, zone_column)
            total_counts[key] = total_counts.get(key, 0) + 1
            if occupancy_state(pixel, ros_map) in traversable_states:
                free_counts[key] = free_counts.get(key, 0) + 1

    zones: list[Zone] = []
    for (row, column), total_count in sorted(total_counts.items()):
        free_count = free_counts.get((row, column), 0)
        walkable_fraction = free_count / total_count
        if free_count == 0 or walkable_fraction < minimum_walkable_fraction:
            continue
        x0, y0 = column * zone_size_m, row * zone_size_m
        x1 = min(x0 + zone_size_m, ros_map.width * ros_map.resolution)
        y1 = min(y0 + zone_size_m, ros_map.height * ros_map.resolution)
        local_ring = ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))
        polygon = tuple(ros_map.world_point(x, y) for x, y in local_ring)
        zones.append(Zone(
            zone_id=f"G-R{row:03d}-C{column:03d}",
            grid_row=row,
            grid_column=column,
            polygon=polygon,
            polygon_area_m2=(x1 - x0) * (y1 - y0),
            walkable_area_m2=free_count * cell_area,
            walkable_fraction=walkable_fraction,
        ))
    return ZoneIndex(ros_map=ros_map, zone_size_m=zone_size_m, zones=tuple(zones))


def filter_zones_by_trajectory_support(
    trajectory_path: Path,
    index: ZoneIndex,
    *,
    minimum_rows_per_zone: int,
) -> tuple[ZoneIndex, dict[str, int]]:
    """Limit a broad map to the empirically tracked ATC analysis region."""
    if minimum_rows_per_zone < 1:
        raise ValueError("minimum_rows_per_zone must be positive")
    counts: dict[str, int] = {}
    outside_candidate_rows = 0
    with trajectory_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            zone = index.zone_for_point(float(row["x_m"]), float(row["y_m"]))
            if zone is None:
                outside_candidate_rows += 1
            else:
                counts[zone.zone_id] = counts.get(zone.zone_id, 0) + 1
    supported = tuple(zone for zone in index.zones if counts.get(zone.zone_id, 0) >= minimum_rows_per_zone)
    return (
        ZoneIndex(ros_map=index.ros_map, zone_size_m=index.zone_size_m, zones=supported),
        {
            "candidate_zone_count": len(index.zones),
            "supported_zone_count": len(supported),
            "outside_candidate_rows": outside_candidate_rows,
            "minimum_rows_per_zone": minimum_rows_per_zone,
        },
    )


def zones_geojson(index: ZoneIndex) -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "zone_id": zone.zone_id,
                    "grid_row": zone.grid_row,
                    "grid_column": zone.grid_column,
                    "area_m2": zone.walkable_area_m2,
                    "polygon_area_m2": zone.polygon_area_m2,
                    "walkable_fraction": zone.walkable_fraction,
                    "walkable": True,
                    "semantic_type": "regular_grid",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x, y] for x, y in zone.polygon]],
                },
            }
            for zone in index.zones
        ],
    }


def write_zones(index: ZoneIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(zones_geojson(index), indent=2) + "\n", encoding="utf-8")
