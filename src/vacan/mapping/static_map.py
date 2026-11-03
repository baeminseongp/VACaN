"""Static 3D voxel and aligned 2D occupancy maps for Stage 4."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dynamic_costmap import GridGeometry


FREE = np.uint8(0)
OCCUPIED = np.uint8(1)
UNKNOWN = np.uint8(2)


@dataclass(frozen=True)
class VoxelMap:
    occupancy: np.ndarray
    origin_xyz_m: tuple[float, float, float]
    resolution_m: float
    environment: str
    source_asset: str
    collision_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.occupancy.ndim != 3 or self.occupancy.dtype != np.uint8:
            raise ValueError("voxel occupancy must be a 3D uint8 array")
        if self.resolution_m <= 0:
            raise ValueError("voxel resolution must be positive")
        values = set(np.unique(self.occupancy).tolist())
        if not values <= {int(FREE), int(OCCUPIED), int(UNKNOWN)}:
            raise ValueError(f"invalid voxel occupancy values: {values}")

    @property
    def bounds_xyz_m(self) -> tuple[float, float, float, float, float, float]:
        nz, ny, nx = self.occupancy.shape
        ox, oy, oz = self.origin_xyz_m
        r = self.resolution_m
        return ox, oy, oz, ox + nx * r, oy + ny * r, oz + nz * r

    def voxel_center(self, z: int, y: int, x: int) -> tuple[float, float, float]:
        r = self.resolution_m
        return tuple(self.origin_xyz_m[i] + (index + 0.5) * r
                     for i, index in enumerate((x, y, z)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "environment": self.environment,
            "source_asset": self.source_asset,
            "world_frame": "isaac_world_enu",
            "axis_order": "zyx",
            "origin_xyz_m": list(self.origin_xyz_m),
            "resolution_m": self.resolution_m,
            "bounds_xyz_m": list(self.bounds_xyz_m),
            "shape_zyx": list(self.occupancy.shape),
            "values": {"free": 0, "occupied": 1, "unknown": 2},
            "collision_metadata": self.collision_metadata,
        }
        np.savez_compressed(path, occupancy=self.occupancy,
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "VoxelMap":
        with np.load(path, allow_pickle=False) as archive:
            occupancy = archive["occupancy"].copy()
            metadata = json.loads(str(archive["metadata_json"]))
        return cls(occupancy, tuple(metadata["origin_xyz_m"]),
                   float(metadata["resolution_m"]), metadata["environment"],
                   metadata["source_asset"], metadata.get("collision_metadata", {}))


class DepthVoxelIntegrator:
    """Fuse world-frame depth endpoints into a conservative ternary voxel map."""

    def __init__(self, bounds_xyz_m: tuple[float, float, float, float, float, float],
                 resolution_m: float) -> None:
        if resolution_m <= 0 or any(bounds_xyz_m[i] >= bounds_xyz_m[i + 3] for i in range(3)):
            raise ValueError("invalid depth voxel geometry")
        self.bounds = tuple(map(float, bounds_xyz_m))
        self.resolution_m = float(resolution_m)
        x0, y0, z0, x1, y1, z1 = self.bounds
        nx, ny, nz = (math.ceil((x1 - x0) / resolution_m),
                      math.ceil((y1 - y0) / resolution_m),
                      math.ceil((z1 - z0) / resolution_m))
        self.occupancy = np.full((nz, ny, nx), UNKNOWN, dtype=np.uint8)
        self.frames_integrated = 0
        self.rays_integrated = 0

    @property
    def origin_xyz_m(self) -> tuple[float, float, float]:
        return self.bounds[:3]

    def world_to_voxel(self, point_xyz: np.ndarray) -> tuple[int, int, int] | None:
        indices = np.floor((np.asarray(point_xyz, dtype=float) - np.asarray(self.bounds[:3]))
                           / self.resolution_m).astype(int)
        x, y, z = map(int, indices)
        if (0 <= z < self.occupancy.shape[0] and 0 <= y < self.occupancy.shape[1]
                and 0 <= x < self.occupancy.shape[2]):
            return z, y, x
        return None

    def integrate(self, camera_xyz_m: np.ndarray, endpoints_xyz_m: np.ndarray,
                  max_range_m: float, ray_stride: int = 1) -> int:
        """Ray-carve free space and mark finite in-bounds depth endpoints occupied."""
        origin = np.asarray(camera_xyz_m, dtype=float)
        endpoints = np.asarray(endpoints_xyz_m, dtype=float).reshape(-1, 3)[::max(1, int(ray_stride))]
        if not len(endpoints):
            self.frames_integrated += 1
            return 0
        finite = np.isfinite(endpoints).all(axis=1)
        distances = np.linalg.norm(endpoints - origin, axis=1)
        lower, upper = np.asarray(self.bounds[:3]), np.asarray(self.bounds[3:])
        in_bounds = ((endpoints >= lower) & (endpoints < upper)).all(axis=1)
        endpoints = endpoints[finite & (distances > 0) & (distances <= max_range_m) & in_bounds]
        if len(endpoints):
            # Many neighboring depth pixels terminate in the same 0.2 m voxel.
            # One representative ray per endpoint voxel retains occupancy while
            # avoiding thousands of redundant Python ray-carving loops.
            target_xyz = np.floor((endpoints - lower) / self.resolution_m).astype(np.int32)
            _, unique_indices = np.unique(target_xyz, axis=0, return_index=True)
            endpoints = endpoints[np.sort(unique_indices)]
        accepted = 0
        for endpoint in endpoints:
            delta = endpoint - origin
            distance = float(np.linalg.norm(delta))
            target = self.world_to_voxel(endpoint)
            if target is None:
                continue
            direction = delta / distance
            # Stop before the measured surface so carving never clears its voxel.
            free_distance = max(0.0, distance - self.resolution_m * 0.75)
            samples = max(1, math.ceil(free_distance / self.resolution_m))
            for sample in range(samples):
                point = origin + direction * (sample / samples) * free_distance
                cell = self.world_to_voxel(point)
                if cell is not None and self.occupancy[cell] != OCCUPIED:
                    self.occupancy[cell] = FREE
            self.occupancy[target] = OCCUPIED
            accepted += 1
        self.frames_integrated += 1
        self.rays_integrated += accepted
        return accepted

    def voxel_map(self, environment: str, source_asset: str,
                  metadata: dict[str, Any]) -> VoxelMap:
        return VoxelMap(self.occupancy.copy(), self.origin_xyz_m, self.resolution_m,
                        environment, source_asset, metadata)


@dataclass(frozen=True)
class OccupancyMap2D:
    raw: np.ndarray
    unknown: np.ndarray
    inflated: np.ndarray
    geometry: GridGeometry
    environment: str
    inflation_radius_m: float
    source_voxel_map: str

    def __post_init__(self) -> None:
        expected = self.geometry.shape
        if self.raw.shape != expected or self.unknown.shape != expected or self.inflated.shape != expected:
            raise ValueError(f"2D map layers must share grid shape {expected}")
        if self.raw.dtype != np.uint8 or self.inflated.dtype != np.uint8:
            raise ValueError("occupancy layers must be uint8")
        if self.unknown.dtype != np.bool_:
            raise ValueError("unknown layer must be boolean")

    @property
    def policy_obstacle(self) -> np.ndarray:
        """Unsafe mask used by the policy; unknown is never treated as free."""
        return np.maximum(self.inflated, self.unknown.astype(np.uint8))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1, "environment": self.environment,
            "world_frame": "isaac_world_enu", "axis_order": "row_y,column_x",
            "origin_xy_m": list(self.geometry.origin_xy),
            "size_xy_m": list(self.geometry.size_xy),
            "resolution_m": self.geometry.resolution_m,
            "shape_rows_columns": list(self.geometry.shape),
            "inflation_radius_m": self.inflation_radius_m,
            "source_voxel_map": self.source_voxel_map,
        }
        np.savez_compressed(path, raw_occupancy=self.raw, unknown_mask=self.unknown,
                            inflated_obstacle=self.inflated,
                            policy_obstacle=self.policy_obstacle,
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "OccupancyMap2D":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            geometry = GridGeometry(tuple(metadata["origin_xy_m"]),
                                    tuple(metadata["size_xy_m"]),
                                    float(metadata["resolution_m"]))
            return cls(archive["raw_occupancy"].copy(), archive["unknown_mask"].copy(),
                       archive["inflated_obstacle"].copy(), geometry,
                       metadata["environment"], float(metadata["inflation_radius_m"]),
                       metadata["source_voxel_map"])


def inflate_obstacles(raw: np.ndarray, resolution_m: float, radius_m: float) -> np.ndarray:
    """Euclidean configuration-space expansion without an optional SciPy dependency."""
    if radius_m < 0 or resolution_m <= 0:
        raise ValueError("invalid inflation geometry")
    result = raw.astype(np.uint8, copy=True)
    occupied = np.argwhere(raw != 0)
    radius_cells = math.ceil(radius_m / resolution_m)
    for row, column in occupied:
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                target_row, target_column = row + dr, column + dc
                if not (0 <= target_row < raw.shape[0] and 0 <= target_column < raw.shape[1]):
                    continue
                if math.hypot(dr * resolution_m, dc * resolution_m) <= radius_m + 1e-12:
                    result[target_row, target_column] = OCCUPIED
    return result


def project_voxel_map(
    voxel_map: VoxelMap,
    geometry: GridGeometry,
    min_height_m: float,
    max_height_m: float,
    inflation_radius_m: float,
    source_voxel_map: str,
) -> OccupancyMap2D:
    """Project collision-relevant voxels onto the exact Stage 3 grid geometry."""
    if max_height_m <= min_height_m:
        raise ValueError("projection height interval must be positive")
    rows, columns = geometry.shape
    raw = np.zeros((rows, columns), dtype=np.uint8)
    known = np.zeros((rows, columns), dtype=np.bool_)
    unknown_seen = np.zeros((rows, columns), dtype=np.bool_)
    r = voxel_map.resolution_m
    ox, oy, oz = voxel_map.origin_xyz_m
    for z in range(voxel_map.occupancy.shape[0]):
        height = oz + (z + 0.5) * r
        if not min_height_m <= height <= max_height_m:
            continue
        for y in range(voxel_map.occupancy.shape[1]):
            world_y = oy + (y + 0.5) * r
            for x in range(voxel_map.occupancy.shape[2]):
                cell = geometry.world_to_grid(ox + (x + 0.5) * r, world_y)
                if cell is None:
                    continue
                value = voxel_map.occupancy[z, y, x]
                if value == OCCUPIED:
                    raw[cell] = OCCUPIED
                    known[cell] = True
                elif value == FREE:
                    known[cell] = True
                else:
                    unknown_seen[cell] = True
    # A coarse cell is Unknown only when the collision map supplied no known
    # free or occupied voxel in the relevant height interval. Occupied takes
    # precedence through ``raw``; observed free takes precedence over other
    # unknown subvoxels in the same coarse cell.
    unknown = ~known
    inflated = inflate_obstacles(raw, geometry.resolution_m, inflation_radius_m)
    return OccupancyMap2D(raw, unknown, inflated, geometry, voxel_map.environment,
                          inflation_radius_m, source_voxel_map)
