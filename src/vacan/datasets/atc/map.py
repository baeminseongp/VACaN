"""ROS map_server occupancy-grid parsing and web-oriented GeoJSON conversion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RosMap:
    image_path: Path
    width: int
    height: int
    pixels: tuple[int, ...]
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    negate: bool
    occupied_thresh: float
    free_thresh: float

    def world_point(self, local_x: float, local_y: float) -> tuple[float, float]:
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        return (
            self.origin_x + cos_yaw * local_x - sin_yaw * local_y,
            self.origin_y + sin_yaw * local_x + cos_yaw * local_y,
        )

    def local_point(self, world_x: float, world_y: float) -> tuple[float, float]:
        """Invert the ROS origin transform for indexing world-space samples."""
        dx, dy = world_x - self.origin_x, world_y - self.origin_y
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        return cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy

    def bounds_polygon(self) -> list[list[float]]:
        width_m = self.width * self.resolution
        height_m = self.height * self.resolution
        points = [
            self.world_point(0, 0),
            self.world_point(width_m, 0),
            self.world_point(width_m, height_m),
            self.world_point(0, height_m),
            self.world_point(0, 0),
        ]
        return [[x, y] for x, y in points]


def _parse_yaml_subset(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, raw_value = line.partition(":")
        if not separator:
            raise ValueError(f"unsupported map YAML line: {raw_line}")
        value = raw_value.strip().strip("'\"")
        if value.startswith("[") and value.endswith("]"):
            result[key.strip()] = [float(item.strip()) for item in value[1:-1].split(",")]
        elif value.lower() in {"true", "false"}:
            result[key.strip()] = value.lower() == "true"
        else:
            try:
                result[key.strip()] = float(value)
            except ValueError:
                result[key.strip()] = value
    return result


def _token(data: bytes, index: int) -> tuple[bytes, int]:
    length = len(data)
    while index < length:
        if data[index:index + 1] == b"#":
            newline = data.find(b"\n", index)
            index = length if newline < 0 else newline + 1
        elif data[index:index + 1].isspace():
            index += 1
        else:
            break
    start = index
    while index < length and not data[index:index + 1].isspace():
        index += 1
    if start == index:
        raise ValueError("unexpected end of PGM header")
    return data[start:index], index


def _read_pgm(path: Path) -> tuple[int, int, tuple[int, ...]]:
    data = path.read_bytes()
    magic, index = _token(data, 0)
    width_token, index = _token(data, index)
    height_token, index = _token(data, index)
    max_token, index = _token(data, index)
    width, height, maximum = int(width_token), int(height_token), int(max_token)
    if maximum <= 0 or maximum > 255:
        raise ValueError("only 8-bit PGM maps are supported")
    count = width * height
    if magic == b"P2":
        values: list[int] = []
        while len(values) < count:
            value, index = _token(data, index)
            values.append(round(int(value) * 255 / maximum))
        return width, height, tuple(values)
    if magic != b"P5":
        raise ValueError(f"unsupported PGM magic {magic!r}")
    if index >= len(data) or not data[index:index + 1].isspace():
        raise ValueError("binary PGM header is not terminated")
    if data[index:index + 2] == b"\r\n":
        index += 2
    else:
        index += 1
    pixels = data[index:index + count]
    if len(pixels) != count:
        raise ValueError("binary PGM pixel count does not match dimensions")
    if maximum == 255:
        return width, height, tuple(pixels)
    return width, height, tuple(round(value * 255 / maximum) for value in pixels)


def load_ros_map(yaml_path: Path) -> RosMap:
    yaml_path = yaml_path.resolve()
    config = _parse_yaml_subset(yaml_path)
    required = {"image", "resolution", "origin", "negate", "occupied_thresh", "free_thresh"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"map YAML is missing: {', '.join(missing)}")
    image_path = (yaml_path.parent / str(config["image"])).resolve()
    width, height, pixels = _read_pgm(image_path)
    origin = config["origin"]
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError("origin must contain [x, y, yaw]")
    return RosMap(
        image_path=image_path,
        width=width,
        height=height,
        pixels=pixels,
        resolution=float(config["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        negate=bool(int(config["negate"])) if isinstance(config["negate"], float) else bool(config["negate"]),
        occupied_thresh=float(config["occupied_thresh"]),
        free_thresh=float(config["free_thresh"]),
    )


def occupancy_state(pixel: int, ros_map: RosMap) -> str:
    probability = pixel / 255 if ros_map.negate else (255 - pixel) / 255
    if probability > ros_map.occupied_thresh:
        return "OCCUPIED"
    if probability < ros_map.free_thresh:
        return "FREE"
    return "UNKNOWN"


def _runs(states: Iterable[str]) -> Iterable[tuple[int, int, str]]:
    values = list(states)
    if not values:
        return
    start = 0
    current = values[0]
    for index, state in enumerate(values[1:], start=1):
        if state != current:
            yield start, index - 1, current
            start, current = index, state
    yield start, len(values) - 1, current


def ros_map_to_geojson(ros_map: RosMap) -> dict[str, object]:
    features: list[dict[str, object]] = []
    for row in range(ros_map.height):
        offset = row * ros_map.width
        states = [occupancy_state(value, ros_map) for value in ros_map.pixels[offset:offset + ros_map.width]]
        for start, end, state in _runs(states):
            x0, x1 = start * ros_map.resolution, (end + 1) * ros_map.resolution
            y0 = (ros_map.height - row - 1) * ros_map.resolution
            y1 = y0 + ros_map.resolution
            ring = [
                ros_map.world_point(x0, y0), ros_map.world_point(x1, y0),
                ros_map.world_point(x1, y1), ros_map.world_point(x0, y1),
                ros_map.world_point(x0, y0),
            ]
            features.append({
                "type": "Feature",
                "properties": {"occupancy": state, "image_row": row, "column_start": start, "column_end": end},
                "geometry": {"type": "Polygon", "coordinates": [[[x, y] for x, y in ring]]},
            })
    return {"type": "FeatureCollection", "features": features}


def write_map_outputs(yaml_path: Path, output_directory: Path) -> RosMap:
    ros_map = load_ros_map(yaml_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    geojson = ros_map_to_geojson(ros_map)
    (output_directory / "occupancy_runs.geojson").write_text(
        json.dumps(geojson, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "coordinate_reference": "ATC_LOCAL_METRIC",
        "width_cells": ros_map.width,
        "height_cells": ros_map.height,
        "resolution_m": ros_map.resolution,
        "origin": [ros_map.origin_x, ros_map.origin_y, ros_map.origin_yaw],
        "bounds_polygon": ros_map.bounds_polygon(),
        "source_image": ros_map.image_path.name,
    }
    (output_directory / "map.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return ros_map
