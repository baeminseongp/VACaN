"""Validated iCrowdNav-inspired Stage 1 scenario configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ScenarioDefinition:
    path: Path
    name: str
    seed: int
    start_timestamp: datetime
    time_step_s: float
    episode_length: int
    grid_resolution_m: float
    map_width_m: float
    map_height_m: float
    robot_start: tuple[float, float, float]
    robot_goal: tuple[float, float]
    robot_radius: float
    pedestrian_radius: float
    pedestrian_max_speed: float
    congestion_zones: dict[str, tuple[float, float, float, float]]
    sensors: dict[str, Any]
    pedestrian_flows: tuple[dict[str, Any], ...]
    obstacles: tuple[dict[str, Any], ...]
    raw: dict[str, Any]


def load_scenario(path: str | Path) -> ScenarioDefinition:
    scenario_path = Path(path).resolve()
    with scenario_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("scenario"), dict):
        raise ValueError(f"Missing scenario metadata: {scenario_path}")
    cfg = raw["scenario"]
    timestamp = datetime.fromisoformat(str(cfg["start_timestamp"]))
    if timestamp.tzinfo is None:
        raise ValueError("start_timestamp must include a timezone")
    map_cfg = cfg["map"]
    episode = cfg["episode"]
    robot = cfg["robot"]
    pedestrians = cfg["pedestrians"]
    flows = tuple(pedestrians["flows"])
    if int(pedestrians["count"]) != len(flows):
        raise ValueError(f"{cfg['name']}: pedestrian count does not match flows")
    zones = {name: tuple(float(v) for v in bounds) for name, bounds in cfg["congestion_zones"].items()}
    for name, bounds in zones.items():
        if len(bounds) != 4 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError(f"Invalid congestion zone {name}: {bounds}")
    obstacles = tuple(dict(item) for item in cfg.get("static_obstacles", []))
    for obstacle in obstacles:
        if obstacle.get("shape") not in {"rectangle", "circle"}:
            raise ValueError(f"Unsupported obstacle shape: {obstacle}")
    return ScenarioDefinition(
        path=scenario_path,
        name=str(cfg["name"]),
        seed=int(cfg["seed"]),
        start_timestamp=timestamp,
        time_step_s=float(episode["time_step_s"]),
        episode_length=int(episode["max_steps"]),
        grid_resolution_m=float(map_cfg["resolution_m"]),
        map_width_m=float(map_cfg["width_m"]),
        map_height_m=float(map_cfg["height_m"]),
        robot_start=tuple(float(v) for v in robot["start"]),
        robot_goal=tuple(float(v) for v in robot["goal"]),
        robot_radius=float(robot["radius_m"]),
        pedestrian_radius=float(pedestrians["radius_m"]),
        pedestrian_max_speed=float(pedestrians["max_speed_m_s"]),
        congestion_zones=zones,
        sensors=dict(cfg["sensors"]),
        pedestrian_flows=flows,
        obstacles=obstacles,
        raw=raw,
    )
