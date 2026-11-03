"""Privileged, timestamped Stage 1 simulation-state recording."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PedestrianState:
    id: int
    kind: str
    x: float
    y: float
    orientation: float
    velocity_x: float
    velocity_y: float
    speed: float
    goal_x: float | None
    goal_y: float | None
    flow_direction: str | None


@dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    orientation: float
    velocity_x: float
    velocity_y: float
    angular_velocity: float
    action: tuple[float, float]
    goal_x: float
    goal_y: float
    goal_orientation: float


@dataclass(frozen=True)
class GroundTruthState:
    scenario: str
    timestamp: str
    simulation_step: int
    simulation_time_s: float
    robot: RobotState
    pedestrians: tuple[PedestrianState, ...]
    static_obstacles: tuple[dict[str, Any], ...]
    interactions: dict[str, list[list[int]]]
    grid: dict[str, Any]
    congestion_zones: dict[str, dict[str, float | int]]
    collision: dict[str, bool]
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundTruthRecorder:
    """Write privileged state to JSONL and flat entity trajectories to CSV."""

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._json_path = self.output_directory / "ground_truth.jsonl"
        self._trajectory_path = self.output_directory / "trajectories.csv"
        self._json_stream = self._json_path.open("w", encoding="utf-8")
        self._trajectory_stream = self._trajectory_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._trajectory_stream)
        self._writer.writerow(
            ["timestamp", "step", "entity_type", "entity_id", "x", "y", "orientation", "vx", "vy"]
        )
        self._digest = hashlib.sha256()
        self._steps = 0
        self._pedestrian_start: dict[int, tuple[float, float]] = {}
        self._pedestrian_max_displacement: dict[int, float] = {}
        self._zone_counts: dict[str, list[int]] = {}

    @property
    def digest(self) -> str:
        return self._digest.hexdigest()

    def record(self, state: GroundTruthState) -> None:
        payload = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
        self._json_stream.write(payload + "\n")
        self._json_stream.flush()
        self._digest.update(payload.encode("utf-8"))
        self._steps += 1
        robot = state.robot
        self._writer.writerow(
            [state.timestamp, state.simulation_step, "robot", 0, robot.x, robot.y, robot.orientation, robot.velocity_x, robot.velocity_y]
        )
        for pedestrian in state.pedestrians:
            self._writer.writerow(
                [state.timestamp, state.simulation_step, "pedestrian", pedestrian.id, pedestrian.x, pedestrian.y, pedestrian.orientation, pedestrian.velocity_x, pedestrian.velocity_y]
            )
            start = self._pedestrian_start.setdefault(pedestrian.id, (pedestrian.x, pedestrian.y))
            displacement = ((pedestrian.x - start[0]) ** 2 + (pedestrian.y - start[1]) ** 2) ** 0.5
            self._pedestrian_max_displacement[pedestrian.id] = max(
                displacement, self._pedestrian_max_displacement.get(pedestrian.id, 0.0)
            )
        for zone_name, zone in state.congestion_zones.items():
            self._zone_counts.setdefault(zone_name, []).append(int(zone["pedestrian_count"]))

    def summary(self) -> dict[str, Any]:
        return {
            "recorded_states": self._steps,
            "sha256": self.digest,
            "max_pedestrian_displacement_m": max(
                self._pedestrian_max_displacement.values(), default=0.0
            ),
            "zone_count_ranges": {
                name: {"min": min(values), "max": max(values), "change": max(values) - min(values)}
                for name, values in self._zone_counts.items()
            },
        }

    def close(self) -> None:
        self._json_stream.close()
        self._trajectory_stream.close()

    def __enter__(self) -> "GroundTruthRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_ground_truth(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)
