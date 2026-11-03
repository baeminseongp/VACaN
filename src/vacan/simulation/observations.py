"""Stable partial-observation interface separated from Ground Truth output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .ground_truth import GroundTruthState


@dataclass(frozen=True)
class SensorView:
    sensor_id: str
    x: float
    y: float
    orientation: float
    range_m: float
    fov_radians: float


class PartialObservationProvider:
    """Simulate geometric visibility and return no global grids or density labels."""

    def __init__(self, sensors: dict[str, Any]) -> None:
        robot = sensors["robot"]
        self._robot_range = float(robot["range_m"])
        self._robot_fov = math.radians(float(robot["fov_degrees"]))
        self._fixed = tuple(
            SensorView(
                sensor_id=str(sensor["id"]),
                x=float(sensor["x"]),
                y=float(sensor["y"]),
                orientation=math.radians(float(sensor["orientation_degrees"])),
                range_m=float(sensor["range_m"]),
                fov_radians=math.radians(float(sensor["fov_degrees"])),
            )
            for sensor in sensors.get("fixed", [])
        )

    def observe(self, state: GroundTruthState, robot_observation: Any) -> dict[str, Any]:
        robot = state.robot
        robot_view = SensorView(
            sensor_id="robot",
            x=robot.x,
            y=robot.y,
            orientation=robot.orientation,
            range_m=self._robot_range,
            fov_radians=self._robot_fov,
        )
        sources = [self._observe_from(view, state) for view in (robot_view, *self._fixed)]
        return {
            "timestamp": state.timestamp,
            "simulation_step": state.simulation_step,
            "robot_observation": robot_observation.tolist(),
            "sources": sources,
        }

    def _observe_from(self, view: SensorView, state: GroundTruthState) -> dict[str, Any]:
        detections = []
        for pedestrian in state.pedestrians:
            dx = pedestrian.x - view.x
            dy = pedestrian.y - view.y
            distance = math.hypot(dx, dy)
            bearing = _wrap(math.atan2(dy, dx) - view.orientation)
            if distance <= view.range_m and (
                view.fov_radians >= 2 * math.pi or abs(bearing) <= view.fov_radians / 2
            ):
                detections.append(
                    {
                        "track_id": pedestrian.id,
                        "relative_x": dx * math.cos(view.orientation) + dy * math.sin(view.orientation),
                        "relative_y": -dx * math.sin(view.orientation) + dy * math.cos(view.orientation),
                        "velocity_x": pedestrian.velocity_x,
                        "velocity_y": pedestrian.velocity_y,
                    }
                )
        return {"sensor_id": view.sensor_id, "pedestrians": detections}


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
