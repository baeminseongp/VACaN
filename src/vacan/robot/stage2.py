"""Host-side Stage 2 navigation configuration and controller utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SCENARIOS = {"office", "hospital", "warehouse"}


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class VelocityLimits:
    max_linear_m_s: float
    max_angular_rad_s: float
    max_linear_accel_m_s2: float
    max_angular_accel_rad_s2: float


class ActionLimiter:
    """Clip unicycle velocity and acceleration commands deterministically."""

    def __init__(self, limits: VelocityLimits) -> None:
        self.limits = limits
        self.previous = (0.0, 0.0)

    def reset(self) -> None:
        self.previous = (0.0, 0.0)

    def apply(self, action: tuple[float, float], dt: float) -> tuple[float, float]:
        requested_v = max(-self.limits.max_linear_m_s, min(self.limits.max_linear_m_s, action[0]))
        requested_w = max(-self.limits.max_angular_rad_s, min(self.limits.max_angular_rad_s, action[1]))
        previous_v, previous_w = self.previous
        max_dv = self.limits.max_linear_accel_m_s2 * dt
        max_dw = self.limits.max_angular_accel_rad_s2 * dt
        v = max(previous_v - max_dv, min(previous_v + max_dv, requested_v))
        w = max(previous_w - max_dw, min(previous_w + max_dw, requested_w))
        self.previous = (v, w)
        return self.previous


class WaypointFollower:
    """Small heading-first navigation baseline; it is not a DRL policy."""

    def __init__(
        self,
        waypoints_xy: list[tuple[float, float]],
        cruise_speed_m_s: float,
        angular_gain: float,
        waypoint_tolerance_m: float,
        forward_heading_limit_rad: float,
    ) -> None:
        if not waypoints_xy:
            raise ValueError("at least one waypoint is required")
        self.waypoints = waypoints_xy
        self.cruise_speed = cruise_speed_m_s
        self.angular_gain = angular_gain
        self.tolerance = waypoint_tolerance_m
        self.forward_heading_limit = forward_heading_limit_rad
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def action(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        while self.index < len(self.waypoints) - 1:
            target = self.waypoints[self.index]
            if math.hypot(target[0] - x, target[1] - y) > self.tolerance:
                break
            self.index += 1
        target_x, target_y = self.waypoints[self.index]
        error = wrap_angle(math.atan2(target_y - y, target_x - x) - yaw)
        angular = self.angular_gain * error
        alignment = max(0.0, 1.0 - abs(error) / self.forward_heading_limit)
        linear = self.cruise_speed * alignment if abs(error) < self.forward_heading_limit else 0.0
        return linear, angular


def load_stage2_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected Stage 2 schema_version 1")
    if data.get("backend") != "isaac-sim-pegasus-nova-carter":
        raise ValueError(f"{path}: invalid backend")
    if data.get("scenario") not in REQUIRED_SCENARIOS:
        raise ValueError(f"{path}: invalid scenario")
    robot = data.get("robot", {})
    if robot.get("asset") != "Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd":
        raise ValueError(f"{path}: official Nova Carter asset is required")
    if robot.get("wheel_joints") != ["joint_wheel_left", "joint_wheel_right"]:
        raise ValueError(f"{path}: verified Nova Carter wheel joints are required")
    if robot.get("contact_scope") != "all_rigid_bodies":
        raise ValueError(f"{path}: all Nova Carter rigid bodies must be contact-monitored")
    if float(robot.get("wheel_radius_m", 0)) <= 0 or float(robot.get("wheel_base_m", 0)) <= 0:
        raise ValueError(f"{path}: invalid wheel geometry")
    for key in ("start_xyz", "goal_xyz"):
        if not isinstance(robot.get(key), list) or len(robot[key]) != 3:
            raise ValueError(f"{path}: robot.{key} must contain three coordinates")
    episode = data.get("episode", {})
    if int(episode.get("max_steps", 0)) <= 0 or float(episode.get("goal_tolerance_m", 0)) <= 0:
        raise ValueError(f"{path}: invalid episode definition")
    if int(episode.get("reset_settle_steps", 0)) < 1:
        raise ValueError(f"{path}: episode.reset_settle_steps must be positive")
    seeds = data.get("seeds", {})
    for split in ("train", "validation", "test"):
        if not isinstance(seeds.get(split), list) or not seeds[split]:
            raise ValueError(f"{path}: seeds.{split} is required")
    return data


def velocity_limits(config: dict[str, Any]) -> VelocityLimits:
    value = config["action_limits"]
    return VelocityLimits(
        max_linear_m_s=float(value["max_linear_m_s"]),
        max_angular_rad_s=float(value["max_angular_rad_s"]),
        max_linear_accel_m_s2=float(value["max_linear_accel_m_s2"]),
        max_angular_accel_rad_s2=float(value["max_angular_accel_rad_s2"]),
    )
