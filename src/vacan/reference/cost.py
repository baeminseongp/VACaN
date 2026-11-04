"""One shared navigation cost evaluator for DRL and reference executions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CostBreakdown:
    travel_time_s: float
    path_length_m: float
    collision_events: int
    social_violation_integral_s: float
    smoothness: float
    terminal_failure: float
    weighted_time: float
    weighted_length: float
    weighted_collision: float
    weighted_social: float
    weighted_smoothness: float
    weighted_terminal_failure: float
    trajectory_efficiency: float
    context_penalty: float
    safety_penalty: float
    total: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class SharedNavigationCost:
    def __init__(self, config: dict[str, Any]) -> None:
        self.objective = str(config.get("objective", "composite_navigation"))
        if self.objective not in {"composite_navigation", "trajectory_efficiency"}:
            raise ValueError(f"unsupported navigation cost objective {self.objective}")
        self.weights = {key: float(value) for key, value in config["weights"].items()}
        self.personal_space = float(config["personal_space_m"])
        self.max_linear = float(config["normalization"]["max_linear_m_s"])
        self.max_angular = float(config["normalization"]["max_angular_rad_s"])
        self.failure_penalty = float(config["terminal_failure_penalty"])

    def evaluate(self, samples: list[dict[str, Any]], *, goal_reached: bool) -> CostBreakdown:
        if not samples:
            raise ValueError("navigation cost requires at least one trajectory sample")
        times = np.asarray([float(sample["simulation_time_s"]) for sample in samples])
        if np.any(np.diff(times) < -1e-9):
            raise ValueError("trajectory timestamps must be monotonic")
        positions = np.asarray([sample["position_xy_m"] for sample in samples], dtype=float)
        actions = np.asarray([sample.get("action", (0.0, 0.0)) for sample in samples], dtype=float)
        elapsed = float(times[-1] - times[0])
        length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        collision_flags = [bool(sample.get("collision", False)) for sample in samples]
        collisions = int(sum(flag and (index == 0 or not collision_flags[index - 1])
                             for index, flag in enumerate(collision_flags)))
        social = 0.0
        for index in range(1, len(samples)):
            distance = samples[index].get("nearest_person_distance_m")
            if distance is not None:
                severity = max(0.0, self.personal_space - float(distance)) / self.personal_space
                social += severity * float(times[index] - times[index - 1])
        smoothness = 0.0
        if len(actions) > 1:
            changes = np.abs(np.diff(actions, axis=0))
            smoothness = float((changes[:, 0] / self.max_linear
                                + changes[:, 1] / self.max_angular).sum())
        failure = 0.0 if goal_reached else self.failure_penalty
        weighted = {
            "time": self.weights["time"] * elapsed,
            "length": self.weights["length"] * length,
            "collision": self.weights["collision"] * collisions,
            "social": self.weights["social"] * social,
            "smoothness": self.weights["smoothness"] * smoothness,
            "failure": self.weights["terminal_failure"] * failure,
        }
        trajectory_efficiency = (
            weighted["time"] + weighted["length"] + weighted["smoothness"]
        )
        context_penalty = weighted["social"]
        safety_penalty = weighted["collision"] + weighted["failure"]
        return CostBreakdown(
            elapsed, length, collisions, social, smoothness, failure,
            weighted["time"], weighted["length"], weighted["collision"],
            weighted["social"], weighted["smoothness"], weighted["failure"],
            float(trajectory_efficiency), float(context_penalty), float(safety_penalty),
            float(trajectory_efficiency + context_penalty + safety_penalty),
        )
