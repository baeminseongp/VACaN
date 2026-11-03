"""Gymnasium boundary joining Stage 2 physics, Stage 3 costmap, and Stage 4 RL logic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from vacan.mapping.dynamic_costmap import DynamicCostmap
from vacan.mapping.static_map import OccupancyMap2D
from vacan.rl.stage4 import (
    NavigationReward,
    NormalizedActionAdapter,
    ObservationBuilder,
    ObservationSpec,
)
from vacan.robot.stage2 import VelocityLimits


class PhysicalNavigation(Protocol):
    def reset(self, seed: int, scenario: str) -> dict[str, Any]: ...
    def step(self, action: tuple[float, float]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...
    def pedestrian_states(self) -> list[dict[str, Any]]: ...


class VacanNavigationEnv(gym.Env):
    """Single physical environment; no simulator Ground Truth enters observations."""

    metadata = {"render_modes": []}

    def __init__(self, physical: PhysicalNavigation, scenario: str, seed: int,
                 static_map_path: Path, stage3_config: dict[str, Any],
                 stage4_config: dict[str, Any], velocity_limits: VelocityLimits) -> None:
        super().__init__()
        self.physical = physical
        self.scenario = scenario
        self.base_seed = int(seed)
        self.stage3_config = stage3_config
        self.stage4 = stage4_config
        self.static_map = OccupancyMap2D.load(static_map_path)
        observation = stage4_config["observation"]
        self.builder = ObservationBuilder(
            self.static_map,
            ObservationSpec(
                int(observation["crop_rows"]), int(observation["crop_columns"]),
                bool(observation["rotate_with_robot"]), float(observation["crowd_clip"]),
                float(observation["max_goal_distance_m"]),
                velocity_limits.max_linear_m_s, velocity_limits.max_angular_rad_s,
            ),
            dynamic_geometry=stage4_config["shared_grid"],
        )
        self.control_dt = 1.0 / float(stage4_config["frequencies"]["control_hz"])
        self.physics_dt = 1.0 / float(stage4_config["frequencies"]["physics_hz"])
        self.control_repeat = round(self.control_dt / self.physics_dt)
        if not math.isclose(self.control_repeat * self.physics_dt, self.control_dt, abs_tol=1e-9):
            raise ValueError("control frequency must be an integer divisor of physics frequency")
        self.action_adapter = NormalizedActionAdapter(velocity_limits, self.control_dt)
        self.reward_evaluator = NavigationReward(stage4_config["reward"])
        rows, columns = int(observation["crop_rows"]), int(observation["crop_columns"])
        self.observation_space = gym.spaces.Dict({
            "map": gym.spaces.Box(0.0, 1.0, shape=(2, rows, columns), dtype=np.float32),
            "robot": gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.episode_index = 0
        self.previous_action = (0.0, 0.0)
        self.previous_distance = 0.0
        self.episode_social_violation_steps = 0
        self.episode_min_person_distance_m: float | None = None
        self.costmap: DynamicCostmap | None = None
        self.latest_costmap_record: dict[str, Any] | None = None

    @property
    def goal_xy(self) -> tuple[float, float]:
        goal = self.stage4["runtime_goal_xyz"]
        return float(goal[0]), float(goal[1])

    def _tick_costmap(self, observation: dict[str, Any]) -> None:
        assert self.costmap is not None
        now = float(observation["simulation_step"]) * self.physics_dt
        record = self.costmap.tick(now, observation["robot"], self.physical.pedestrian_states())
        if record is not None:
            self.latest_costmap_record = record

    def _policy_observation(self, physical_observation: dict[str, Any]) -> dict[str, np.ndarray]:
        assert self.costmap is not None
        return self.builder.build(
            np.asarray(self.costmap.current_cost, dtype=np.float32),
            physical_observation["robot"], self.goal_xy, self.previous_action,
        )

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        episode_seed = self.base_seed + self.episode_index if seed is None else int(seed)
        self.episode_index += 1
        physical_observation = self.physical.reset(episode_seed, self.scenario)
        self.action_adapter.reset()
        self.previous_action = (0.0, 0.0)
        self.previous_distance = float(physical_observation["goal_distance_m"])
        self.episode_social_violation_steps = 0
        self.episode_min_person_distance_m = None
        self.costmap = DynamicCostmap(self.stage3_config, episode_seed)
        self.latest_costmap_record = None
        self._tick_costmap(physical_observation)
        observation = self._policy_observation(physical_observation)
        return observation, {"seed": episode_seed, "scenario": self.scenario}

    def step(self, action: np.ndarray):
        applied = self.action_adapter.apply(action)
        collision = False
        terminated = truncated = False
        physical_observation: dict[str, Any] | None = None
        physics_info: dict[str, Any] = {}
        for _ in range(self.control_repeat):
            physical_observation, _, step_terminated, step_truncated, physics_info = self.physical.step(applied)
            collision |= bool(physics_info.get("collision", False))
            self._tick_costmap(physical_observation)
            terminated |= bool(step_terminated)
            truncated |= bool(step_truncated)
            if terminated or truncated:
                break
        assert physical_observation is not None
        distance = float(physical_observation["goal_distance_m"])
        goal = bool(physics_info.get("goal_reached", False))
        pedestrians = self.physical.pedestrian_states()
        position = np.asarray(physical_observation["robot"]["position_xyz_m"][:2], dtype=float)
        nearest = min(
            (float(np.linalg.norm(position - np.asarray(p["position_xyz_m"][:2], dtype=float)))
             for p in pedestrians), default=None,
        )
        if nearest is not None:
            if self.episode_min_person_distance_m is None:
                self.episode_min_person_distance_m = nearest
            else:
                self.episode_min_person_distance_m = min(
                    self.episode_min_person_distance_m, nearest
                )
            if nearest < self.reward_evaluator.c["personal_space_m"]:
                self.episode_social_violation_steps += 1
        reward, reward_terms = self.reward_evaluator.compute(
            self.previous_distance, distance, goal, collision, nearest, applied, self.previous_action
        )
        self.previous_distance = distance
        self.previous_action = applied
        observation = self._policy_observation(physical_observation)
        timeout = bool(physics_info.get("timeout", False) or truncated)
        if goal:
            termination_reason = "goal"
        elif collision and terminated:
            termination_reason = "collision"
        elif timeout:
            termination_reason = "timeout"
        else:
            termination_reason = "running"
        info = {
            "reward_terms": reward_terms, "applied_action": list(applied),
            "goal_reached": goal, "collision": collision,
            "timeout": timeout, "termination_reason": termination_reason,
            "goal_distance_m": distance, "nearest_person_distance_m": nearest,
            "episode_min_person_distance_m": self.episode_min_person_distance_m,
            "episode_social_violation_steps": self.episode_social_violation_steps,
            "path_length_m": float(physics_info.get("path_length_m", 0.0)),
            "simulation_step": physical_observation["simulation_step"],
            "simulation_time_s": physical_observation["simulation_step"] * self.physics_dt,
            "robot_state": physical_observation["robot"],
            "goal_xy_m": list(self.goal_xy),
            "privileged_pedestrians": pedestrians,
            "costmap_diagnostics": None if self.latest_costmap_record is None
            else self.latest_costmap_record["diagnostics"],
        }
        return observation, reward, terminated, truncated, info
