"""Stage 5 Recurrent PPO policy and reproducibility utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VacanFeatureExtractor(BaseFeaturesExtractor):
    """Encode the two map channels and robot vector without privileged inputs."""

    def __init__(self, observation_space: gym.spaces.Dict,
                 spatial_features: int = 128, robot_features: int = 32) -> None:
        if set(observation_space.spaces) != {"map", "robot"}:
            raise ValueError("VACaN policy observation must contain only map and robot")
        map_shape = observation_space["map"].shape
        robot_shape = observation_space["robot"].shape
        if map_shape is None or map_shape[0] != 2:
            raise ValueError("map observation must have exactly two channels")
        if robot_shape != (8,):
            raise ValueError("robot observation must contain eight Stage 4 values")
        super().__init__(observation_space, spatial_features + robot_features)
        self.spatial = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, kernel_size=5, stride=2), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2), torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            flattened = int(self.spatial(torch.zeros((1, *map_shape))).shape[1])
        self.spatial_projection = torch.nn.Sequential(
            torch.nn.Linear(flattened, spatial_features), torch.nn.ReLU()
        )
        self.robot = torch.nn.Sequential(
            torch.nn.Linear(8, robot_features), torch.nn.ReLU(),
            torch.nn.Linear(robot_features, robot_features), torch.nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        spatial = self.spatial_projection(self.spatial(observations["map"].float()))
        robot = self.robot(observations["robot"].float())
        return torch.cat((spatial, robot), dim=1)


def recurrent_policy_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Translate validated YAML policy values into sb3-contrib arguments."""
    policy = config["policy"]
    return {
        "features_extractor_class": VacanFeatureExtractor,
        "features_extractor_kwargs": {
            "spatial_features": int(policy["spatial_features"]),
            "robot_features": int(policy["robot_features"]),
        },
        "lstm_hidden_size": int(policy["recurrent_hidden_size"]),
        "n_lstm_layers": int(policy["recurrent_layers"]),
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "net_arch": {"pi": list(policy["actor_layers"]),
                     "vf": list(policy["critic_layers"])},
    }


@dataclass(frozen=True)
class DeterministicPrediction:
    action: np.ndarray
    hidden_state: tuple[np.ndarray, np.ndarray]


def predict_deterministic(model: Any, observation: dict[str, np.ndarray],
                          hidden_state: tuple[np.ndarray, np.ndarray] | None,
                          episode_start: np.ndarray | bool) -> DeterministicPrediction:
    """Stable explicit recurrent-state interface used by later replay stages."""
    starts = np.asarray([episode_start], dtype=bool) if np.ndim(episode_start) == 0 else np.asarray(
        episode_start, dtype=bool
    )
    action, next_state = model.predict(
        observation, state=hidden_state, episode_start=starts, deterministic=True
    )
    if next_state is None or len(next_state) != 2:
        raise RuntimeError("recurrent policy did not return an LSTM hidden state")
    return DeterministicPrediction(np.asarray(action, dtype=np.float32), next_state)


class DecisionLogger:
    """JSONL logger separating policy-visible records from privileged evaluation data."""

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.policy_stream = (run_dir / "decisions.jsonl").open("a", encoding="utf-8")
        self.privileged_stream = (run_dir / "privileged-ground-truth.jsonl").open(
            "a", encoding="utf-8"
        )

    def write(self, policy_record: dict[str, Any], privileged_record: dict[str, Any]) -> None:
        self.policy_stream.write(json.dumps(policy_record, separators=(",", ":"), sort_keys=True) + "\n")
        self.privileged_stream.write(
            json.dumps(privileged_record, separators=(",", ":"), sort_keys=True) + "\n"
        )
        self.policy_stream.flush()
        self.privileged_stream.flush()

    def close(self) -> None:
        self.policy_stream.close()
        self.privileged_stream.close()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
