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
from sb3_contrib.common.recurrent.type_aliases import RNNStates
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


@dataclass(frozen=True)
class RecurrentDiagnosticState:
    """Actor and critic recurrent states required for faithful value logging."""

    actor: tuple[np.ndarray, np.ndarray]
    critic: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class DiagnosticPrediction:
    """One deterministic recurrent inference result with analysis diagnostics."""

    action: np.ndarray
    recurrent_state: RecurrentDiagnosticState
    value_estimate: float
    action_log_probability: float
    action_mean: np.ndarray
    action_std: np.ndarray


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


def predict_deterministic_with_diagnostics(
    model: Any,
    observation: dict[str, np.ndarray],
    recurrent_state: RecurrentDiagnosticState | None,
    episode_start: np.ndarray | bool,
) -> DiagnosticPrediction:
    """Run inference while retaining the critic state and policy statistics."""
    policy = model.policy
    policy.set_training_mode(False)
    obs_tensor, vectorized = policy.obs_to_tensor(observation)
    if isinstance(obs_tensor, dict):
        n_envs = int(next(iter(obs_tensor.values())).shape[0])
    else:
        n_envs = int(obs_tensor.shape[0])
    starts = np.asarray([episode_start], dtype=bool) if np.ndim(episode_start) == 0 else np.asarray(
        episode_start, dtype=bool
    )
    if starts.shape != (n_envs,):
        raise ValueError(f"episode_start shape {starts.shape} does not match {n_envs} environments")
    if recurrent_state is None:
        shape = policy.lstm_hidden_state_shape
        zeros = np.concatenate([np.zeros(shape, dtype=np.float32) for _ in range(n_envs)], axis=1)
        recurrent_state = RecurrentDiagnosticState(
            actor=(zeros.copy(), zeros.copy()), critic=(zeros.copy(), zeros.copy())
        )

    def tensors(state: tuple[np.ndarray, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(torch.as_tensor(item, dtype=torch.float32, device=policy.device)
                     for item in state)  # type: ignore[return-value]

    torch_states = RNNStates(pi=tensors(recurrent_state.actor),
                             vf=tensors(recurrent_state.critic))
    starts_tensor = torch.as_tensor(starts, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        actions, values, log_probabilities, next_states = policy.forward(
            obs_tensor, torch_states, starts_tensor, deterministic=True
        )
        distribution, _ = policy.get_distribution(obs_tensor, torch_states.pi, starts_tensor)
        means = distribution.distribution.mean
        standard_deviations = distribution.distribution.stddev

    action = actions.cpu().numpy()
    if isinstance(policy.action_space, gym.spaces.Box):
        if policy.squash_output:
            action = policy.unscale_action(action)
        else:
            action = np.clip(action, policy.action_space.low, policy.action_space.high)
    if not vectorized:
        action = action.squeeze(axis=0)
        means = means.squeeze(0)
        standard_deviations = standard_deviations.squeeze(0)

    def arrays(state: tuple[torch.Tensor, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        return tuple(item.detach().cpu().numpy() for item in state)  # type: ignore[return-value]

    return DiagnosticPrediction(
        action=np.asarray(action, dtype=np.float32),
        recurrent_state=RecurrentDiagnosticState(
            actor=arrays(next_states.pi), critic=arrays(next_states.vf)
        ),
        value_estimate=float(values.reshape(-1)[0].cpu()),
        action_log_probability=float(log_probabilities.reshape(-1)[0].cpu()),
        action_mean=np.asarray(means.cpu(), dtype=np.float32),
        action_std=np.asarray(standard_deviations.cpu(), dtype=np.float32),
    )


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
