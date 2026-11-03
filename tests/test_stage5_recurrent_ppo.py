from __future__ import annotations

from pathlib import Path
import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO

from vacan.rl.stage5 import VacanFeatureExtractor, predict_deterministic, recurrent_policy_kwargs
from vacan.rl.stage5_config import load_stage5_config
from vacan.rl.curriculum import resolve_pedestrian_population
from vacan.rl.policy_registry import (
    checkpoint_metadata, validate_checkpoint_scenario, write_policy_metadata,
)
import yaml


ROOT = Path(__file__).resolve().parents[1]


class TinyVacanEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict({
            "map": gym.spaces.Box(0, 1, (2, 61, 61), np.float32),
            "robot": gym.spaces.Box(-1, 1, (8,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        self.steps = 0

    def _observation(self):
        return {"map": np.zeros((2, 61, 61), np.float32),
                "robot": np.zeros(8, np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        self.steps += 1
        return self._observation(), 0.1, self.steps == 3, False, {}


@pytest.fixture(scope="module")
def config():
    return load_stage5_config(ROOT / "configs/stage5/recurrent_ppo.yaml")


def test_config_preserves_observation_and_decay(config):
    assert config["policy_scope"] == "single_environment"
    assert config["observation"]["map_channels"] == ["static_obstacle", "dynamic_crowd"]
    assert config["observation"]["freshness_channel"] is False
    assert config["observation"]["privileged_ground_truth"] is False
    assert config["costmap_decay_s"] == 300.0
    assert config["num_envs"] == 1
    phase4_config = config["curriculum"]["phases"][-1]
    phase4 = phase4_config["stage3_overrides"]["robot_observation"]
    assert phase4_config["pedestrian_count"] == 12
    assert phase4_config.get("reward_overrides", {}) == {}
    assert "horizontal_fov_deg" not in phase4
    assert "max_range_m" not in phase4
    assert phase4["position_noise_std_m"] == 0.30
    assert phase4["velocity_noise_std_m_s"] == 0.20
    assert phase4["minimum_cell_overlap_ratio"] == 0.40


def test_phase4_population_reuses_validated_routes_with_separated_spawns():
    stage1 = yaml.safe_load((ROOT / "configs/scenarios/office_lobby.yaml").read_text())
    routes = json.loads((
        ROOT / "artifacts/stage1-isaac/train/office/episode-000/resolved_routes.json"
    ).read_text())
    resolved_stage1, resolved_routes = resolve_pedestrian_population(stage1, routes, 12)
    assert len(resolved_stage1["pedestrians"]) == 12
    assert len(resolved_routes["pedestrians"]) == 12
    ids = [person["id"] for person in resolved_stage1["pedestrians"]]
    assert len(ids) == len(set(ids))
    spawns = [resolved_routes["pedestrians"][identifier]["spawn_xyz_m"] for identifier in ids]
    for index, first in enumerate(spawns):
        for second in spawns[index + 1:]:
            assert np.linalg.norm(np.asarray(first[:2]) - np.asarray(second[:2])) >= 0.8 - 1e-9
    for identifier in ids[6:]:
        cached = resolved_routes["pedestrians"][identifier]
        assert cached["initial_path_xyz_m"][0] == cached["spawn_xyz_m"]
        assert np.allclose(
            cached["initial_path_xyz_m"][-1], cached["destinations_xyz_m"][0], atol=1e-6
        )


def test_feature_extractor_has_separate_map_and_robot_encoders():
    space = TinyVacanEnv().observation_space
    extractor = VacanFeatureExtractor(space, 64, 16)
    output = extractor({"map": torch.zeros((2, 2, 61, 61)),
                        "robot": torch.zeros((2, 8))})
    assert output.shape == (2, 80)
    assert torch.isfinite(output).all()


def test_feature_extractor_rejects_freshness_or_ground_truth():
    space = TinyVacanEnv().observation_space
    bad = gym.spaces.Dict({**space.spaces,
                           "freshness": gym.spaces.Box(0, 1, (61, 61), np.float32)})
    with pytest.raises(ValueError, match="only map and robot"):
        VacanFeatureExtractor(bad)


def test_recurrent_forward_hidden_reset_and_reload(config, tmp_path):
    env = TinyVacanEnv()
    kwargs = recurrent_policy_kwargs(config)
    model = RecurrentPPO(
        "MultiInputLstmPolicy", env, n_steps=8, batch_size=4, n_epochs=1,
        learning_rate=1e-4, policy_kwargs=kwargs, seed=17, device="cpu", verbose=0,
    )
    observation, _ = env.reset(seed=17)
    first = predict_deterministic(model, observation, None, True)
    continued = predict_deterministic(model, observation, first.hidden_state, False)
    reset = predict_deterministic(model, observation, continued.hidden_state, True)
    assert first.action.shape == (2,)
    assert np.isfinite(first.action).all()
    assert first.hidden_state[0].shape == (1, 1, 128)
    assert np.allclose(first.action, reset.action)
    model.learn(total_timesteps=8)
    checkpoint = tmp_path / "policy"
    model.save(checkpoint)
    loaded = RecurrentPPO.load(checkpoint, env=env, device="cpu")
    expected = predict_deterministic(model, observation, None, True)
    actual = predict_deterministic(loaded, observation, None, True)
    assert np.allclose(expected.action, actual.action)
    assert np.isfinite(np.asarray(model.logger.name_to_value.get("train/loss", 0.0))).all()


def test_environment_specific_policy_metadata_rejects_cross_environment(tmp_path):
    checkpoint = tmp_path / "policy-latest.zip"
    checkpoint.write_bytes(b"test checkpoint")
    metadata = write_policy_metadata(
        tmp_path / "policy-metadata.json", scenario="hospital",
        phase="static_navigation", seed=5101, training_steps=4096,
        checkpoint=checkpoint, parent_checkpoint=None,
    )
    assert metadata["policy_scope"] == "single_environment"
    assert checkpoint_metadata(checkpoint)["scenario"] == "hospital"
    assert validate_checkpoint_scenario(checkpoint, "hospital")["scenario"] == "hospital"
    with pytest.raises(ValueError, match="belongs to hospital, not office"):
        validate_checkpoint_scenario(checkpoint, "office")


def test_legacy_summary_establishes_environment_ownership(tmp_path):
    checkpoint = tmp_path / "policy-latest.zip"
    checkpoint.write_bytes(b"legacy checkpoint")
    (tmp_path / "summary.json").write_text(
        json.dumps({"scenario": "warehouse"}), encoding="utf-8"
    )
    assert validate_checkpoint_scenario(checkpoint, "warehouse")["policy_scope"] == (
        "single_environment"
    )


def test_checkpoint_without_environment_metadata_is_rejected(tmp_path):
    checkpoint = tmp_path / "orphan.zip"
    checkpoint.write_bytes(b"orphan checkpoint")
    with pytest.raises(ValueError, match="cannot establish environment ownership"):
        validate_checkpoint_scenario(checkpoint, "office")


def test_legacy_cross_environment_lineage_is_rejected(tmp_path):
    office_dir = tmp_path / "office"
    hospital_dir = tmp_path / "hospital"
    office_dir.mkdir()
    hospital_dir.mkdir()
    office_checkpoint = office_dir / "policy-latest.zip"
    hospital_checkpoint = hospital_dir / "policy-latest.zip"
    office_checkpoint.write_bytes(b"office")
    hospital_checkpoint.write_bytes(b"hospital fine tune")
    (office_dir / "summary.json").write_text(
        json.dumps({"scenario": "office", "resumed_from": None}), encoding="utf-8"
    )
    (hospital_dir / "summary.json").write_text(json.dumps({
        "scenario": "hospital", "resumed_from": str(office_checkpoint),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage crosses environments"):
        validate_checkpoint_scenario(hospital_checkpoint, "hospital")
