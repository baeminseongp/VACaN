"""Frozen recurrent-policy snapshots and decision-level counterfactual diagnostics."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vacan.rl.stage5 import (
    RecurrentDiagnosticState,
    predict_deterministic_with_diagnostics,
)


@dataclass(frozen=True)
class DecisionSnapshot:
    simulation_time_s: float
    observation: dict[str, np.ndarray]
    recurrent_state: RecurrentDiagnosticState | None
    episode_start: bool


def recurrent_state_from_record(record: dict[str, Any]) -> RecurrentDiagnosticState | None:
    state = record.get("input_hidden_state")
    if state is None:
        return None
    return RecurrentDiagnosticState(
        actor=(np.asarray(state["actor"]["hidden"], dtype=np.float32),
               np.asarray(state["actor"]["cell"], dtype=np.float32)),
        critic=(np.asarray(state["critic"]["hidden"], dtype=np.float32),
                np.asarray(state["critic"]["cell"], dtype=np.float32)),
    )


def snapshot_from_record(record: dict[str, Any]) -> DecisionSnapshot:
    return DecisionSnapshot(
        float(record["simulation_time_s"]),
        {key: np.asarray(value, dtype=np.float32)
         for key, value in record["policy_observation"].items()},
        recurrent_state_from_record(record), bool(record.get("policy_episode_start", False)),
    )


def parameter_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.policy.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _evaluate(model: Any, snapshot: DecisionSnapshot) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    captures: dict[str, np.ndarray] = {}
    hooks = []
    seen: set[int] = set()
    for root_name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        root = getattr(model.policy, root_name, None)
        if root is None or id(root) in seen:
            continue
        seen.add(id(root))
        for name, module in root.named_modules():
            if name == "" or isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                key = f"{root_name}.{name or 'output'}"
                hooks.append(module.register_forward_hook(
                    lambda _module, _inputs, output, key=key:
                    captures.__setitem__(key, output.detach().cpu().numpy().copy())
                ))
    try:
        prediction = predict_deterministic_with_diagnostics(
            model, copy.deepcopy(snapshot.observation), snapshot.recurrent_state,
            snapshot.episode_start,
        )
    finally:
        for hook in hooks:
            hook.remove()
    diagnostics = {
        "action": prediction.action.tolist(),
        "action_mean": prediction.action_mean.tolist(),
        "action_std": prediction.action_std.tolist(),
        "action_log_probability": prediction.action_log_probability,
        "critic_value": prediction.value_estimate,
        "next_actor_hidden": prediction.recurrent_state.actor[0].tolist(),
        "next_actor_cell": prediction.recurrent_state.actor[1].tolist(),
        "next_critic_hidden": prediction.recurrent_state.critic[0].tolist(),
        "next_critic_cell": prediction.recurrent_state.critic[1].tolist(),
    }
    return diagnostics, captures


def evaluate_decision_pair(model: Any, original: DecisionSnapshot,
                           counterfactual_observation: dict[str, np.ndarray]) \
        -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate two inputs from the exact same pre-decision recurrent state."""
    before = parameter_fingerprint(model)
    counterfactual = DecisionSnapshot(
        original.simulation_time_s,
        {key: np.asarray(value, dtype=np.float32).copy()
         for key, value in counterfactual_observation.items()},
        original.recurrent_state, original.episode_start,
    )
    original_result, original_features = _evaluate(model, original)
    counterfactual_result, counterfactual_features = _evaluate(model, counterfactual)
    after = parameter_fingerprint(model)
    if before != after:
        raise AssertionError("frozen policy parameters changed during counterfactual evaluation")
    action_original = np.asarray(original_result["action"], dtype=float)
    action_counterfactual = np.asarray(counterfactual_result["action"], dtype=float)
    actor_original = np.asarray(original_result["next_actor_hidden"], dtype=float)
    actor_counterfactual = np.asarray(counterfactual_result["next_actor_hidden"], dtype=float)
    feature_metrics = {}
    arrays: dict[str, np.ndarray] = {}
    for key in sorted(set(original_features) & set(counterfactual_features)):
        first, second = original_features[key], counterfactual_features[key]
        arrays[f"original::{key}"] = first
        arrays[f"counterfactual::{key}"] = second
        difference = second.astype(np.float64) - first.astype(np.float64)
        denominator = max(float(np.linalg.norm(first)), 1e-12)
        feature_metrics[key] = {
            "shape": list(first.shape), "l2": float(np.linalg.norm(difference)),
            "normalized_l2": float(np.linalg.norm(difference) / denominator),
        }
    result = {
        "simulation_time_s": original.simulation_time_s,
        "same_input_recurrent_state": True,
        "policy_parameter_sha256_before": before,
        "policy_parameter_sha256_after": after,
        "original": original_result, "counterfactual": counterfactual_result,
        "differences": {
            "action_l2": float(np.linalg.norm(action_counterfactual - action_original)),
            "linear_action": float(action_counterfactual[0] - action_original[0]),
            "angular_action": float(action_counterfactual[1] - action_original[1]),
            "critic_value": float(counterfactual_result["critic_value"]
                                  - original_result["critic_value"]),
            "actor_hidden_l2": float(np.linalg.norm(actor_counterfactual - actor_original)),
            "input_costmap_l2": float(np.linalg.norm(
                counterfactual.observation["map"] - original.observation["map"])),
            "features": feature_metrics,
        },
    }
    return result, arrays


def replace_dynamic_policy_channel(original: dict[str, np.ndarray], global_costmap: np.ndarray,
                                   builder: Any, robot_xy: tuple[float, float], yaw: float) \
        -> dict[str, np.ndarray]:
    """Re-sample a modified Stage 3 world grid while holding all other policy input fixed."""
    result = {key: np.asarray(value, dtype=np.float32).copy() for key, value in original.items()}
    crowd = builder._sample(global_costmap, builder.dynamic_geometry, robot_xy, yaw, 0.0)
    result["map"][1] = np.clip(crowd / builder.spec.crowd_clip, 0.0, 1.0)
    return result
