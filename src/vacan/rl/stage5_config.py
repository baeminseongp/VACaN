"""Strict Stage 5 training configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_stage5_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: expected Stage 5 schema_version 1")
    if data.get("algorithm") != "sb3-contrib/RecurrentPPO":
        raise ValueError(f"{path}: unsupported algorithm")
    if data.get("policy_scope") != "single_environment":
        raise ValueError(f"{path}: Stage 5 requires one independent policy per environment")
    timing = data.get("timing", {})
    if float(timing.get("costmap_hz", 0)) != 5.0 or float(timing.get("control_hz", 0)) != 5.0:
        raise ValueError(f"{path}: Stage 5 requires validated 5 Hz costmap/control timing")
    if float(timing.get("physics_hz", 0)) != 60.0:
        raise ValueError(f"{path}: Stage 5 requires validated 60 Hz physics")
    if float(data.get("costmap_decay_s", 0)) != 300.0:
        raise ValueError(f"{path}: costmap decay must remain 300 seconds")
    ppo = data.get("ppo", {})
    rollout, batch, sequence = (int(ppo.get(key, 0)) for key in
                                ("rollout_steps", "batch_size", "sequence_length"))
    if min(rollout, batch, sequence) <= 0:
        raise ValueError(f"{path}: invalid rollout, batch, or sequence length")
    if rollout % batch != 0 or batch % sequence != 0:
        raise ValueError(f"{path}: rollout must divide into batches and recurrent sequences")
    phases = data.get("curriculum", {}).get("phases", [])
    if [phase.get("name") for phase in phases] != [
        "static_navigation", "low_density", "dynamic_crowd", "partial_coverage_noise"
    ]:
        raise ValueError(f"{path}: four ordered curriculum phases are required")
    phase4 = phases[-1]
    robot_override = phase4.get("stage3_overrides", {}).get("robot_observation", {})
    required_phase4 = {
        "position_noise_std_m", "velocity_noise_std_m_s", "minimum_cell_overlap_ratio",
    }
    if set(robot_override) != required_phase4:
        raise ValueError(f"{path}: Phase 4 must define the complete robot-observation override")
    if int(phase4.get("pedestrian_count", 0)) <= 6:
        raise ValueError(f"{path}: Phase 4 must increase pedestrian count above the base six")
    reward_overrides = phase4.get("reward_overrides", {})
    if not set(reward_overrides).issubset({"social"}):
        raise ValueError(f"{path}: unsupported Phase 4 reward override")
    if "social" in reward_overrides and float(reward_overrides["social"]) <= 0:
        raise ValueError(f"{path}: social reward override must be positive")
    if int(data.get("num_envs", 0)) != 1:
        raise ValueError(f"{path}: initial validated implementation requires num_envs=1")
    return data
