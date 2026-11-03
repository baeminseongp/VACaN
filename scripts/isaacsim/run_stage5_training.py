#!/usr/bin/env python3
"""Train/evaluate Recurrent PPO inside the isolated Isaac Sim process."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True, help="Stage 2 scenario config")
parser.add_argument("--stage4-config", type=Path, required=True)
parser.add_argument("--training-config", type=Path, required=True)
parser.add_argument("--static-map", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), required=True)
parser.add_argument("--phase", default="dynamic_crowd")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--total-steps", type=int)
ARGS, _ = parser.parse_known_args()


def _load_vacan_gymnasium() -> None:
    """Pin SB3's Gymnasium without shadowing Isaac's CUDA Torch/Numpy stack."""
    site_packages = Path(os.environ.get("VACAN_VENV_SITE_PACKAGES", ""))
    package = site_packages / "gymnasium" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(
            "VACAN Gymnasium package is unavailable; run Stage 5 through scripts/run_stage5.py"
        )
    existing = sys.modules.get("gymnasium")
    if existing is not None:
        loaded_path = Path(getattr(existing, "__file__", "")).resolve()
        if site_packages.resolve() not in loaded_path.parents:
            raise RuntimeError(f"incompatible Gymnasium was imported before Stage 5: {loaded_path}")
        return
    sys.path.insert(0, str(site_packages))
    try:
        import gymnasium
    finally:
        sys.path.remove(str(site_packages))
    if gymnasium.__version__ != "0.29.1":
        raise RuntimeError(f"Stage 5 requires Gymnasium 0.29.1, got {gymnasium.__version__}")


def _load_stage2_runtime():
    path = Path(__file__).with_name("run_stage2_episode.py")
    spec = importlib.util.spec_from_file_location("vacan_stage2_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Stage 2 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase_population(stage2: dict, config_path: Path, output: Path,
                      phase: dict) -> int:
    from vacan.rl.curriculum import resolve_pedestrian_population

    original = (config_path.parent / stage2["stage1_config"]).resolve()
    data = yaml.safe_load(original.read_text(encoding="utf-8"))
    route_path = (config_path.parent / stage2["pedestrian_routes"]).resolve()
    routes = json.loads(route_path.read_text(encoding="utf-8"))
    base_count = len(data.get("pedestrians", []))
    count = int(phase.get(
        "pedestrian_count",
        round(base_count * float(phase.get("pedestrian_density_scale", 1.0))),
    ))
    data, routes = resolve_pedestrian_population(data, routes, count)
    phase_path = output / "resolved-stage1.yaml"
    route_output = output / "resolved-routes.json"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    phase_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    route_output.write_text(json.dumps(routes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage2["stage1_config"] = str(phase_path)
    stage2["pedestrian_routes"] = str(route_output)
    return count


def _resolved_stage3(stage3: dict, phase: dict, output: Path) -> dict:
    """Apply explicit curriculum overrides without mutating Stage 3 source YAML."""
    resolved = copy.deepcopy(stage3)
    for section, values in phase.get("stage3_overrides", {}).items():
        if section not in resolved or not isinstance(resolved[section], dict):
            raise ValueError(f"invalid Stage 3 override section: {section}")
        for key, value in values.items():
            if key not in resolved[section]:
                raise ValueError(f"invalid Stage 3 override key: {section}.{key}")
            resolved[section][key] = value
    path = output / "resolved-stage3.yaml"
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return resolved


class PhysicalAdapter:
    def __init__(self, runtime_env) -> None:
        self.env = runtime_env

    def reset(self, seed: int, scenario: str):
        return self.env.reset(seed, scenario)

    def step(self, action):
        return self.env.step(action)

    def pedestrian_states(self):
        result = []
        for cfg, person, controller in self.env.people:
            velocity = (controller.position - controller.previous_position) / self.env.dt
            result.append({
                "id": cfg["id"],
                "position_xyz_m": np.asarray(person.state.position, dtype=float).tolist(),
                "velocity_xyz_m_s": np.asarray(velocity, dtype=float).tolist(),
                "current_destination_xyz_m": controller.current_destination.tolist(),
            })
        return result


def run() -> dict:
    _load_vacan_gymnasium()
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from vacan.mapping.dynamic_costmap import load_stage3_config
    from vacan.rl.config import load_stage4_config
    from vacan.rl.stage5 import file_sha256, predict_deterministic, recurrent_policy_kwargs
    from vacan.rl.stage5_config import load_stage5_config
    from vacan.rl.stage5_env import VacanNavigationEnv
    from vacan.rl.policy_registry import (
        POLICY_SCOPE, validate_checkpoint_scenario, write_policy_metadata,
    )
    from vacan.robot.stage2 import load_stage2_config, velocity_limits

    runtime = _load_stage2_runtime()
    stage2 = load_stage2_config(ARGS.config.resolve())
    stage4 = load_stage4_config(ARGS.stage4_config.resolve())
    training = load_stage5_config(ARGS.training_config.resolve())
    checkpoint_metadata = None
    if ARGS.checkpoint:
        checkpoint_metadata = validate_checkpoint_scenario(
            ARGS.checkpoint.resolve(), stage2["scenario"]
        )
    phase = next((p for p in training["curriculum"]["phases"] if p["name"] == ARGS.phase), None)
    if phase is None:
        raise ValueError(f"unknown curriculum phase: {ARGS.phase}")
    ARGS.output.mkdir(parents=True, exist_ok=True)
    phase_pedestrian_count = _phase_population(
        stage2, ARGS.config.resolve(), ARGS.output, phase
    )
    stage2["collision_probe"] = {"enabled": False}
    stage2["episode"]["max_steps"] = round(
        float(stage4["episode"]["max_duration_s"]) / float(stage2["episode"]["physics_dt_s"])
    )
    stage2["episode"]["goal_tolerance_m"] = float(stage4["episode"]["goal_tolerance_m"])
    stage4["runtime_goal_xyz"] = list(stage2["robot"]["goal_xyz"])
    reward_overrides = phase.get("reward_overrides", {})
    for key, value in reward_overrides.items():
        if key not in stage4["reward"]:
            raise ValueError(f"invalid Stage 4 reward override: {key}")
        stage4["reward"][key] = float(value)
    (ARGS.output / "resolved-reward.json").write_text(
        json.dumps(stage4["reward"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stage3 = _resolved_stage3(
        load_stage3_config(Path(stage4["resolved_stage3_config"])), phase, ARGS.output
    )
    physical_runtime = runtime.NovaCarterNavigationEnv(stage2, ARGS.output, ARGS.seed)
    env = Monitor(
        VacanNavigationEnv(
            PhysicalAdapter(physical_runtime), stage2["scenario"], ARGS.seed,
            ARGS.static_map.resolve(), stage3, stage4, velocity_limits(stage2),
        ),
        filename=str(ARGS.output / "monitor.csv"),
        info_keywords=("goal_reached", "collision", "timeout", "termination_reason",
                       "goal_distance_m", "path_length_m",
                       "episode_min_person_distance_m", "episode_social_violation_steps"),
    )

    ppo = training["ppo"]
    device = f"cuda:{ARGS.gpu}"
    if ARGS.checkpoint:
        model = RecurrentPPO.load(ARGS.checkpoint, env=env, device=device)
        # SB3 serializes tensorboard_log in the checkpoint. A resumed run must
        # never append metrics to the source run's artifact directory.
        model.tensorboard_log = str(ARGS.output / "tensorboard")
    else:
        model = RecurrentPPO(
            "MultiInputLstmPolicy", env,
            learning_rate=float(ppo["learning_rate"]), n_steps=int(ppo["rollout_steps"]),
            batch_size=int(ppo["batch_size"]), n_epochs=int(ppo["optimization_epochs"]),
            gamma=float(ppo["gamma"]), gae_lambda=float(ppo["gae_lambda"]),
            clip_range=float(ppo["clip_range"]), ent_coef=float(ppo["entropy_coefficient"]),
            vf_coef=float(ppo["value_coefficient"]), max_grad_norm=float(ppo["max_grad_norm"]),
            policy_kwargs=recurrent_policy_kwargs(training), tensorboard_log=str(ARGS.output / "tensorboard"),
            seed=ARGS.seed, device=device, verbose=1,
        )
    started = time.perf_counter()
    evaluation = None
    if ARGS.mode in {"smoke", "train"}:
        steps = (int(ppo["smoke_steps"]) if ARGS.mode == "smoke" else
                 int(ARGS.total_steps or phase["steps"]))
        callback = CheckpointCallback(
            save_freq=max(1, int(training["checkpoints"]["save_frequency_steps"])),
            save_path=str(ARGS.output / "checkpoints"), name_prefix=f"{ARGS.phase}-seed-{ARGS.seed}",
            save_replay_buffer=False, save_vecnormalize=False,
        )
        model.learn(total_timesteps=steps, callback=callback,
                    reset_num_timesteps=ARGS.checkpoint is None, progress_bar=False)
        model.save(ARGS.output / "policy-latest")
        policy_metadata = write_policy_metadata(
            ARGS.output / "policy-metadata.json", scenario=stage2["scenario"],
            phase=ARGS.phase, seed=ARGS.seed, training_steps=int(model.num_timesteps),
            checkpoint=ARGS.output / "policy-latest.zip",
            parent_checkpoint=ARGS.checkpoint,
        )
    else:
        if ARGS.checkpoint is None:
            raise ValueError("evaluate mode requires --checkpoint")
        observation, reset_info = env.reset(seed=ARGS.seed)
        state = None
        episode_start = np.ones((1,), dtype=bool)
        total_reward = 0.0
        records = []
        for step in range(round(float(stage4["episode"]["max_duration_s"]) * 5.0)):
            input_episode_start = bool(episode_start[0])
            policy_observation = {
                key: np.asarray(value, dtype=np.float32).tolist()
                for key, value in observation.items()
            }
            input_hidden_state = None if state is None else {
                "hidden": np.asarray(state[0], dtype=np.float32).tolist(),
                "cell": np.asarray(state[1], dtype=np.float32).tolist(),
            }
            inference_started = time.perf_counter()
            prediction = predict_deterministic(model, observation, state, episode_start)
            inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
            observation, reward, terminated, truncated, info = env.step(prediction.action)
            state = prediction.hidden_state
            episode_start = np.asarray([terminated or truncated], dtype=bool)
            total_reward += float(reward)
            records.append({"step": step, "action": prediction.action.tolist(),
                            "reward": float(reward), "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "episode_start": input_episode_start,
                            "policy_observation": policy_observation,
                            "input_hidden_state": input_hidden_state,
                            "next_hidden_state": {
                                "hidden": np.asarray(state[0], dtype=np.float32).tolist(),
                                "cell": np.asarray(state[1], dtype=np.float32).tolist(),
                            },
                            "inference_latency_ms": inference_latency_ms,
                            "info": info})
            if terminated or truncated:
                break
        (ARGS.output / "evaluation.jsonl").write_text(
            "".join(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                    for record in records), encoding="utf-8"
        )
        latencies = [record["inference_latency_ms"] for record in records]
        steady_latencies = latencies[1:] if len(latencies) > 1 else latencies
        evaluation = {
            "return": total_reward, "steps": len(records),
            "mean_inference_latency_ms": float(np.mean(latencies)) if latencies else None,
            "p95_inference_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "cold_start_inference_latency_ms": latencies[0] if latencies else None,
            "steady_state_mean_inference_latency_ms": (
                float(np.mean(steady_latencies)) if steady_latencies else None
            ),
            "steady_state_p95_inference_latency_ms": (
                float(np.percentile(steady_latencies, 95)) if steady_latencies else None
            ),
            "termination": records[-1]["info"] if records else reset_info,
        }
        policy_metadata = checkpoint_metadata
    elapsed = time.perf_counter() - started
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True,
                              capture_output=True, check=False).stdout.strip() or "unknown"
    result = {
        "status": "pass", "mode": ARGS.mode, "scenario": stage2["scenario"],
        "policy_scope": POLICY_SCOPE, "policy_metadata": policy_metadata,
        "phase": ARGS.phase, "seed": ARGS.seed, "elapsed_s": elapsed,
        "pedestrian_count": phase_pedestrian_count,
        "phase_stage3_overrides": phase.get("stage3_overrides", {}),
        "phase_reward_overrides": reward_overrides,
        "training_steps": int(model.num_timesteps),
        "resumed_from": str(ARGS.checkpoint) if ARGS.checkpoint else None,
        "checkpoint": str(ARGS.output / "policy-latest.zip") if ARGS.mode != "evaluate"
        else str(ARGS.checkpoint), "evaluation": evaluation,
        "versions": {"isaac_sim": "5.1.0", "isaac_lab": "2.3.2", "pegasus": "5.1.0",
                     "stable_baselines3": "2.3.2", "sb3_contrib": "2.3.0",
                     "gymnasium": "0.29.1"},
        "source_revision": revision,
        "training_config_sha256": file_sha256(ARGS.training_config),
        "static_map_sha256": file_sha256(ARGS.static_map),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (ARGS.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    env.close()
    runtime.simulation_app.close()
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2, sort_keys=True))
    except Exception as error:
        failure = {"status": "fail", "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(), "created_utc": datetime.now(UTC).isoformat()}
        ARGS.output.mkdir(parents=True, exist_ok=True)
        (ARGS.output / "summary.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise
