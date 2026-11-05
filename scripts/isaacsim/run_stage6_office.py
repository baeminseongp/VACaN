#!/usr/bin/env python3
"""Run matched frozen-DRL and full-information reference Office episodes."""

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
parser.add_argument("--stage6-config", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True, help="Stage 2 config for reused runtime")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--headless", action="store_true")
ARGS, _ = parser.parse_known_args()


def _load_vacan_gymnasium() -> None:
    site_packages = Path(os.environ.get("VACAN_VENV_SITE_PACKAGES", ""))
    package = site_packages / "gymnasium/__init__.py"
    if not package.is_file():
        raise RuntimeError("VACAN Gymnasium is unavailable; use scripts/run_stage6_office.py")
    sys.path.insert(0, str(site_packages))
    try:
        import gymnasium
    finally:
        sys.path.remove(str(site_packages))
    if gymnasium.__version__ != "0.29.1":
        raise RuntimeError(f"Stage 6 requires Gymnasium 0.29.1, got {gymnasium.__version__}")


def _load_stage2_runtime():
    path = Path(__file__).with_name("run_stage2_episode.py")
    spec = importlib.util.spec_from_file_location("vacan_stage2_runtime_stage6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Stage 2 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhysicalAdapter:
    def __init__(self, runtime_env) -> None:
        self.env = runtime_env

    def reset(self, seed: int, scenario: str):
        return self.env.reset(seed, scenario)

    def step(self, action):
        return self.env.step(action)

    def pedestrian_states(self):
        return [{
            "id": cfg["id"],
            "position_xyz_m": np.asarray(person.state.position, dtype=float).tolist(),
            "velocity_xyz_m_s": ((controller.position - controller.previous_position)
                                  / self.env.dt).tolist(),
            "current_destination_xyz_m": controller.current_destination.tolist(),
        } for cfg, person, controller in self.env.people]


def _resolve_phase(stage2: dict, stage2_path: Path, stage3: dict, training: dict,
                   output: Path, target_count: int) -> tuple[dict, int]:
    from vacan.rl.curriculum import resolve_pedestrian_population

    phase = next(p for p in training["curriculum"]["phases"]
                 if p["name"] == "partial_coverage_noise")
    source_stage1 = (stage2_path.parent / stage2["stage1_config"]).resolve()
    source_routes = (stage2_path.parent / stage2["pedestrian_routes"]).resolve()
    stage1 = yaml.safe_load(source_stage1.read_text(encoding="utf-8"))
    routes = json.loads(source_routes.read_text(encoding="utf-8"))
    count = int(target_count)
    stage1, routes = resolve_pedestrian_population(stage1, routes, count)
    stage1_path = (output / "resolved-stage1.yaml").resolve()
    routes_path = (output / "resolved-routes.json").resolve()
    stage1_path.write_text(yaml.safe_dump(stage1, sort_keys=False), encoding="utf-8")
    routes_path.write_text(json.dumps(routes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage2["stage1_config"], stage2["pedestrian_routes"] = str(stage1_path), str(routes_path)
    resolved_stage3 = copy.deepcopy(stage3)
    for section, values in phase.get("stage3_overrides", {}).items():
        resolved_stage3[section].update(values)
    (output / "resolved-stage3.yaml").write_text(
        yaml.safe_dump(resolved_stage3, sort_keys=False), encoding="utf-8")
    return resolved_stage3, count


def _ground_truth_record(time_s: float, adapter: PhysicalAdapter,
                         density: list[list[int]] | None = None) -> dict:
    return {"simulation_time_s": time_s, "pedestrians": adapter.pedestrian_states(),
            "ground_truth_density": density}


def _sample(time_s: float, robot: dict, action, info: dict, nearest: float | None,
            **extra) -> dict:
    record = {
        "simulation_time_s": float(time_s),
        "position_xy_m": list(map(float, robot["position_xyz_m"][:2])),
        "yaw_rad": float(robot["yaw_rad"]),
        "linear_velocity_m_s": float(robot["linear_velocity_xyz_m_s"][0]),
        "angular_velocity_rad_s": float(robot["angular_velocity_xyz_rad_s"][2]),
        "action": list(map(float, action)), "collision": bool(info.get("collision", False)),
        "goal_reached": bool(info.get("goal_reached", False)),
        "timeout": bool(info.get("timeout", False)),
        "nearest_person_distance_m": nearest,
    }
    record.update(extra)
    return record


def _nearest(robot: dict, pedestrians: list[dict]) -> float | None:
    position = np.asarray(robot["position_xyz_m"][:2], dtype=float)
    return min((float(np.linalg.norm(position - np.asarray(p["position_xyz_m"][:2])))
                for p in pedestrians), default=None)


def run() -> dict:
    _load_vacan_gymnasium()
    from sb3_contrib import RecurrentPPO

    from vacan.mapping.dynamic_costmap import load_stage3_config
    from vacan.mapping.static_map import OccupancyMap2D
    from vacan.reference.comparison import (
        compare_costs, information_error_candidates, trajectory_divergence,
    )
    from vacan.reference.config import load_stage6_config
    from vacan.reference.cost import SharedNavigationCost
    from vacan.reference.dataset import render_debug_figures, write_matched_dataset
    from vacan.reference.planner import (
        PedestrianState, ReferenceWaypointController, TimeExpandedAStar,
    )
    from vacan.rl.config import load_stage4_config
    from vacan.rl.stage5 import file_sha256, predict_deterministic_with_diagnostics
    from vacan.rl.stage5_config import load_stage5_config
    from vacan.rl.stage5_env import VacanNavigationEnv
    from vacan.rl.policy_registry import validate_checkpoint_scenario
    from vacan.robot.stage2 import load_stage2_config, velocity_limits

    started = time.perf_counter()
    config = load_stage6_config(ARGS.stage6_config.resolve())
    if config["scenario"] != "office":
        raise ValueError("Office-only Stage 6 integration received another scenario")
    ARGS.output = ARGS.output.resolve()
    ARGS.output.mkdir(parents=True, exist_ok=True)
    stage2_path = Path(config["resolved_stage2_config"])
    stage2 = load_stage2_config(stage2_path)
    stage4 = load_stage4_config(Path(config["resolved_stage4_config"]))
    training = load_stage5_config(Path(config["resolved_stage5_config"]))
    stage3 = load_stage3_config(Path(stage4["resolved_stage3_config"]))
    stage3, pedestrian_count = _resolve_phase(
        stage2, stage2_path, stage3, training, ARGS.output,
        int(config["evaluation"]["pedestrian_count"]),
    )
    checkpoint = Path(config["resolved_policy_checkpoint"])
    checkpoint_metadata = validate_checkpoint_scenario(checkpoint, "office")
    stage2["collision_probe"] = {"enabled": False}
    stage2["episode"]["max_steps"] = round(
        float(stage4["episode"]["max_duration_s"]) / float(stage2["episode"]["physics_dt_s"]))
    stage2["episode"]["goal_tolerance_m"] = float(stage4["episode"]["goal_tolerance_m"])
    stage4["runtime_goal_xyz"] = list(stage2["robot"]["goal_xyz"])
    runtime = _load_stage2_runtime()
    physical_runtime = runtime.NovaCarterNavigationEnv(stage2, ARGS.output, ARGS.seed)
    adapter = PhysicalAdapter(physical_runtime)
    limits = velocity_limits(stage2)
    env = VacanNavigationEnv(adapter, "office", ARGS.seed,
                             Path(config["resolved_static_map"]), stage3, stage4, limits)
    model = RecurrentPPO.load(checkpoint, env=env, device=f"cuda:{ARGS.gpu}")

    observation, _ = env.reset(seed=ARGS.seed)
    initial_drl_robot = physical_runtime.observe()["robot"]
    initial_drl_pedestrians = adapter.pedestrian_states()
    drl, drl_gt = [], []
    drl.append(_sample(0.0, initial_drl_robot, (0.0, 0.0), {},
                       _nearest(initial_drl_robot, initial_drl_pedestrians),
                       policy_observation={key: value.tolist() for key, value in observation.items()},
                       costmap_diagnostics=(None if env.latest_costmap_record is None else
                                            env.latest_costmap_record["diagnostics"])))
    drl_gt.append(_ground_truth_record(
        0.0, adapter, None if env.latest_costmap_record is None else
        env.latest_costmap_record["privileged"]["ground_truth_count"]))
    recurrent_state = None
    episode_start = np.ones((1,), dtype=bool)
    for _ in range(round(float(stage4["episode"]["max_duration_s"]) * 5.0)):
        decision_observation = {
            key: np.asarray(value, dtype=np.float32).tolist()
            for key, value in observation.items()
        }
        input_episode_start = bool(episode_start[0])
        input_state = None if recurrent_state is None else {
            "actor": {
                "hidden": np.asarray(recurrent_state.actor[0], dtype=np.float32).tolist(),
                "cell": np.asarray(recurrent_state.actor[1], dtype=np.float32).tolist(),
            },
            "critic": {
                "hidden": np.asarray(recurrent_state.critic[0], dtype=np.float32).tolist(),
                "cell": np.asarray(recurrent_state.critic[1], dtype=np.float32).tolist(),
            },
        }
        prediction = predict_deterministic_with_diagnostics(
            model, observation, recurrent_state, episode_start
        )
        observation, _, terminated, truncated, info = env.step(prediction.action)
        recurrent_state = prediction.recurrent_state
        episode_start = np.asarray([terminated or truncated], dtype=bool)
        pedestrians = adapter.pedestrian_states()
        drl.append(_sample(info["simulation_time_s"], info["robot_state"], info["applied_action"],
                           info, info["nearest_person_distance_m"],
                           policy_observation=decision_observation,
                           policy_requested_action=np.asarray(
                               prediction.action, dtype=np.float32
                           ).tolist(),
                           policy_episode_start=input_episode_start,
                           input_hidden_state=input_state,
                           next_hidden_state={
                               "actor": {
                                   "hidden": np.asarray(
                                       recurrent_state.actor[0], dtype=np.float32
                                   ).tolist(),
                                   "cell": np.asarray(
                                       recurrent_state.actor[1], dtype=np.float32
                                   ).tolist(),
                               },
                               "critic": {
                                   "hidden": np.asarray(
                                       recurrent_state.critic[0], dtype=np.float32
                                   ).tolist(),
                                   "cell": np.asarray(
                                       recurrent_state.critic[1], dtype=np.float32
                                   ).tolist(),
                               },
                           },
                           policy_diagnostics={
                               "value_estimate": prediction.value_estimate,
                               "action_log_probability": prediction.action_log_probability,
                               "action_mean": prediction.action_mean.tolist(),
                               "action_std": prediction.action_std.tolist(),
                           },
                           costmap_diagnostics=info["costmap_diagnostics"],
                           reward_terms=info["reward_terms"]))
        drl_gt.append(_ground_truth_record(
            info["simulation_time_s"], adapter,
            None if env.latest_costmap_record is None else
            env.latest_costmap_record["privileged"]["ground_truth_count"]))
        if terminated or truncated:
            break

    reference_initial = adapter.reset(ARGS.seed, "office")
    initial_reference_robot = reference_initial["robot"]
    initial_reference_pedestrians = adapter.pedestrian_states()
    robot_error = float(np.linalg.norm(np.asarray(initial_drl_robot["position_xyz_m"])
                                       - np.asarray(initial_reference_robot["position_xyz_m"])))
    robot_velocity_error = float(np.linalg.norm(
        np.asarray(initial_drl_robot["linear_velocity_xyz_m_s"])
        - np.asarray(initial_reference_robot["linear_velocity_xyz_m_s"])))
    pedestrian_error = max((float(np.linalg.norm(
        np.asarray(a["position_xyz_m"]) - np.asarray(b["position_xyz_m"])))
        for a, b in zip(initial_drl_pedestrians, initial_reference_pedestrians)), default=0.0)
    pedestrian_velocity_error = max((float(np.linalg.norm(
        np.asarray(a["velocity_xyz_m_s"]) - np.asarray(b["velocity_xyz_m_s"])))
        for a, b in zip(initial_drl_pedestrians, initial_reference_pedestrians)), default=0.0)
    if (robot_error > 0.02 or robot_velocity_error > 0.02 or pedestrian_error > 0.02
            or pedestrian_velocity_error > 0.02):
        raise AssertionError(
            f"matched reset mismatch: robot={robot_error}, robot_velocity={robot_velocity_error}, "
            f"people={pedestrian_error}, people_velocity={pedestrian_velocity_error}"
        )

    static_map = OccupancyMap2D.load(Path(config["resolved_static_map"]))
    reference_config = config["reference"]
    planner = TimeExpandedAStar(
        static_map, limits, time_step_s=float(reference_config["time_step_s"]),
        horizon_s=float(reference_config["horizon_s"]),
        pedestrian_radius_m=float(reference_config["pedestrian_radius_m"]),
        goal_tolerance_m=float(stage4["episode"]["goal_tolerance_m"]),
    )
    controller = ReferenceWaypointController(
        limits, float(reference_config["cruise_speed_m_s"]),
        float(reference_config["angular_gain"]),
        float(reference_config["waypoint_tolerance_m"]),
    )
    reference, reference_gt, planning_history = [], [], []
    initial_nearest = _nearest(initial_reference_robot, initial_reference_pedestrians)
    reference.append(_sample(0.0, initial_reference_robot, (0.0, 0.0), {}, initial_nearest))
    reference_gt.append(_ground_truth_record(0.0, adapter))
    control_repeat = round((1.0 / float(config["evaluation"]["control_hz"]))
                           / float(stage2["episode"]["physics_dt_s"]))
    replan_ticks = max(1, round(float(reference_config["replan_interval_s"])
                                * float(config["evaluation"]["control_hz"])))
    last_observation, last_info = reference_initial, {}
    for control_step in range(round(float(stage4["episode"]["max_duration_s"])
                                    * float(config["evaluation"]["control_hz"]))):
        if control_step % replan_ticks == 0:
            robot = last_observation["robot"]
            people = [PedestrianState.from_record(record) for record in adapter.pedestrian_states()]
            plan = planner.plan(tuple(robot["position_xyz_m"][:2]), float(robot["yaw_rad"]),
                                tuple(stage2["robot"]["goal_xyz"][:2]), people,
                                start_time_s=control_step / float(config["evaluation"]["control_hz"]))
            planning_history.append({
                "simulation_time_s": control_step / float(config["evaluation"]["control_hz"]),
                "feasible": plan.feasible, "reason": plan.reason,
                "expanded_states": plan.expanded_states,
                "planned_world_xy_m": plan.world_xy_m,
                "planned_timestamps_s": plan.timestamps_s,
            })
            controller.set_path([] if not plan.feasible else [
                point for index, point in enumerate(plan.world_xy_m)
                if index == 0 or point != plan.world_xy_m[index - 1]
            ])
        robot = last_observation["robot"]
        action = controller.action(tuple(robot["position_xyz_m"][:2]), float(robot["yaw_rad"]))
        terminated = truncated = False
        for _ in range(control_repeat):
            last_observation, _, terminated, truncated, last_info = adapter.step(action)
            if terminated or truncated:
                break
        time_s = float(last_observation["simulation_step"]) * float(stage2["episode"]["physics_dt_s"])
        people = adapter.pedestrian_states()
        reference.append(_sample(time_s, last_observation["robot"],
                                 last_info.get("applied_action", action), last_info,
                                 _nearest(last_observation["robot"], people)))
        reference_gt.append(_ground_truth_record(time_s, adapter))
        if terminated or truncated:
            break

    drl_success = bool(drl[-1]["goal_reached"])
    reference_success = bool(reference[-1]["goal_reached"])
    evaluator = SharedNavigationCost(config["shared_cost"])
    drl_cost = evaluator.evaluate(drl, goal_reached=drl_success)
    reference_cost = evaluator.evaluate(reference, goal_reached=reference_success)
    comparison = compare_costs(
        drl_cost, reference_cost, drl_success=drl_success,
        reference_success=reference_success,
        tolerance=float(config["comparison"]["inefficiency_tolerance"]),
    )
    candidates = trajectory_divergence(
        drl, reference,
        distance_threshold_m=float(config["comparison"]["divergence_distance_m"]),
        heading_change_threshold_rad_s=float(
            config["comparison"]["angular_action_change_rad_s"]),
    )
    information_candidates = information_error_candidates(
        drl, drl_gt, crowd_error_threshold=1.0, intention_error_threshold=1.0,
        speed_change_threshold_m_s=0.2, angular_change_threshold_rad_s=0.5,
        local_penalty_threshold=0.5,
    )
    comparison_payload = comparison.to_dict()
    comparison_payload.update({"drl_cost": drl_cost.to_dict(),
                               "reference_cost": reference_cost.to_dict(),
                               "candidate_divergence_points": candidates,
                               "candidate_information_points": information_candidates})
    metadata = {
        "schema_version": 1, "scenario": "office", "episode_id": f"office-{ARGS.seed}",
        "seed": ARGS.seed, "world_frame": "isaac_world_enu",
        "policy_checkpoint": str(checkpoint), "policy_checkpoint_sha256": file_sha256(checkpoint),
        "policy_metadata": checkpoint_metadata, "policy_frozen": True,
        "policy_inference": "deterministic", "reference_algorithm": reference_config["algorithm"],
        "reference_uses_actual_future_trajectory": False,
        "static_map": config["resolved_static_map"],
        "static_map_sha256": file_sha256(Path(config["resolved_static_map"])),
        "matched_initial_robot_error_m": robot_error,
        "matched_initial_robot_velocity_error_m_s": robot_velocity_error,
        "matched_initial_pedestrian_max_error_m": pedestrian_error,
        "matched_initial_pedestrian_velocity_max_error_m_s": pedestrian_velocity_error,
        "pedestrian_count": pedestrian_count,
    }
    write_matched_dataset(ARGS.output, metadata, drl, reference, comparison_payload,
                          drl_ground_truth=drl_gt, reference_ground_truth=reference_gt,
                          planning_history=planning_history)
    render_debug_figures(ARGS.output, static_map, drl, reference, comparison_payload,
                         tuple(stage2["robot"]["start_xyz"][:2]),
                         tuple(stage2["robot"]["goal_xyz"][:2]),
                         pedestrian_xy=[tuple(record["position_xyz_m"][:2])
                                        for record in initial_reference_pedestrians])
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True,
                              capture_output=True, check=False).stdout.strip() or "unknown"
    result = {
        "status": "pass", "scope": "office_only", "seed": ARGS.seed,
        "drl_success": drl_success, "reference_success": reference_success,
        "comparison": comparison_payload, "planning_calls": len(planning_history),
        "infeasible_planning_calls": sum(not record["feasible"] for record in planning_history),
        "drl_steps": len(drl) - 1, "reference_steps": len(reference) - 1,
        "policy_checkpoint": str(checkpoint), "source_revision": revision,
        "elapsed_wall_s": time.perf_counter() - started,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (ARGS.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env.close()
    runtime.simulation_app.close()
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2, sort_keys=True))
    except Exception as error:
        failure = {"status": "fail", "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc(),
                   "created_utc": datetime.now(UTC).isoformat()}
        ARGS.output.mkdir(parents=True, exist_ok=True)
        (ARGS.output / "summary.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise
