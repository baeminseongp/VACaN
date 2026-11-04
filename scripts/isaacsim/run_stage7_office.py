#!/usr/bin/env python3
"""Run matched closed-loop Office counterfactual branches in Isaac Sim."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--stage7-config", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--headless", action="store_true")
ARGS, _ = parser.parse_known_args()


def _load_vacan_gymnasium() -> None:
    site_packages = Path(os.environ.get("VACAN_VENV_SITE_PACKAGES", ""))
    if not (site_packages / "gymnasium/__init__.py").is_file():
        raise RuntimeError("VACAN Gymnasium is unavailable; use scripts/run_stage7_office.py")
    sys.path.insert(0, str(site_packages))
    try:
        import gymnasium
    finally:
        sys.path.remove(str(site_packages))
    if gymnasium.__version__ != "0.29.1":
        raise RuntimeError(f"Stage 7 requires Gymnasium 0.29.1, got {gymnasium.__version__}")


def _load_stage2_runtime():
    path = Path(__file__).with_name("run_stage2_episode.py")
    spec = importlib.util.spec_from_file_location("vacan_stage2_runtime_stage7", path)
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
                   output: Path) -> tuple[dict, int]:
    from vacan.rl.curriculum import resolve_pedestrian_population
    phase = next(item for item in training["curriculum"]["phases"]
                 if item["name"] == "partial_coverage_noise")
    stage1 = yaml.safe_load((stage2_path.parent / stage2["stage1_config"]).resolve().read_text())
    routes = json.loads((stage2_path.parent / stage2["pedestrian_routes"]).resolve().read_text())
    count = int(phase["pedestrian_count"])
    stage1, routes = resolve_pedestrian_population(stage1, routes, count)
    stage1_path, routes_path = output / "resolved-stage1.yaml", output / "resolved-routes.json"
    stage1_path.write_text(yaml.safe_dump(stage1, sort_keys=False), encoding="utf-8")
    routes_path.write_text(json.dumps(routes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage2["stage1_config"], stage2["pedestrian_routes"] = str(stage1_path), str(routes_path)
    resolved = copy.deepcopy(stage3)
    for section, values in phase.get("stage3_overrides", {}).items():
        resolved[section].update(values)
    (output / "resolved-stage3.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return resolved, count


def _nearest(robot: dict, pedestrians: list[dict]) -> float | None:
    position = np.asarray(robot["position_xyz_m"][:2], dtype=float)
    return min((float(np.linalg.norm(position - np.asarray(item["position_xyz_m"][:2])))
                for item in pedestrians), default=None)


def _sample(time_s: float, robot: dict, action, info: dict, nearest: float | None,
            **extra) -> dict:
    result = {
        "simulation_time_s": float(time_s),
        "position_xy_m": list(map(float, robot["position_xyz_m"][:2])),
        "yaw_rad": float(robot["yaw_rad"]), "action": list(map(float, action)),
        "collision": bool(info.get("collision", False)),
        "goal_reached": bool(info.get("goal_reached", False)),
        "timeout": bool(info.get("timeout", False)),
        "nearest_person_distance_m": nearest,
    }
    result.update(extra)
    return result


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n"
                            for item in records), encoding="utf-8")


def run() -> dict:
    _load_vacan_gymnasium()
    from sb3_contrib import RecurrentPPO
    from vacan.counterfactual.config import load_stage7_config
    from vacan.counterfactual.case_studies import (
        coverage_gap_metrics, freshness_validity_metrics,
        paired_freshness_metrics, paired_trajectory_metrics,
    )
    from vacan.counterfactual.dataset import (
        render_cctv_case_figure, render_coverage_gap_case_figure,
        render_episode_figure, render_freshness_case_figure,
    )
    from vacan.counterfactual.intervention import InterventionManager
    from vacan.mapping.dynamic_costmap import DynamicCostmap, load_stage3_config
    from vacan.reference.config import load_stage6_config
    from vacan.reference.cost import SharedNavigationCost
    from vacan.rl.config import load_stage4_config
    from vacan.rl.stage5 import predict_deterministic_with_diagnostics
    from vacan.rl.stage5_config import load_stage5_config
    from vacan.rl.stage5_env import VacanNavigationEnv
    from vacan.robot.stage2 import load_stage2_config, velocity_limits

    started = time.perf_counter()
    cfg7 = load_stage7_config(ARGS.stage7_config.resolve())
    cfg6 = load_stage6_config(Path(cfg7["resolved_stage6_config"]))
    reference_metadata = json.loads(
        (Path(cfg7["resolved_stage6_artifact"]) / "metadata.json").read_text(encoding="utf-8")
    )
    if int(reference_metadata["seed"]) != ARGS.seed:
        raise ValueError(
            f"Stage 7 seed {ARGS.seed} does not match Stage 6 reference seed "
            f"{reference_metadata['seed']}"
        )
    stage2_path = Path(cfg6["resolved_stage2_config"])
    stage2 = load_stage2_config(stage2_path)
    stage4 = load_stage4_config(Path(cfg6["resolved_stage4_config"]))
    training = load_stage5_config(Path(cfg6["resolved_stage5_config"]))
    stage3 = load_stage3_config(Path(stage4["resolved_stage3_config"]))
    ARGS.output.mkdir(parents=True, exist_ok=True)
    stage3, pedestrian_count = _resolve_phase(stage2, stage2_path, stage3, training, ARGS.output)
    stage2["collision_probe"] = {"enabled": False}
    stage2["episode"]["max_steps"] = round(
        float(stage4["episode"]["max_duration_s"]) / float(stage2["episode"]["physics_dt_s"]))
    stage2["episode"]["goal_tolerance_m"] = float(stage4["episode"]["goal_tolerance_m"])
    stage4["runtime_goal_xyz"] = list(stage2["robot"]["goal_xyz"])
    limits = velocity_limits(stage2)
    runtime = _load_stage2_runtime()
    physical = runtime.NovaCarterNavigationEnv(stage2, ARGS.output, ARGS.seed)
    adapter = PhysicalAdapter(physical)
    checkpoint = Path(cfg7["resolved_policy_checkpoint"])
    model = None
    evaluator = SharedNavigationCost(cfg6["shared_cost"])
    specs_by_id = {item.intervention_id: item for item in cfg7["parsed_interventions"]}
    branches = [{"id": "original", "intervention_ids": []}] + list(cfg7["experiments"])
    baseline_by_id = {"original": None, **{
        item["id"]: item.get("baseline_id", "original") for item in cfg7["experiments"]
    }}
    branch_records: dict[str, list[dict]] = {}
    branch_ground_truth: dict[str, list[dict]] = {}
    branch_results = []
    original_reset = None
    original_trajectory = None
    original_cost = None

    for branch in branches:
        branch_dir = ARGS.output / "branches" / branch["id"]
        branch_dir.mkdir(parents=True, exist_ok=True)
        selected = [specs_by_id[item] for item in branch["intervention_ids"]]
        probe = DynamicCostmap(stage3, ARGS.seed)
        manager = None if not selected else InterventionManager(
            selected, probe.geometry, probe.cctv_cells
        )
        env = VacanNavigationEnv(
            adapter, "office", ARGS.seed, Path(cfg6["resolved_static_map"]), stage3,
            stage4, limits, intervention_manager=manager,
        )
        if model is None:
            model = RecurrentPPO.load(checkpoint, env=env, device=f"cuda:{ARGS.gpu}")
        else:
            model.set_env(env)
        observation, _ = env.reset(seed=ARGS.seed)
        initial_robot = physical.observe()["robot"]
        initial_people = adapter.pedestrian_states()
        reset_state = {
            "robot_position": initial_robot["position_xyz_m"],
            "robot_velocity": initial_robot["linear_velocity_xyz_m_s"],
            "pedestrian_positions": [item["position_xyz_m"] for item in initial_people],
            "pedestrian_velocities": [item["velocity_xyz_m_s"] for item in initial_people],
        }
        if original_reset is None:
            original_reset = reset_state
        reset_error = max(
            float(np.max(np.abs(np.asarray(reset_state[key], dtype=float)
                                - np.asarray(original_reset[key], dtype=float))))
            for key in reset_state
        )
        if reset_error > 1e-9:
            raise AssertionError(f"{branch['id']}: matched reset error {reset_error}")
        common = {"scenario_id": "office", "episode_id": f"office-{ARGS.seed}",
                  "counterfactual_branch": branch["id"],
                  "policy_checkpoint": str(checkpoint)}
        trajectory = [_sample(0.0, initial_robot, (0, 0), {},
                              _nearest(initial_robot, initial_people), **common)]
        ground_truth = [{"simulation_time_s": 0.0, "pedestrians": initial_people,
                         "ground_truth_density": env.latest_costmap_record["privileged"]["ground_truth_count"],
                         **common}]
        state = None
        episode_start = np.ones((1,), dtype=bool)
        for _step in range(round(float(stage4["episode"]["max_duration_s"]) * 5.0)):
            decision_observation = {key: np.asarray(value, dtype=np.float32).tolist()
                                    for key, value in observation.items()}
            input_state = None if state is None else {
                "actor": {"hidden": state.actor[0].tolist(), "cell": state.actor[1].tolist()},
                "critic": {"hidden": state.critic[0].tolist(), "cell": state.critic[1].tolist()},
            }
            prediction = predict_deterministic_with_diagnostics(
                model, observation, state, episode_start
            )
            observation, _, terminated, truncated, info = env.step(prediction.action)
            state = prediction.recurrent_state
            episode_start = np.asarray([terminated or truncated], dtype=bool)
            people = adapter.pedestrian_states()
            trajectory.append(_sample(
                info["simulation_time_s"], info["robot_state"], info["applied_action"], info,
                info["nearest_person_distance_m"], policy_observation=decision_observation,
                policy_requested_action=prediction.action.tolist(), input_hidden_state=input_state,
                policy_diagnostics={"value_estimate": prediction.value_estimate,
                                    "action_mean": prediction.action_mean.tolist(),
                                    "action_std": prediction.action_std.tolist()},
                costmap_diagnostics=info["costmap_diagnostics"],
                **common,
            ))
            ground_truth.append({
                "simulation_time_s": info["simulation_time_s"], "pedestrians": people,
                "ground_truth_density": env.latest_costmap_record["privileged"]["ground_truth_count"],
                **common,
            })
            if terminated or truncated:
                break
        success = bool(trajectory[-1]["goal_reached"])
        cost = evaluator.evaluate(trajectory, goal_reached=success)
        _write_jsonl(branch_dir / "trajectory.jsonl", trajectory)
        _write_jsonl(branch_dir / "ground-truth.jsonl", ground_truth)
        intervention_logs = [] if manager is None else [
            {**item, **common} for item in manager.logs
        ]
        _write_jsonl(branch_dir / "intervention-events.jsonl", intervention_logs)
        if original_trajectory is None:
            original_trajectory, original_cost = trajectory, cost
        delta_vs_actual = 0.0 if original_cost is None else float(cost.total - original_cost.total)
        branch_result = {
            "branch_id": branch["id"], "intervention_ids": branch["intervention_ids"],
            "success": success, "steps": len(trajectory) - 1, "reset_max_error": reset_error,
            "cost": cost.to_dict(), "delta_j_vs_actual": delta_vs_actual,
            "intervention_event_count": 0 if manager is None else len(manager.logs),
        }
        branch_results.append(branch_result)
        branch_records[branch["id"]] = trajectory
        branch_ground_truth[branch["id"]] = ground_truth

    assert original_trajectory is not None and original_cost is not None
    branch_time = 0.0
    for result in branch_results[1:]:
        records = branch_records[result["branch_id"]]
        baseline_id = baseline_by_id[result["branch_id"]]
        assert baseline_id is not None
        baseline_records = branch_records[baseline_id]
        prefix = [(r["position_xy_m"], r["action"]) for r in records
                  if r["simulation_time_s"] <= branch_time + 1e-9]
        baseline_prefix = [(r["position_xy_m"], r["action"]) for r in baseline_records
                           if r["simulation_time_s"] <= branch_time + 1e-9]
        result["pre_branch_baseline"] = baseline_id
        result["pre_branch_trajectory_equal"] = prefix == baseline_prefix
        if not result["pre_branch_trajectory_equal"]:
            raise AssertionError(f"{result['branch_id']}: state diverged before intervention")
    by_branch = {item["branch_id"]: item for item in branch_results}
    full_cost = float(by_branch["full_information"]["cost"]["total"])
    for branch_result in branch_results:
        branch_result["delta_j_vs_full_information"] = float(
            branch_result["cost"]["total"] - full_cost
        )
        branch_result["delta_j_cf"] = branch_result["delta_j_vs_full_information"]
        baseline_id = baseline_by_id[branch_result["branch_id"]]
        if baseline_id is None:
            branch_result["comparison_baseline"] = None
            branch_result["delta_j_vs_baseline"] = 0.0
        else:
            branch_result["comparison_baseline"] = baseline_id
            branch_result["delta_j_vs_baseline"] = float(
                branch_result["cost"]["total"] - by_branch[baseline_id]["cost"]["total"]
            )
    no_op = by_branch["no_op"]
    no_op_core = [(r["position_xy_m"], r["yaw_rad"], r["action"], r["goal_reached"])
                  for r in branch_records["no_op"]]
    original_core = [(r["position_xy_m"], r["yaw_rad"], r["action"], r["goal_reached"])
                     for r in original_trajectory]
    no_op["full_trajectory_equal"] = no_op_core == original_core
    if (not no_op["full_trajectory_equal"]
            or abs(no_op["delta_j_vs_actual"]) > cfg7["comparison_tolerance"]):
        raise AssertionError("no-op replay changed the closed-loop episode")
    factor_attribution = {
        "delta_j_actual_observation": by_branch["original"]["delta_j_cf"],
        "delta_j_congestion_error": by_branch["congestion_error"]["delta_j_cf"],
        "delta_j_intention_error": by_branch["intention_error"]["delta_j_cf"],
        "delta_j_combined_error": by_branch["combined_error"]["delta_j_cf"],
    }
    factor_attribution["non_additive_interaction"] = (
        factor_attribution["delta_j_combined_error"]
        - factor_attribution["delta_j_congestion_error"]
        - factor_attribution["delta_j_intention_error"]
    )
    for branch_result in branch_results[1:]:
        baseline_id = branch_result["pre_branch_baseline"]
        baseline_result = by_branch[baseline_id]
        render_episode_figure(
            ARGS.output / f"episode-{branch_result['branch_id']}.png",
            branch_records[baseline_id], branch_records[branch_result["branch_id"]],
            {"original_cost": baseline_result["cost"],
             "counterfactual_cost": branch_result["cost"],
             "delta_j_cf": branch_result["delta_j_vs_baseline"]},
        )
    np.savez_compressed(
        ARGS.output / "branch-trajectories.npz",
        **{f"{identifier}_time_s": np.asarray(
                [item["simulation_time_s"] for item in records], dtype=np.float64)
           for identifier, records in branch_records.items()},
        **{f"{identifier}_xy_m": np.asarray(
                [item["position_xy_m"] for item in records], dtype=np.float32)
           for identifier, records in branch_records.items()},
    )
    result = {
        "status": "pass", "scope": "office_only", "seed": ARGS.seed,
        "policy_checkpoint": str(checkpoint), "policy_frozen": True,
        "pedestrian_count": pedestrian_count, "branches": branch_results,
        "shared_cost_config": cfg6["shared_cost"],
        "pedestrians_react_to_robot": False,
        "interventions": [item.to_dict() for item in cfg7["parsed_interventions"]],
        "factor_attribution": factor_attribution,
        "replay_semantics": "closed_loop_physical_matched_reset",
        "primary_comparison_baseline": "full_information_drl",
        "stage6_reference_role": "auxiliary_efficiency_comparator",
        "elapsed_wall_s": time.perf_counter() - started,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    stage6_comparison = json.loads(
        (Path(cfg7["resolved_stage6_artifact"]) / "comparison.json").read_text(encoding="utf-8")
    )
    result["stage6_auxiliary_reference"] = {
        "cost": stage6_comparison["reference_cost"],
        "drl_cost": stage6_comparison["drl_cost"],
        "delta_j_drl_minus_reference": stage6_comparison["delta_j"],
        "primary_counterfactual_baseline": False,
    }
    case_cfg = cfg7["case_studies"]
    trajectory_threshold = float(case_cfg["trajectory_divergence_m"])
    action_threshold = float(case_cfg["action_divergence_l2"])
    minimum_cost = float(case_cfg["minimum_cost_increase"])

    def pair(correct_id: str, comparison_id: str) -> dict:
        metrics = paired_trajectory_metrics(
            branch_records[correct_id], branch_records[comparison_id],
            trajectory_threshold_m=trajectory_threshold,
            action_threshold_l2=action_threshold,
        )
        delta = float(by_branch[comparison_id]["cost"]["total"]
                      - by_branch[correct_id]["cost"]["total"])
        metrics["correct_branch"] = correct_id
        metrics["comparison_branch"] = comparison_id
        metrics["delta_j_comparison_minus_correct"] = delta
        metrics["navigation_failure_candidate"] = bool(
            delta > minimum_cost and metrics["path_difference_observed"]
        )
        return metrics

    case1_1_cfg = case_cfg["case_1_1"]
    case1_1 = pair(case1_1_cfg["correct_branch"], case1_1_cfg["error_branch"])
    case1_1.update({
        "case_id": "case_1_1_missing_cctv_coverage",
        "information_error": "unobserved_non_cctv_crowd",
        "base_cctv_assumption": "exact_where_installed",
        "missing_coverage_metrics": coverage_gap_metrics(
            branch_records[case1_1_cfg["error_branch"]],
            branch_ground_truth[case1_1_cfg["error_branch"]],
            cells=case1_1_cfg["affected_cells"], geometry=cfg7["geometry"],
        ),
        "interpretation": (
            "paired comparison against counterfactual exact coverage in an otherwise non-CCTV cell"
        ),
    })
    render_coverage_gap_case_figure(
        ARGS.output / "case-1-1-missing-cctv-coverage.png",
        branch_records[case1_1_cfg["correct_branch"]],
        branch_records[case1_1_cfg["error_branch"]],
        branch_ground_truth[case1_1_cfg["error_branch"]],
        case1_1_cfg["affected_cells"][0],
    )
    case1_2_cfg = case_cfg["case_1_2"]
    case1_2 = pair(case1_2_cfg["correct_branch"], case1_2_cfg["error_branch"])
    case1_2.update({
        "case_id": "case_1_2_cctv_sensing_error",
        "information_error": "counterfactual_cctv_corruption",
        "base_cctv_assumption": "exact_at_every_tick",
        "interpretation": "controlled robustness test; not an Actual-DRL sensor claim",
    })
    render_cctv_case_figure(
        ARGS.output / "case-1-2-cctv-sensing-error.png",
        branch_records[case1_2_cfg["correct_branch"]],
        branch_records[case1_2_cfg["error_branch"]],
        case1_2_cfg["affected_cells"][0],
    )
    case2_cfg = case_cfg["case_2"]
    case2 = pair(case2_cfg["correct_branch"], case2_cfg["error_branch"])
    affected_cells = case2_cfg["affected_cells"]
    case2.update({
        "case_id": "case_2_intention_induced_freshness_error",
        "information_error": "incorrect_controller_destination",
        "freshness_difference": paired_freshness_metrics(
            branch_records[case2_cfg["correct_branch"]],
            branch_records[case2_cfg["error_branch"]], cells=affected_cells,
        ),
        "correct_branch_validity": freshness_validity_metrics(
            branch_records[case2_cfg["correct_branch"]],
            branch_ground_truth[case2_cfg["correct_branch"]],
            crowd_clip=float(case_cfg["crowd_clip"]), cells=affected_cells,
        ),
        "error_branch_validity": freshness_validity_metrics(
            branch_records[case2_cfg["error_branch"]],
            branch_ground_truth[case2_cfg["error_branch"]],
            crowd_clip=float(case_cfg["crowd_clip"]), cells=affected_cells,
        ),
    })
    render_freshness_case_figure(
        ARGS.output / "case-2-intention-freshness-error.png",
        branch_records[case2_cfg["correct_branch"]],
        branch_records[case2_cfg["error_branch"]], affected_cells,
    )
    reference_path = Path(cfg7["resolved_stage6_artifact"]) / "reference-trajectory.jsonl"
    if not reference_path.is_file():
        raise RuntimeError(f"Case 3 requires Stage 6 reference trajectory: {reference_path}")
    reference_records = [json.loads(line) for line in reference_path.read_text(
        encoding="utf-8").splitlines()]
    case3_cfg = case_cfg["case_3"]
    drl_branch = case3_cfg["drl_branch"]
    case3 = paired_trajectory_metrics(
        reference_records, branch_records[drl_branch],
        trajectory_threshold_m=trajectory_threshold, action_threshold_l2=action_threshold,
    )
    reference_cost = stage6_comparison["reference_cost"]
    delta_policy = float(by_branch[drl_branch]["cost"]["total"] - reference_cost["total"])
    reference_success = float(reference_cost["terminal_failure"]) == 0.0
    case3.update({
        "case_id": "case_3_policy_limitation_vs_time_expanded_astar",
        "drl_branch": drl_branch,
        "reference": case3_cfg["reference"],
        "drl_cost": by_branch[drl_branch]["cost"],
        "reference_cost": reference_cost,
        "delta_j_drl_minus_reference": delta_policy,
        "reference_success": reference_success,
        "navigation_failure_candidate": bool(
            reference_success and delta_policy > minimum_cost
            and case3["path_difference_observed"]
        ),
        "global_optimality_claimed": False,
    })
    render_episode_figure(
        ARGS.output / "case-3-full-information-vs-astar.png",
        reference_records, branch_records[drl_branch],
        {"original_cost": reference_cost,
         "counterfactual_cost": by_branch[drl_branch]["cost"],
         "delta_j_cf": delta_policy},
    )
    result["case_studies"] = {
        "case_1_1": case1_1, "case_1_2": case1_2,
        "case_2": case2, "case_3": case3,
        "selection_rule": (
            "candidate only when the comparison cost is higher and the path/outcome diverges"
        ),
        "causal_scope": "matched execution-specific counterfactual association",
    }
    (ARGS.output / "case-study-results.json").write_text(
        json.dumps(result["case_studies"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ARGS.output / "episode-results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env.close(); runtime.simulation_app.close()
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2, sort_keys=True))
    except Exception as error:
        failure = {"status": "fail", "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(), "created_utc": datetime.now(UTC).isoformat()}
        ARGS.output.mkdir(parents=True, exist_ok=True)
        (ARGS.output / "episode-results.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2, sort_keys=True)); raise
