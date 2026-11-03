#!/usr/bin/env python3
"""Generate Stage 1 Isaac Sim datasets using one clean process per episode."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacan.simulation.stage1 import episode_seed, load_stage1_config

SCENARIOS = {
    "office": ROOT / "configs/scenarios/office_lobby.yaml",
    "hospital": ROOT / "configs/scenarios/hospital_corridor.yaml",
    "warehouse": ROOT / "configs/scenarios/warehouse.yaml",
}


def compare_recordings(first: Path, second: Path) -> dict[str, float | int]:
    first_lines = first.read_text(encoding="utf-8").splitlines()
    second_lines = second.read_text(encoding="utf-8").splitlines()
    if len(first_lines) != len(second_lines):
        raise AssertionError(f"Recording length mismatch: {len(first_lines)} != {len(second_lines)}")
    squared_error = 0.0
    sample_count = 0
    max_position_error = 0.0
    exact_density_frames = 0
    matching_density_assignments = 0
    for first_line, second_line in zip(first_lines, second_lines, strict=True):
        left, right = json.loads(first_line), json.loads(second_line)
        if left["simulation_step"] != right["simulation_step"]:
            raise AssertionError("Repeated recording step mismatch")
        left_people = {item["id"]: item for item in left["pedestrians"]}
        right_people = {item["id"]: item for item in right["pedestrians"]}
        if left_people.keys() != right_people.keys():
            raise AssertionError("Repeated recording pedestrian ID mismatch")
        for pedestrian_id in left_people:
            a = left_people[pedestrian_id]["position_xyz_m"]
            b = right_people[pedestrian_id]["position_xyz_m"]
            error = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))
            squared_error += error * error
            sample_count += 1
            max_position_error = max(max_position_error, error)
            grid = left["ground_truth_density_grid"]
            origin_x, origin_y = (float(value) for value in grid["origin_xy_m"])
            resolution = float(grid["resolution_m"])
            left_cell = (
                math.floor((float(a[0]) - origin_x) / resolution),
                math.floor((float(a[1]) - origin_y) / resolution),
            )
            right_cell = (
                math.floor((float(b[0]) - origin_x) / resolution),
                math.floor((float(b[1]) - origin_y) / resolution),
            )
            matching_density_assignments += left_cell == right_cell
        if (left["ground_truth_density_grid"]["counts"] ==
                right["ground_truth_density_grid"]["counts"]):
            exact_density_frames += 1
    return {
        "position_samples": sample_count,
        "max_position_error_m": max_position_error,
        "rms_position_error_m": math.sqrt(squared_error / sample_count),
        "exact_density_frames": exact_density_frames,
        "total_frames": len(first_lines),
        "exact_density_frame_fraction": exact_density_frames / len(first_lines),
        "matching_density_assignments": matching_density_assignments,
        "density_cell_assignment_fraction": matching_density_assignments / sample_count,
    }


def run_process(config_path, output, split, index, seed, args, route_cache_input=None) -> dict:
    isaac_python = args.isaacsim_root / "python.sh"
    command = [
        str(isaac_python), str(ROOT / "scripts/isaacsim/run_pegasus_environment.py"),
        "--config", str(config_path), "--output", str(output), "--seed", str(seed),
        "--split", split, "--episode-index", str(index), "--gpu", str(args.gpu),
    ]
    if args.headless:
        command.append("--headless")
    if args.steps is not None:
        command.extend(("--steps", str(args.steps)))
    if route_cache_input is not None:
        command.extend(("--route-cache-input", str(Path(route_cache_input).resolve())))
    env = os.environ.copy()
    extension = args.pegasus_root / "extensions/pegasus.simulator"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(extension), env.get("PYTHONPATH", ""))
    )
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "isaac-sim.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Isaac process produced no summary; see {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if process.returncode != 0 or summary.get("status") != "pass":
        raise RuntimeError(f"Isaac episode failed: {summary}; see {log_path}")
    shutil.copy2(config_path, output / "scenario.yaml")
    visual_command = [
        sys.executable, str(ROOT / "scripts/visualize_stage1.py"),
        "--recording", str(output / "ground_truth.jsonl"),
        "--config", str(config_path), "--output", str(output / "top_down.png"),
    ]
    subprocess.run(visual_command, cwd=ROOT, check=True)
    summary["command"] = command
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="train")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stage1-isaac")
    parser.add_argument("--isaacsim-root", type=Path, default=Path("/home/engineer/IsaacLab/_isaac_sim"))
    parser.add_argument("--pegasus-root", type=Path, default=Path("/home/engineer/IsaacLab/external/PegasusSimulator"))
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, help="override episode length (smoke tests only)")
    parser.add_argument("--verify-reproducibility", action="store_true")
    parser.add_argument(
        "--reproducibility-tolerance-m", type=float, default=0.25,
        help="maximum independent-run position deviation (default: half of the 0.5 m GT cell)",
    )
    parser.add_argument("--minimum-density-cell-agreement", type=float, default=0.90)
    args = parser.parse_args()
    if not (args.isaacsim_root / "python.sh").is_file():
        raise SystemExit(f"Isaac Sim is unavailable at {args.isaacsim_root}")
    scenarios = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    splits = ("train", "validation", "test") if args.split == "all" else (args.split,)
    manifest: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": "isaac-sim-pegasus", "gpu_index": args.gpu,
        "headless": args.headless, "episodes": [], "reproducibility": [],
    }
    for scenario_name, config_path in scenarios.items():
        config = load_stage1_config(config_path)
        for split in splits:
            count = min(args.episodes, len(config["seeds"][split]))
            for index in range(count):
                seed = episode_seed(config, split, index)
                output = args.output / split / scenario_name / f"episode-{index:03d}"
                summary = run_process(config_path, output, split, index, seed, args)
                manifest["episodes"].append(str(output.relative_to(args.output)))
                if args.verify_reproducibility:
                    repeat_output = (
                        args.output / "reproducibility" / split / scenario_name
                        / f"episode-{index:03d}"
                    )
                    repeated = run_process(
                        config_path, repeat_output, split, index, seed, args,
                        route_cache_input=output / "resolved_routes.json",
                    )
                    comparison = compare_recordings(
                        output / "ground_truth.jsonl", repeat_output / "ground_truth.jsonl"
                    )
                    match = repeated["trajectory_digest_4dp"] == summary["trajectory_digest_4dp"]
                    within_tolerance = (
                        comparison["max_position_error_m"] <= args.reproducibility_tolerance_m
                    )
                    density_reproducible = (
                        comparison["density_cell_assignment_fraction"]
                        >= args.minimum_density_cell_agreement
                    )
                    manifest["reproducibility"].append({
                        "episode": str(output.relative_to(args.output)), "seed": seed,
                        "digest": summary["trajectory_digest_4dp"],
                        "repeat_digest": repeated["trajectory_digest_4dp"],
                        "digest_match": match,
                        "tolerance_m": args.reproducibility_tolerance_m,
                        "within_tolerance": within_tolerance,
                        "minimum_density_cell_agreement": args.minimum_density_cell_agreement,
                        "density_reproducible": density_reproducible,
                        **comparison,
                    })
                    if not within_tolerance or not density_reproducible:
                        raise RuntimeError(
                            f"Repeat-seed reproducibility failed for {output}: "
                            f"max deviation {comparison['max_position_error_m']:.6f} m "
                            f"(limit {args.reproducibility_tolerance_m:.6f} m), density-cell "
                            f"agreement {comparison['density_cell_assignment_fraction']:.6f} "
                            f"(minimum {args.minimum_density_cell_agreement:.6f})"
                        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
