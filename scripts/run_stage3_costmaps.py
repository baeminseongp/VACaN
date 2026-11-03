#!/usr/bin/env python3
"""Generate deterministic Stage 3 costmaps from synchronized Stage 2 state logs."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacan.mapping.dynamic_costmap import DynamicCostmap, load_stage3_config

CONFIGS = {name: ROOT / "configs/stage3" / f"{name}.yaml" for name in
           ("office", "hospital", "warehouse")}


def run_scenario(config_path: Path, output: Path) -> dict:
    config = load_stage3_config(config_path)
    recording = (config_path.parent / config["stage2_recording"]).resolve()
    if not recording.is_file():
        raise FileNotFoundError(f"Stage 2 recording is missing: {recording}")
    output.mkdir(parents=True, exist_ok=True)
    manager = DynamicCostmap(config, int(config["seed"]))
    tick_count = observation_count = event_count = 0
    processing_started = time.perf_counter()
    policy_path = output / "policy_costmap.jsonl.gz"
    diagnostics_path = output / "diagnostics.jsonl.gz"
    privileged_path = output / "privileged_ground_truth.jsonl.gz"
    events_path = output / "intention_events.jsonl"
    with (
        policy_path.open("wb") as policy_raw,
        diagnostics_path.open("wb") as diagnostics_raw,
        privileged_path.open("wb") as privileged_raw,
        events_path.open("w", encoding="utf-8") as event_stream,
        gzip.open(policy_raw, "wt", encoding="utf-8") as policy_stream,
        gzip.open(diagnostics_raw, "wt", encoding="utf-8") as diagnostics_stream,
        gzip.open(privileged_raw, "wt", encoding="utf-8") as privileged_stream,
    ):
        for line in recording.read_text(encoding="utf-8").splitlines():
            source = json.loads(line)
            simulation_time = float(source["simulation_time_s"])
            robot = source["observation"]["robot"]
            pedestrians = source["privileged_log"]["pedestrians"]
            result = manager.tick(simulation_time, robot, pedestrians)
            if result is None:
                continue
            common = {
                "timestamp": source["timestamp"], "simulation_time_s": simulation_time,
                "scenario": config["scenario"], "world_frame": config["world_frame"],
                "seed": config["seed"],
            }
            policy_stream.write(json.dumps({**common, **result["policy"]}, separators=(",", ":")) + "\n")
            diagnostic = {
                **common,
                "robot": {"position_xyz_m": robot["position_xyz_m"], "yaw_rad": robot["yaw_rad"]},
                **result["diagnostics"],
            }
            diagnostics_stream.write(json.dumps(diagnostic, separators=(",", ":")) + "\n")
            privileged_stream.write(json.dumps({**common, **result["privileged"]},
                                               separators=(",", ":")) + "\n")
            for event in result["diagnostics"]["intention_events"]:
                event_stream.write(json.dumps({**common, **event}, separators=(",", ":")) + "\n")
            tick_count += 1
            observation_count += len(result["diagnostics"]["observations"])
            event_count += len(result["diagnostics"]["intention_events"])
    elapsed = time.perf_counter() - processing_started
    simulated_duration = manager.next_update_time - manager.update_interval if manager.next_update_time else 0.0
    summary = {
        "status": "pass", "created_utc": datetime.now(UTC).isoformat(),
        "scenario": config["scenario"], "seed": config["seed"],
        "source_recording": str(recording), "costmap_shape": list(manager.geometry.shape),
        "update_frequency_hz": config["update_frequency_hz"], "ticks": tick_count,
        "observations": observation_count, "intention_events": event_count,
        "processing_elapsed_s": elapsed,
        "processing_ms_per_tick": 1000.0 * elapsed / tick_count,
        "throughput_ticks_per_s": tick_count / elapsed,
        "last_simulation_time_s": simulated_duration,
        "outputs": [policy_path.name, diagnostics_path.name, privileged_path.name, events_path.name],
    }
    (output / "scenario.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    print(f"[Stage 3] {config['scenario']}: rendering {tick_count} timestep images", flush=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/visualize_stage3.py"),
        "--directory", str(output), "--output", str(output / "costmap.png"),
        "--frames-dir", str(output / "frames"),
    ], cwd=ROOT, check=True)
    print(f"[Stage 3] {config['scenario']}: visualization complete", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("all", *CONFIGS), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stage3")
    args = parser.parse_args()
    selected = CONFIGS if args.scenario == "all" else {args.scenario: CONFIGS[args.scenario]}
    manifest = {"backend": "stage2-state-replay", "created_utc": datetime.now(UTC).isoformat(),
                "episodes": []}
    for scenario, path in selected.items():
        directory = args.output / scenario
        summary = run_scenario(path, directory)
        manifest["episodes"].append({"scenario": scenario, "path": scenario, "summary": summary})
        print(f"[Stage 3] {scenario}: {summary['ticks']} ticks, "
              f"{summary['processing_ms_per_tick']:.3f} ms/tick", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
