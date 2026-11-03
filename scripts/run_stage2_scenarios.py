#!/usr/bin/env python3
"""Launch isolated Stage 2 Isaac Sim episodes without modifying Isaac Lab."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacan.robot.stage2 import load_stage2_config

SCENARIOS = {
    name: ROOT / "configs" / "stage2" / f"{name}.yaml"
    for name in ("office", "hospital", "warehouse")
}


def launch(config_path: Path, output: Path, seed: int, args: argparse.Namespace) -> dict:
    command = [
        str(args.isaacsim_root / "python.sh"),
        str(ROOT / "scripts" / "isaacsim" / "run_stage2_episode.py"),
        "--config", str(config_path),
        "--output", str(output),
        "--seed", str(seed),
        "--gpu", str(args.gpu),
    ]
    if args.headless:
        command.append("--headless")
    if args.dry_run:
        return {"status": "dry-run", "command": command}

    output.mkdir(parents=True, exist_ok=True)
    # Do not leave a previous successful visualization/recording beside a new
    # failed summary, which would make acceptance evidence ambiguous.
    for stale_name in (
        "summary.json", "runtime-checks.json", "trajectory.jsonl", "top_down.png", "scenario.yaml"
    ):
        stale_path = output / stale_name
        if stale_path.is_file():
            stale_path.unlink()
    environment = os.environ.copy()
    pegasus_extension = args.pegasus_root / "extensions" / "pegasus.simulator"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(pegasus_extension), environment.get("PYTHONPATH", ""))
    )
    log_path = output / "isaac-sim.log"
    mode = "headless" if args.headless else "visible real-time preview"
    print(
        f"[Stage 2] Starting {config_path.stem} seed={seed} ({mode}). "
        f"Log: {log_path}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Isaac process produced no summary; see {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if process.returncode != 0 or summary.get("status") != "pass":
        raise RuntimeError(f"Stage 2 episode failed: {summary}; see {log_path}")
    summary["command"] = command
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(config_path, output / "scenario.yaml")
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "visualize_stage2.py"),
            "--recording", str(output / "trajectory.jsonl"),
            "--config", str(config_path),
            "--output", str(output / "top_down.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    print(
        f"[Stage 2] Finished {config_path.stem}: {output / 'summary.json'}; "
        f"view: {output / 'top_down.png'}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "stage2-isaac")
    parser.add_argument(
        "--isaacsim-root", type=Path, default=Path("/home/engineer/IsaacLab/_isaac_sim")
    )
    parser.add_argument(
        "--pegasus-root",
        type=Path,
        default=Path("/home/engineer/IsaacLab/external/PegasusSimulator"),
    )
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if not args.dry_run and not (args.isaacsim_root / "python.sh").is_file():
        raise SystemExit(f"Isaac Sim is unavailable at {args.isaacsim_root}")

    selected = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    manifest_path = args.output / "manifest.json"
    if not args.dry_run and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("backend") != "isaac-sim-pegasus-nova-carter":
            raise SystemExit(f"Incompatible existing manifest: {manifest_path}")
        manifest["updated_utc"] = datetime.now(UTC).isoformat()
        manifest["gpu_index"] = args.gpu
        manifest["headless"] = args.headless
    else:
        manifest: dict[str, object] = {
            "created_utc": datetime.now(UTC).isoformat(),
            "backend": "isaac-sim-pegasus-nova-carter",
            "isaacsim_root": str(args.isaacsim_root),
            "isaaclab_tree_modified": False,
            "gpu_index": args.gpu,
            "headless": args.headless,
            "episodes": [],
        }
    for scenario, config_path in selected.items():
        config = load_stage2_config(config_path)
        seeds = config["seeds"][args.split]
        for index, seed in enumerate(seeds[: args.episodes]):
            output = args.output / args.split / scenario / f"episode-{index:03d}"
            summary = launch(config_path, output, int(seed), args)
            relative = output.relative_to(args.output)
            manifest["episodes"] = [
                item for item in manifest["episodes"] if item["path"] != str(relative)
            ]
            manifest["episodes"].append(
                {"path": str(relative), "scenario": scenario, "seed": seed, "summary": summary}
            )

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
