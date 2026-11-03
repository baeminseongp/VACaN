#!/usr/bin/env python3
"""Launch one visible Stage 4 depth mapping session without Pegasus people."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacan.rl import load_stage4_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("office", "hospital", "warehouse"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stage4-teleop")
    parser.add_argument("--isaacsim-root", type=Path, default=Path("/home/engineer/IsaacLab/_isaac_sim"))
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/stage4" / f"{args.scenario}.yaml"
    config = load_stage4_config(config_path)
    directory = args.output / args.scenario
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(args.isaacsim_root / "python.sh"),
               str(ROOT / "scripts/isaacsim/run_stage4_teleop_mapper.py"),
               "--config", str(config_path), "--output", str(directory), "--gpu", str(args.gpu)]
    if args.dry_run:
        print(" ".join(command))
        return
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), environment.get("PYTHONPATH", "")))
    log_path = directory / "isaac-sim.log"
    print(f"[Stage 4 teleop] Starting visible {args.scenario} session", flush=True)
    print("Controls: W/S drive, A/D turn, Space stop, P save and quit", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"teleop mapper produced no summary; see {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if process.returncode or summary.get("status") != "pass":
        raise RuntimeError(f"teleop mapping failed: {summary}; see {log_path}")
    stage2_path = (config_path.parent / config["stage2_config"]).resolve()
    subprocess.run([sys.executable, str(ROOT / "scripts/visualize_stage4_maps.py"),
                    "--directory", str(directory), "--stage2-config", str(stage2_path)],
                   cwd=ROOT, check=True)
    summary["command"] = command
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[Stage 4 teleop] Saved {args.scenario}: {directory}", flush=True)


if __name__ == "__main__":
    main()
