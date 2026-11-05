#!/usr/bin/env python3
"""Launch one isolated Office Stage 6 matched DRL/reference evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vacan-stage6-matplotlib")
sys.path.insert(0, str(ROOT / "src"))

from vacan.reference.config import load_stage6_config
from vacan.rl.policy_registry import validate_checkpoint_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage6/office.yaml")
    parser.add_argument("--seed", type=int, default=15101)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stage6/office")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaacsim-root", type=Path,
                        default=Path("/home/engineer/IsaacLab/_isaac_sim"))
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/engineer/IsaacLab"))
    args = parser.parse_args()
    config = load_stage6_config(args.config.resolve())
    if config["scenario"] != "office":
        raise SystemExit("this integration launcher is intentionally limited to Office")
    checkpoint = Path(config["resolved_policy_checkpoint"])
    validate_checkpoint_scenario(checkpoint, "office")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output.resolve() / f"matched-seed{args.seed}-{stamp}"
    if run_dir.exists() and not args.dry_run:
        raise SystemExit(f"refusing to overwrite existing run: {run_dir}")
    command = [
        str(args.isaacsim_root / "python.sh"),
        str(ROOT / "scripts/isaacsim/run_stage6_office.py"),
        "--stage6-config", str(args.config.resolve()),
        "--config", config["resolved_stage2_config"], "--output", str(run_dir),
        "--seed", str(args.seed), "--gpu", str(args.gpu),
    ]
    if args.headless:
        command.append("--headless")
    environment = os.environ.copy()
    environment["PYTHONEXE"] = str(ROOT / ".venv/bin/python")
    environment["VACAN_VENV_SITE_PACKAGES"] = str(
        ROOT / ".venv/lib/python3.11/site-packages"
    )
    environment["MPLCONFIGDIR"] = str(run_dir / ".matplotlib")
    environment["PYTHONUNBUFFERED"] = "1"
    python_paths = [
        ROOT / "src",
        args.isaaclab_root / "external/PegasusSimulator/extensions/pegasus.simulator",
        args.isaaclab_root / "source/isaaclab",
        args.isaaclab_root / "source/isaaclab_assets",
        args.isaaclab_root / "source/isaaclab_tasks",
        args.isaaclab_root / "source/isaaclab_rl",
    ]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in python_paths if path.exists()] + [environment.get("PYTHONPATH", "")]
    )
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "scenario": "office",
                          "policy_checkpoint": str(checkpoint), "run_dir": str(run_dir),
                          "command": command}, indent=2, sort_keys=True))
        return
    run_dir.mkdir(parents=True)
    log_path = run_dir / "isaac-sim-stage6.log"
    print(f"[Stage 6] Office matched seed={args.seed}; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
        process.stdout.close(); process.wait()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Stage 6 produced no summary; see {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["command"] = command
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if process.returncode != 0 or summary.get("status") != "pass":
        raise RuntimeError(f"Stage 6 Office run failed: {summary}; see {log_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
