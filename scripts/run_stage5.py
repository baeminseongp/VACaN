#!/usr/bin/env python3
"""Launch isolated Stage 5 Recurrent PPO training/evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("office", "hospital", "warehouse")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vacan-stage5-matplotlib")
sys.path.insert(0, str(ROOT / "src"))

from vacan.rl.policy_registry import validate_checkpoint_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), default="smoke")
    parser.add_argument("--scenario", choices=SCENARIOS, default="office")
    parser.add_argument("--phase", choices=("static_navigation", "low_density", "dynamic_crowd",
                                             "partial_coverage_noise"), default="dynamic_crowd")
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "stage5")
    parser.add_argument("--run-name")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaacsim-root", type=Path,
                        default=Path("/home/engineer/IsaacLab/_isaac_sim"))
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/engineer/IsaacLab"))
    args = parser.parse_args()
    if args.mode == "evaluate" and args.checkpoint is None:
        raise SystemExit("--mode evaluate requires --checkpoint")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.mode}-{args.phase}-seed{args.seed}-{timestamp}"
    run_dir = args.output / "policies" / args.scenario / run_name
    if run_dir.exists() and not args.dry_run:
        raise SystemExit(f"refusing to overwrite existing run: {run_dir}")
    stage2 = ROOT / "configs" / "stage2" / f"{args.scenario}.yaml"
    stage4 = ROOT / "configs" / "stage4" / f"{args.scenario}.yaml"
    static_map = ROOT / "artifacts" / "stage4" / args.scenario / "occupancy_2d.npz"
    training = ROOT / "configs" / "stage5" / "recurrent_ppo.yaml"
    for required in (stage2, stage4, static_map, training, args.isaacsim_root / "python.sh",
                     ROOT / ".venv" / "bin" / "python"):
        if not required.exists():
            raise SystemExit(f"required Stage 5 input is missing: {required}")
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint is missing: {checkpoint}")
        try:
            validate_checkpoint_scenario(checkpoint, args.scenario)
        except (ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"environment-specific checkpoint validation failed: {error}") from error
    dependency_check = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-c",
         "import gymnasium,sb3_contrib,stable_baselines3,tensorboard"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if dependency_check.returncode != 0:
        raise SystemExit(
            "VACaN Stage 5 dependencies are incomplete; install the rl extra with "
            "`.venv/bin/python -m pip install -e '.[rl]'`.\n" + dependency_check.stderr
        )
    command = [
        str(args.isaacsim_root / "python.sh"),
        str(ROOT / "scripts" / "isaacsim" / "run_stage5_training.py"),
        "--config", str(stage2), "--stage4-config", str(stage4),
        "--training-config", str(training), "--static-map", str(static_map),
        "--output", str(run_dir), "--seed", str(args.seed), "--gpu", str(args.gpu),
        "--mode", args.mode, "--phase", args.phase,
    ]
    if args.headless:
        command.append("--headless")
    if args.total_steps is not None:
        command.extend(("--total-steps", str(args.total_steps)))
    if args.checkpoint is not None:
        command.extend(("--checkpoint", str(args.checkpoint.resolve())))
    environment = os.environ.copy()
    environment["PYTHONEXE"] = str(ROOT / ".venv" / "bin" / "python")
    # Isaac Sim 5.1 exposes Gymnasium 1.2.1 before the selected interpreter's
    # site-packages. SB3 2.3.2 requires Gymnasium 0.29.1. The child selectively
    # imports only Gymnasium from this directory; putting the whole directory on
    # PYTHONPATH would incorrectly shadow Isaac's CUDA Torch with VACaN's CPU Torch.
    environment["VACAN_VENV_SITE_PACKAGES"] = str(
        ROOT / ".venv" / "lib" / "python3.11" / "site-packages"
    )
    environment["MPLCONFIGDIR"] = str(run_dir / ".matplotlib")
    environment["PYTHONUNBUFFERED"] = "1"
    python_paths = [
        ROOT / "src",
        args.isaaclab_root / "external" / "PegasusSimulator" / "extensions" / "pegasus.simulator",
        args.isaaclab_root / "source" / "isaaclab",
        args.isaaclab_root / "source" / "isaaclab_assets",
        args.isaaclab_root / "source" / "isaaclab_tasks",
        args.isaaclab_root / "source" / "isaaclab_mimic",
        args.isaaclab_root / "source" / "isaaclab_rl",
    ]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in python_paths if path.exists()] + [environment.get("PYTHONPATH", "")]
    )
    payload = {"status": "dry-run", "run_dir": str(run_dir), "command": command,
               "policy_scope": "single_environment", "policy_scenario": args.scenario,
               "python_executable": environment["PYTHONEXE"]}
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    run_dir.mkdir(parents=True)
    log_path = run_dir / "isaac-sim-training.log"
    print(f"[Stage 5] {args.mode} {args.scenario}/{args.phase}; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            # The terminal also delivers SIGINT to the child process group.
            # Give Isaac Sim time to execute its explicit shutdown path.
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            raise
        finally:
            process.stdout.close()
        process.wait()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Stage 5 process produced no summary; see {log_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["command"] = command
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if process.returncode != 0 or summary.get("status") != "pass":
        raise RuntimeError(f"Stage 5 run failed: {summary}; see {log_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
