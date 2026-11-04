#!/usr/bin/env python3
"""Launch Office-only Stage 7 decision and closed-loop analyses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vacan-stage7-matplotlib")
sys.path.insert(0, str(ROOT / "src"))

from vacan.counterfactual.config import load_stage7_config
from vacan.reference.config import load_stage6_config


def _stream(command: list[str], environment: dict[str, str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
        process.stdout.close(); process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"command failed with {process.returncode}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage7/office.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stage7/office")
    parser.add_argument("--seed", type=int, default=15101)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--decision-only", action="store_true")
    parser.add_argument("--episode-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isaacsim-root", type=Path, default=Path("/home/engineer/IsaacLab/_isaac_sim"))
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/engineer/IsaacLab"))
    args = parser.parse_args()
    if args.decision_only and args.episode_only:
        raise SystemExit("choose at most one of --decision-only and --episode-only")
    config = load_stage7_config(args.config.resolve())
    stage6 = load_stage6_config(Path(config["resolved_stage6_config"]))
    reference_metadata_path = Path(config["resolved_stage6_artifact"]) / "metadata.json"
    reference_metadata = json.loads(reference_metadata_path.read_text(encoding="utf-8"))
    if int(reference_metadata["seed"]) != args.seed:
        raise SystemExit(
            f"Stage 7 seed {args.seed} does not match the configured Stage 6 reference "
            f"seed {reference_metadata['seed']}; generate/select a matched Stage 6 artifact"
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output / f"run-seed{args.seed}-{stamp}"
    environment = os.environ.copy()
    environment.update({
        "PYTHONEXE": str(ROOT / ".venv/bin/python"),
        "VACAN_VENV_SITE_PACKAGES": str(ROOT / ".venv/lib/python3.11/site-packages"),
        "MPLCONFIGDIR": str(run_dir / ".matplotlib"), "PYTHONUNBUFFERED": "1",
    })
    python_paths = [ROOT / "src", args.isaaclab_root / "external/PegasusSimulator/extensions/pegasus.simulator",
                    args.isaaclab_root / "source/isaaclab", args.isaaclab_root / "source/isaaclab_assets",
                    args.isaaclab_root / "source/isaaclab_tasks", args.isaaclab_root / "source/isaaclab_rl"]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(item) for item in python_paths if item.exists()] + [environment.get("PYTHONPATH", "")])
    decision_command = [str(ROOT / ".venv/bin/python"),
                        str(ROOT / "scripts/analyze_stage7_office_decisions.py"),
                        "--config", str(args.config.resolve()), "--output", str(run_dir / "decision")]
    episode_command = [str(args.isaacsim_root / "python.sh"),
                       str(ROOT / "scripts/isaacsim/run_stage7_office.py"),
                       "--stage7-config", str(args.config.resolve()),
                       "--config", stage6["resolved_stage2_config"],
                       "--output", str(run_dir / "episode"), "--seed", str(args.seed),
                       "--gpu", str(args.gpu)]
    if args.headless:
        episode_command.append("--headless")
    commands = []
    if not args.episode_only:
        commands.append(decision_command)
    if not args.decision_only:
        commands.append(episode_command)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "scope": "office_only",
                          "run_dir": str(run_dir), "commands": commands}, indent=2)); return
    run_dir.mkdir(parents=True)
    if not args.episode_only:
        _stream(decision_command, environment, run_dir / "decision.log")
    if not args.decision_only:
        _stream(episode_command, environment, run_dir / "isaac-sim-stage7.log")
    summary = {"status": "pass", "scope": "office_only", "seed": args.seed,
               "run_dir": str(run_dir), "commands": commands}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
