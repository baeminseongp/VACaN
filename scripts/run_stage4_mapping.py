#!/usr/bin/env python3
"""Launch isolated collision-map generation without modifying the Isaac Lab tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacan.rl import load_stage4_config

CONFIGS = {name: ROOT / "configs" / "stage4" / f"{name}.yaml"
           for name in ("office", "hospital", "warehouse")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("all", *CONFIGS), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "stage4")
    parser.add_argument("--isaacsim-root", type=Path, default=Path("/home/engineer/IsaacLab/_isaac_sim"))
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="regenerate even if matching cached maps exist")
    args = parser.parse_args()
    selected = CONFIGS if args.scenario == "all" else {args.scenario: CONFIGS[args.scenario]}
    episodes = []
    for name, config_path in selected.items():
        config = load_stage4_config(config_path)
        directory = args.output / name
        input_hash = hashlib.sha256()
        for path in (
            config_path,
            (config_path.parent / config["stage1_config"]).resolve(),
            (config_path.parent / config["stage2_config"]).resolve(),
            Path(config["resolved_stage3_config"]),
            ROOT / "scripts/isaacsim/generate_stage4_static_map.py",
            ROOT / "src/vacan/mapping/static_map.py",
        ):
            input_hash.update(path.read_bytes())
        command = [str(args.isaacsim_root / "python.sh"),
                   str(ROOT / "scripts/isaacsim/generate_stage4_static_map.py"),
                   "--config", str(config_path), "--output", str(directory),
                   "--gpu", str(args.gpu)]
        if args.headless:
            command.append("--headless")
        if args.dry_run:
            print(" ".join(command))
            continue
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / "summary.json"
        required = ("voxel_map.npz", "occupancy_2d.npz", "official_omap_2d_raw.npz",
                    "voxel_map_3d.png", "occupancy_2d.png", "official_omap_2d_raw.png")
        if not args.force and summary_path.is_file() and all((directory / value).is_file() for value in required):
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("status") == "pass" and cached.get("input_sha256") == input_hash.hexdigest():
                episodes.append({"scenario": name, "path": name, "summary": cached,
                                 "command": command, "cache_reused": True})
                print(f"[Stage 4] Reused cached {name} map", flush=True)
                continue
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), environment.get("PYTHONPATH", "")))
        log_path = directory / "isaac-sim.log"
        print(f"[Stage 4] Mapping {name}; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if process.returncode or summary.get("status") != "pass":
            raise RuntimeError(f"Stage 4 mapping failed: {summary}; see {log_path}")
        stage2_path = (config_path.parent / config["stage2_config"]).resolve()
        subprocess.run([sys.executable, str(ROOT / "scripts/visualize_stage4_maps.py"),
                        "--directory", str(directory), "--stage2-config", str(stage2_path)],
                       cwd=ROOT, check=True)
        episodes.append({"scenario": name, "path": name, "summary": summary, "command": command})
        print(f"[Stage 4] Finished {name}: {summary['occupied_voxels']} occupied voxels", flush=True)
    if not args.dry_run:
        manifest = {"schema_version": 1, "backend": "isaac-sim-physx-collision-voxelization",
                    "created_utc": datetime.now(UTC).isoformat(), "isaaclab_tree_modified": False,
                    "episodes": episodes}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
