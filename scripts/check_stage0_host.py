#!/usr/bin/env python3
"""Collect non-destructive VACaN Stage 0 host and installation evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def command(*args: str) -> dict[str, object]:
    executable = shutil.which(args[0])
    if executable is None:
        return {"available": False, "command": list(args)}
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=20)
    return {
        "available": True,
        "command": list(args),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def path_evidence(path: Path) -> dict[str, object]:
    return {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    isaaclab = Path(os.environ.get("ISAACLAB_ROOT", "/home/engineer/IsaacLab"))
    isaacsim = Path(os.environ.get("ISAACSIM_ROOT", str(isaaclab / "_isaac_sim")))
    pegasus = Path(
        os.environ.get("PEGASUS_ROOT", str(isaaclab / "external/PegasusSimulator"))
    )
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "memory": command("free", "-h"),
        "storage": command("df", "-h", "."),
        "gpu": command(
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ),
        "cuda_compiler": command("nvcc", "--version"),
        "installations": {
            "isaac_lab": path_evidence(isaaclab),
            "isaac_sim": path_evidence(isaacsim),
            "isaac_sim_python": path_evidence(isaacsim / "python.sh"),
            "pegasus": path_evidence(pegasus),
            "pegasus_person": path_evidence(
                pegasus
                / "extensions/pegasus.simulator/pegasus/simulator/logic/people/person.py"
            ),
            "pegasus_person_controller": path_evidence(
                pegasus
                / "extensions/pegasus.simulator/pegasus/simulator/logic/people/person_controller.py"
            ),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    ready = bool(
        report["installations"]["isaac_sim_python"]["exists"]
        and report["installations"]["pegasus_person"]["exists"]
        and report["gpu"].get("returncode") == 0
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
