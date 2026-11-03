#!/usr/bin/env python3
"""Report whether the optional full ROS/Isaac iCrowdNav runtime is available."""

from __future__ import annotations

import json
import os
import shutil
import subprocess


def command_output(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    result = subprocess.run([executable, *command[1:]], text=True, capture_output=True, check=False)
    return (result.stdout or result.stderr).strip()


def main() -> None:
    usd = os.environ.get("ICROWDNAV_USD")
    report = {
        "roscore": shutil.which("roscore"),
        "catkin_make": shutil.which("catkin_make"),
        "nvidia_smi": command_output(["nvidia-smi", "-L"]),
        "icrowdnav_usd": usd,
        "usd_exists": bool(usd and os.path.isfile(usd)),
        "ready": bool(shutil.which("roscore") and shutil.which("catkin_make") and usd and os.path.isfile(usd)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
