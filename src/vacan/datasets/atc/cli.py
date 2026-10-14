"""Command-line interface for ATC normalization and alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .alignment import write_alignment_report
from .map import load_ros_map, write_map_outputs
from .trajectory import normalize_trajectories


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vacan.datasets.atc.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    map_parser = subparsers.add_parser("map")
    map_parser.add_argument("yaml", type=Path)
    map_parser.add_argument("output_directory", type=Path)
    trajectory_parser = subparsers.add_parser("trajectory")
    trajectory_parser.add_argument("source", type=Path)
    trajectory_parser.add_argument("output", type=Path)
    trajectory_parser.add_argument("--max-rows", type=int)
    alignment_parser = subparsers.add_parser("align")
    alignment_parser.add_argument("yaml", type=Path)
    alignment_parser.add_argument("trajectories", type=Path)
    alignment_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "map":
        ros_map = write_map_outputs(args.yaml, args.output_directory)
        print(json.dumps({"width": ros_map.width, "height": ros_map.height, "output": str(args.output_directory)}))
    elif args.command == "trajectory":
        count = normalize_trajectories(args.source, args.output, max_rows=args.max_rows)
        print(json.dumps({"rows": count, "output": str(args.output)}))
    else:
        ros_map = load_ros_map(args.yaml)
        report = write_alignment_report(ros_map, args.trajectories, args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
