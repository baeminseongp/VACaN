#!/usr/bin/env python3
"""Create a seed-deduplicated Stage 6 scenario summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vacan.reference.aggregate import aggregate_runs


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=ROOT / "artifacts/stage6/office")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate_runs(args.root.resolve())
    output = args.output or args.root / "aggregate-summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
