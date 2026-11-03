#!/usr/bin/env python3
"""Render a saved Stage 1 Ground Truth episode as a top-down PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacan.simulation.visualization import visualize_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(visualize_episode(args.ground_truth, args.output))


if __name__ == "__main__":
    main()
