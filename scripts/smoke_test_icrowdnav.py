#!/usr/bin/env python3
"""Smoke-test the local iCrowdNav-compatible Stage 0/1 boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vacan.simulation import DynamicPedestrianSimulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="warehouse")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render-output", type=Path)
    args = parser.parse_args()
    config = ROOT / "configs" / "scenarios" / f"{args.scenario}.yaml"
    simulation = DynamicPedestrianSimulation(config)
    try:
        observation_a, info = simulation.reset(seed=args.seed)
        observation_b, _ = simulation.reset(seed=args.seed)
        assert observation_a == observation_b
        transition = simulation.step(np.array([0.25, 0.0], dtype=np.float32))
        assert set(transition[0]) == {"timestamp", "simulation_step", "robot_observation", "sources"}
        assert "grid" not in transition[0]
        frame = simulation.render_frame()
        assert frame.ndim == 3 and frame.shape[2] == 3
        if args.render_output:
            import cv2

            args.render_output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.render_output), frame)
        print(json.dumps({
            "status": "PASS", "backend": info["backend"], "scenario": args.scenario,
            "action_space": simulation.action_space, "observation_keys": sorted(observation_a),
            "frame_shape": list(frame.shape), "ground_truth_exposed_to_policy": False,
        }, indent=2, sort_keys=True))
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
