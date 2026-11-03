#!/usr/bin/env python3
"""Render robot and privileged pedestrian trajectories in the Isaac world frame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vacan-mpl")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.recording.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise SystemExit("recording is empty")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)

    robot = np.asarray([record["observation"]["robot"]["position_xyz_m"] for record in records])
    axis.plot(robot[:, 0], robot[:, 1], color="black", linewidth=2.5, label="Nova Carter")
    axis.scatter(robot[0, 0], robot[0, 1], marker="o", s=70, color="tab:green", label="start")
    goal = np.asarray(config["robot"]["goal_xyz"])
    axis.scatter(goal[0], goal[1], marker="*", s=170, color="tab:red", label="goal")

    pedestrian_ids = [item["id"] for item in records[0]["privileged_log"]["pedestrians"]]
    for pedestrian_id in pedestrian_ids:
        points = np.asarray([
            next(
                item for item in record["privileged_log"]["pedestrians"]
                if item["id"] == pedestrian_id
            )["position_xyz_m"]
            for record in records
        ])
        axis.plot(points[:, 0], points[:, 1], linewidth=0.9, alpha=0.65, label=pedestrian_id)

    all_x = np.concatenate((robot[:, 0], np.asarray([goal[0]])))
    all_y = np.concatenate((robot[:, 1], np.asarray([goal[1]])))
    margin = 1.5
    axis.set(
        xlim=(all_x.min() - margin, all_x.max() + margin),
        ylim=(all_y.min() - margin, all_y.max() + margin),
        aspect="equal",
        xlabel="world x (m)",
        ylabel="world y (m)",
        title=f"Stage 2 {config['scenario']} trajectory",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
