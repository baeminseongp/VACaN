#!/usr/bin/env python3
"""Create a deterministic top-down trajectory/density view from Stage 1 GT."""

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
    records = [json.loads(line) for line in args.recording.read_text().splitlines()]
    config = yaml.safe_load(args.config.read_text())
    figure, (axis, density_axis) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ids = [p["id"] for p in records[0]["pedestrians"]]
    for pedestrian_id in ids:
        points = np.asarray([
            next(p for p in record["pedestrians"] if p["id"] == pedestrian_id)["position_xyz_m"]
            for record in records
        ])
        axis.plot(points[:, 0], points[:, 1], linewidth=1.4, label=pedestrian_id)
        axis.scatter(points[0, 0], points[0, 1], s=18, marker="o")
        axis.scatter(points[-1, 0], points[-1, 1], s=22, marker="x")
    for name, bounds in config["ground_truth"]["congestion_zones"].items():
        x0, y0, x1, y1 = bounds
        axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linestyle="--"))
        axis.text(x0, y1, name, fontsize=8)
    xmin, xmax, ymin, ymax = config["visualization"]["bounds_xy"]
    axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", title=f"{config['scenario']['name']} trajectories", xlabel="world x (m)", ylabel="world y (m)")
    axis.legend(fontsize=6, ncol=2)
    peak = max(records, key=lambda record: max(max(row) for row in record["ground_truth_density_grid"]["counts"]))
    grid = peak["ground_truth_density_grid"]
    image = density_axis.imshow(grid["people_per_m2"], origin="lower", cmap="magma", extent=(xmin, xmax, ymin, ymax), interpolation="nearest")
    density_axis.set(title=f"GT density at t={peak['simulation_time_s']:.2f}s", xlabel="world x (m)", ylabel="world y (m)", aspect="equal")
    figure.colorbar(image, ax=density_axis, label="people / m²")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
