"""Top-down Stage 1 trajectory and congestion visualization."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from .ground_truth import read_ground_truth


def visualize_episode(ground_truth_path: str | Path, output_path: str | Path) -> Path:
    records = list(read_ground_truth(ground_truth_path))
    if not records:
        raise ValueError("Ground Truth file is empty")
    first = records[0]
    figure, (map_axis, density_axis) = plt.subplots(1, 2, figsize=(13, 5.5))

    trajectories: dict[int, tuple[list[float], list[float]]] = {}
    robot_x, robot_y = [], []
    for record in records:
        robot_x.append(record["robot"]["x"])
        robot_y.append(record["robot"]["y"])
        for pedestrian in record["pedestrians"]:
            xs, ys = trajectories.setdefault(int(pedestrian["id"]), ([], []))
            xs.append(pedestrian["x"])
            ys.append(pedestrian["y"])

    for obstacle in first["static_obstacles"]:
        if obstacle["shape"] == "circle":
            map_axis.add_patch(
                Circle((obstacle["x"], obstacle["y"]), obstacle["radius"], color="0.35")
            )
        else:
            map_axis.add_patch(
                Rectangle(
                    (obstacle["x"] - obstacle["length"] / 2, obstacle["y"] - obstacle["width"] / 2),
                    obstacle["length"],
                    obstacle["width"],
                    angle=math.degrees(obstacle.get("orientation", 0.0)),
                    rotation_point="center",
                    color="0.35",
                )
            )
    for name, zone in first["congestion_zones"].items():
        # Bounds are reconstructed from cells selected by the scenario; annotate density only here.
        density_axis.plot(
            [record["simulation_time_s"] for record in records],
            [record["congestion_zones"][name]["density_per_m2"] for record in records],
            label=name,
        )
    for pedestrian_id, (xs, ys) in trajectories.items():
        map_axis.plot(xs, ys, linewidth=1.0, alpha=0.75, label=f"ped {pedestrian_id}")
        map_axis.scatter(xs[0], ys[0], s=10)
    map_axis.plot(robot_x, robot_y, color="black", linewidth=2.5, label="robot")
    map_axis.scatter(robot_x[0], robot_y[0], marker="s", color="black", s=30)
    map_axis.scatter(robot_x[-1], robot_y[-1], marker="*", color="gold", edgecolor="black", s=90)
    origin_x, origin_y = first["grid"]["origin"]
    rows, cols = first["grid"]["shape"]
    resolution = first["grid"]["resolution_m"]
    map_axis.set_xlim(origin_x, origin_x + cols * resolution)
    map_axis.set_ylim(origin_y, origin_y + rows * resolution)
    map_axis.set_aspect("equal")
    map_axis.set_title(f"{first['scenario']} trajectories")
    map_axis.set_xlabel("world x [m]")
    map_axis.set_ylabel("world y [m]")
    density_axis.set_title("Ground Truth local pedestrian density")
    density_axis.set_xlabel("simulation time [s]")
    density_axis.set_ylabel("pedestrians / m²")
    density_axis.legend()
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
