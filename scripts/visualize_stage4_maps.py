#!/usr/bin/env python3
"""Render independent Stage 4 3D and 2D mapping diagnostics."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vacan-mpl")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from vacan.mapping import OccupancyMap2D, VoxelMap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--stage2-config", type=Path, required=True)
    args = parser.parse_args()
    voxel = VoxelMap.load(args.directory / "voxel_map.npz")
    grid = OccupancyMap2D.load(args.directory / "occupancy_2d.npz")
    stage2 = yaml.safe_load(args.stage2_config.read_text(encoding="utf-8"))
    robot = stage2["robot"]["start_xyz"]

    indices = np.argwhere(voxel.occupancy == 1)
    if len(indices) > 50000:
        indices = indices[::math.ceil(len(indices) / 50000)]
    points = np.asarray([voxel.voxel_center(*index) for index in indices]) if len(indices) else np.empty((0, 3))
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    if len(points):
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.35, color="slategray")
    axis.scatter([robot[0]], [robot[1]], [robot[2]], marker="*", s=120, color="tab:red", label="Nova Carter")
    axis.set(xlabel="world X [m]", ylabel="world Y [m]", zlabel="world Z [m]",
             title=f"{voxel.environment}: static collision voxels")
    axis.legend()
    figure.savefig(args.directory / "voxel_map_3d.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    layers = [(grid.raw, "Raw occupancy"), (grid.unknown, "Unknown mask"),
              (grid.inflated, "Inflated obstacles"), (grid.policy_obstacle, "Policy obstacle")]
    for axis, (layer, title) in zip(axes, layers, strict=True):
        axis.imshow(layer, origin="lower", vmin=0, vmax=1, cmap="gray_r", interpolation="nearest")
        axis.set(title=title, xlabel="column", ylabel="row")
    figure.suptitle(f"{voxel.environment}: aligned 2D map ({grid.geometry.resolution_m:g} m/cell)")
    figure.savefig(args.directory / "occupancy_2d.png", dpi=160)
    plt.close(figure)

    native_path = args.directory / "official_omap_2d_raw.npz"
    if native_path.is_file():
        with np.load(native_path, allow_pickle=False) as archive:
            dimensions = tuple(map(int, archive["dimensions"]))
            native = archive["buffer"].reshape(dimensions[1], dimensions[0])
        display = np.full(native.shape, 0.5, dtype=float)
        display[np.isclose(native, 0.0)] = 1.0
        display[np.isclose(native, 1.0)] = 0.0
        figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
        axis.imshow(display, origin="lower", vmin=0, vmax=1, cmap="gray",
                    interpolation="nearest")
        axis.set(title=f"{voxel.environment}: official 0.2 m occupancy\n"
                       "black=occupied, white=free, gray=unknown",
                 xlabel="generator column", ylabel="generator row")
        figure.savefig(args.directory / "official_omap_2d_raw.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()
