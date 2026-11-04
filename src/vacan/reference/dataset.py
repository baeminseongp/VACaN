"""Machine-readable Stage 6 dataset and development visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vacan.mapping.static_map import OccupancyMap2D


def write_matched_dataset(output: Path, metadata: dict[str, Any],
                          drl: list[dict[str, Any]], reference: list[dict[str, Any]],
                          comparison: dict[str, Any], *,
                          drl_ground_truth: list[dict[str, Any]] | None = None,
                          reference_ground_truth: list[dict[str, Any]] | None = None,
                          planning_history: list[dict[str, Any]] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, records in (("drl-trajectory.jsonl", drl),
                          ("reference-trajectory.jsonl", reference)):
        (output / name).write_text(
            "".join(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                    for record in records), encoding="utf-8")
    for name, records in (
        ("ground-truth-drl.jsonl", drl_ground_truth or []),
        ("ground-truth-reference.jsonl", reference_ground_truth or []),
        ("reference-planning.jsonl", planning_history or []),
    ):
        (output / name).write_text(
            "".join(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                    for record in records), encoding="utf-8")
    np.savez_compressed(
        output / "aligned-trajectories.npz",
        drl_time_s=np.asarray([r["simulation_time_s"] for r in drl], dtype=np.float64),
        drl_xy_m=np.asarray([r["position_xy_m"] for r in drl], dtype=np.float32),
        reference_time_s=np.asarray([r["simulation_time_s"] for r in reference], dtype=np.float64),
        reference_xy_m=np.asarray([r["position_xy_m"] for r in reference], dtype=np.float32),
    )


def render_debug_figures(output: Path, static_map: OccupancyMap2D,
                         drl: list[dict[str, Any]], reference: list[dict[str, Any]],
                         comparison: dict[str, Any], start_xy: tuple[float, float],
                         goal_xy: tuple[float, float], *,
                         pedestrian_xy: list[tuple[float, float]] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = static_map.geometry
    x0, y0 = geometry.origin_xy
    x1, y1 = x0 + geometry.size_xy[0], y0 + geometry.size_xy[1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(static_map.policy_obstacle, origin="lower", extent=(x0, x1, y0, y1),
              cmap="Greys", alpha=0.55)
    for records, color, label in ((drl, "#d62728", "DRL"),
                                  (reference, "#1f77b4", "Reference")):
        points = np.asarray([record["position_xy_m"] for record in records], dtype=float)
        if len(points):
            ax.plot(points[:, 0], points[:, 1], color=color, label=label, linewidth=2)
    ax.scatter(*start_xy, marker="o", color="green", label="Start", zorder=5)
    ax.scatter(*goal_xy, marker="*", color="gold", edgecolor="black", s=140,
               label="Goal", zorder=5)
    if pedestrian_xy:
        people = np.asarray(pedestrian_xy, dtype=float)
        ax.scatter(people[:, 0], people[:, 1], marker="x", color="#ff7f0e",
                   label="Pedestrians at reset", zorder=5)
    candidates = comparison.get("candidate_divergence_points", [])
    if candidates and drl:
        times = np.asarray([record["simulation_time_s"] for record in drl])
        points = np.asarray([record["position_xy_m"] for record in drl])
        indices = [int(np.argmin(np.abs(times - candidate["simulation_time_s"])))
                   for candidate in candidates]
        ax.scatter(points[indices, 0], points[indices, 1], facecolors="none",
                   edgecolors="magenta", label="Divergence candidates", zorder=6)
    ax.set_aspect("equal"); ax.set_xlabel("world x [m]"); ax.set_ylabel("world y [m]")
    ax.legend(); ax.set_title("Office DRL vs full-information reference")
    fig.tight_layout(); fig.savefig(output / "trajectory-comparison.png", dpi=160); plt.close(fig)

    components = comparison["component_differences"]
    labels = [key for key in components if key != "total"]
    fig, ax = plt.subplots(figsize=(8, 4))
    if "drl_cost" in comparison and "reference_cost" in comparison:
        x = np.arange(len(labels)); width = 0.38
        ax.bar(x - width / 2, [comparison["drl_cost"][key] for key in labels],
               width, label="DRL", color="#d62728")
        ax.bar(x + width / 2, [comparison["reference_cost"][key] for key in labels],
               width, label="Reference", color="#1f77b4")
        ax.set_xticks(x, labels); ax.legend(); ax.set_ylabel("raw component value")
    else:
        ax.bar(labels, [components[key] for key in labels], color="#9467bd")
        ax.axhline(0.0, color="black", linewidth=0.8); ax.set_ylabel("DRL - Reference")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Cost breakdown, delta J={comparison['delta_j']:.3f}")
    fig.tight_layout(); fig.savefig(output / "cost-comparison.png", dpi=160); plt.close(fig)
