"""Simulator-independent Stage 7 dataset export and debug figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                            for record in records), encoding="utf-8")


def write_representation_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key.replace("::", "__").replace(".", "_"): value
                               for key, value in arrays.items()})


def render_decision_figure(path: Path, original_map: np.ndarray, modified_map: np.ndarray,
                           result: dict[str, Any], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for axis, layer, name in zip(
        axes[:3], (original_map, modified_map, modified_map - original_map),
        ("Original crowd input", "Counterfactual crowd input", "Difference"),
    ):
        image = axis.imshow(layer, origin="lower", cmap="coolwarm" if name == "Difference" else "viridis")
        axis.set_title(name); figure.colorbar(image, ax=axis, fraction=0.046)
    original_action = result["original"]["action"]
    modified_action = result["counterfactual"]["action"]
    x = np.arange(2); width = 0.36
    axes[3].bar(x - width / 2, original_action, width, label="original")
    axes[3].bar(x + width / 2, modified_action, width, label="counterfactual")
    axes[3].set_xticks(x, ["linear", "angular"]); axes[3].legend()
    axes[3].set_title(f"Action L2={result['differences']['action_l2']:.4g}")
    figure.suptitle(title); figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)


def render_episode_figure(path: Path, original: list[dict[str, Any]],
                          counterfactual: list[dict[str, Any]], result: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for records, label, color in ((original, "original", "#1f77b4"),
                                  (counterfactual, "counterfactual", "#d62728")):
        xy = np.asarray([record["position_xy_m"] for record in records], dtype=float)
        axes[0].plot(xy[:, 0], xy[:, 1], label=label, color=color)
    axes[0].set_aspect("equal"); axes[0].legend(); axes[0].set_title("Closed-loop trajectories")
    labels = ["time", "length", "collision", "social", "smoothness", "failure"]
    keys = ["travel_time_s", "path_length_m", "collision_events",
            "social_violation_integral_s", "smoothness", "terminal_failure"]
    x = np.arange(len(keys)); width = 0.36
    axes[1].bar(x - width / 2, [result["original_cost"][key] for key in keys], width,
                label="original")
    axes[1].bar(x + width / 2, [result["counterfactual_cost"][key] for key in keys], width,
                label="counterfactual")
    axes[1].set_xticks(x, labels, rotation=30); axes[1].legend()
    axes[1].set_title(f"Delta J CF={result['delta_j_cf']:.3f}")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)


def render_cctv_case_figure(path: Path, correct: list[dict[str, Any]],
                            corrupted: list[dict[str, Any]], cell: list[int]) -> None:
    """Render exact/corrupted CCTV count and resulting paired trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = f"{int(cell[0])},{int(cell[1])}"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for records, label, color in ((correct, "correct CCTV", "#1f77b4"),
                                  (corrupted, "corrupted CCTV", "#d62728")):
        xy = np.asarray([record["position_xy_m"] for record in records], dtype=float)
        axes[0].plot(xy[:, 0], xy[:, 1], label=label, color=color)
        timed = [(record["simulation_time_s"], record["costmap_diagnostics"])
                 for record in records if record.get("costmap_diagnostics")]
        axes[1].plot(
            [item[0] for item in timed],
            [float(item[1]["cctv_counts"][key]) for item in timed],
            label=label, color=color,
        )
    truth = [(record["simulation_time_s"], record["costmap_diagnostics"])
             for record in corrupted if record.get("costmap_diagnostics")]
    axes[1].plot(
        [item[0] for item in truth],
        [float(item[1]["cctv_ground_truth_counts"][key]) for item in truth],
        label="privileged exact count", color="#2ca02c", linestyle="--",
    )
    axes[0].set_aspect("equal"); axes[0].set_title("Paired closed-loop paths")
    axes[0].legend(); axes[1].legend(); axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Pedestrian count")
    axes[1].set_title(f"CCTV cell [{cell[0]},{cell[1]}]")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)


def render_freshness_case_figure(path: Path, correct: list[dict[str, Any]],
                                 erroneous: list[dict[str, Any]], cells: list[list[int]]) -> None:
    """Render paired paths and cell freshness after an intention error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for records, label, color in ((correct, "correct intention", "#1f77b4"),
                                  (erroneous, "incorrect intention", "#d62728")):
        xy = np.asarray([record["position_xy_m"] for record in records], dtype=float)
        axes[0].plot(xy[:, 0], xy[:, 1], label=label, color=color)
        timed = [(record["simulation_time_s"], record["costmap_diagnostics"])
                 for record in records if record.get("costmap_diagnostics")]
        for cell in cells:
            axes[1].plot(
                [item[0] for item in timed],
                [float(item[1]["freshness"][cell[0]][cell[1]]) for item in timed],
                label=f"{label} [{cell[0]},{cell[1]}]",
                linestyle="-" if label == "correct intention" else "--",
            )
    axes[0].set_aspect("equal"); axes[0].set_title("Paired closed-loop paths")
    axes[0].legend(); axes[1].legend(fontsize=8); axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("Simulation time (s)"); axes[1].set_ylabel("Freshness")
    axes[1].set_title("Intention-induced freshness updates")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)
