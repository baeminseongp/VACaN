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
