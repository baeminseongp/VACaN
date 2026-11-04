"""Simulator-independent Stage 7 dataset export and debug figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _render_paired_paths_and_separation(
    path_axis: Any,
    separation_axis: Any,
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    first_label: str,
    second_label: str,
    goal_xy_m: tuple[float, float] | None = None,
) -> None:
    """Draw honest path evidence and make small spatial differences inspectable."""
    first_xy = np.asarray([record["position_xy_m"] for record in first], dtype=float)
    second_xy = np.asarray([record["position_xy_m"] for record in second], dtype=float)
    first_time = np.asarray([record["simulation_time_s"] for record in first], dtype=float)
    second_time = np.asarray([record["simulation_time_s"] for record in second], dtype=float)
    colors = ("#1f77b4", "#d62728")
    path_axis.plot(first_xy[:, 0], first_xy[:, 1], label=first_label, color=colors[0],
                   linewidth=2.2)
    path_axis.plot(second_xy[:, 0], second_xy[:, 1], label=second_label, color=colors[1],
                   linewidth=2.2)
    path_axis.scatter(*first_xy[0], marker="o", color="#2ca02c", s=45, zorder=4,
                      label="start")
    if goal_xy_m is not None:
        path_axis.scatter(
            goal_xy_m[0], goal_xy_m[1], marker="X", color="#ffbf00",
            edgecolor="black", s=70, zorder=6, label="goal",
        )
    path_axis.scatter(*first_xy[-1], marker="*", color=colors[0], s=90, zorder=4)
    path_axis.scatter(*second_xy[-1], marker="*", color=colors[1], s=90, zorder=4)

    overlap = (second_time >= first_time[0] - 1e-9) & (
        second_time <= first_time[-1] + 1e-9
    )
    common_time = second_time[overlap]
    comparison_xy = second_xy[overlap]
    if common_time.size:
        interpolated = np.column_stack([
            np.interp(common_time, first_time, first_xy[:, axis]) for axis in (0, 1)
        ])
        separation = np.linalg.norm(comparison_xy - interpolated, axis=1)
        maximum_index = int(np.argmax(separation))
        maximum = float(separation[maximum_index])
        first_point, second_point = interpolated[maximum_index], comparison_xy[maximum_index]
        path_axis.plot([first_point[0], second_point[0]], [first_point[1], second_point[1]],
                       color="#9467bd", linestyle="--", linewidth=1.8,
                       label=f"max separation {maximum:.3f} m")
        path_axis.scatter([first_point[0], second_point[0]],
                          [first_point[1], second_point[1]], color="#9467bd", s=28,
                          zorder=5)
        separation_axis.plot(common_time, separation, color="#9467bd", linewidth=2.2)
        separation_axis.scatter(common_time[maximum_index], maximum, color="#9467bd",
                                s=38, zorder=4)
        separation_axis.annotate(
            f"max {maximum:.3f} m\n@ {common_time[maximum_index]:.1f} s",
            (common_time[maximum_index], maximum), xytext=(8, 8),
            textcoords="offset points", fontsize=9,
        )
    separation_axis.axhline(0.75, color="#7f7f7f", linestyle=":", linewidth=1.3,
                            label="case threshold 0.75 m")
    separation_axis.set_xlabel("Simulation time (s)")
    separation_axis.set_ylabel("Paired path separation (m)")
    separation_axis.set_title("Spatial divergence over time")
    separation_axis.legend(fontsize=8)
    separation_axis.grid(alpha=0.25)
    path_axis.set_xlabel("World x (m)")
    path_axis.set_ylabel("World y (m)")
    path_axis.set_aspect("equal")
    path_axis.set_title("Paired closed-loop paths")
    path_axis.legend(fontsize=8)
    path_axis.grid(alpha=0.2)


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

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _render_paired_paths_and_separation(
        axes[0], axes[1], original, counterfactual,
        first_label="original", second_label="counterfactual",
    )
    labels = ["time", "length", "collision", "social", "smoothness", "failure"]
    keys = ["travel_time_s", "path_length_m", "collision_events",
            "social_violation_integral_s", "smoothness", "terminal_failure"]
    x = np.arange(len(keys)); width = 0.36
    axes[2].bar(x - width / 2, [result["original_cost"][key] for key in keys], width,
                label="original")
    axes[2].bar(x + width / 2, [result["counterfactual_cost"][key] for key in keys], width,
                label="counterfactual")
    axes[2].set_xticks(x, labels, rotation=30); axes[2].legend()
    axes[2].set_title(f"Delta J CF={result['delta_j_cf']:.3f}")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)


def render_cctv_case_figure(path: Path, correct: list[dict[str, Any]],
                            corrupted: list[dict[str, Any]],
                            corrupted_ground_truth: list[dict[str, Any]],
                            cells: list[list[int]],
                            goal_xy_m: tuple[float, float]) -> None:
    """Render exact/corrupted added-CCTV cells and resulting trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [tuple(map(int, cell)) for cell in cells]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _render_paired_paths_and_separation(
        axes[0], axes[1], correct, corrupted,
        first_label="correct CCTV", second_label="corrupted CCTV",
        goal_xy_m=goal_xy_m,
    )
    for records, label, color in ((correct, "correct CCTV", "#1f77b4"),
                                  (corrupted, "corrupted CCTV", "#d62728")):
        timed = [(record["simulation_time_s"], record["costmap_diagnostics"])
                 for record in records if record.get("costmap_diagnostics")]
        axes[2].plot(
            [item[0] for item in timed],
            [sum(float(item[1]["cost_after"][row][column])
                 for row, column in selected) for item in timed],
            label=label, color=color,
        )
    truth = {round(float(item["simulation_time_s"]), 6): item
             for item in corrupted_ground_truth}
    timed_corrupted = [record for record in corrupted if record.get("costmap_diagnostics")]
    axes[2].step(
        [record["simulation_time_s"] for record in timed_corrupted],
        [sum(float(truth[round(float(record["simulation_time_s"]), 6)]
                   ["ground_truth_density"][row][column]) for row, column in selected)
         for record in timed_corrupted],
        label="privileged exact count", color="#2ca02c", linestyle="--", where="post",
    )
    axes[2].legend(); axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel("Pedestrian count")
    axes[2].set_title(f"Added CCTV direct-route cells ({len(cells)} cells)")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)


def render_coverage_gap_case_figure(
    path: Path, restored: list[dict[str, Any]], missing: list[dict[str, Any]],
    missing_ground_truth: list[dict[str, Any]], cells: list[list[int]],
    goal_xy_m: tuple[float, float],
) -> None:
    """Render a path pair and missing-coverage estimate versus exact crowd."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [tuple(map(int, cell)) for cell in cells]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _render_paired_paths_and_separation(
        axes[0], axes[1], restored, missing,
        first_label="coverage restored", second_label="missing CCTV coverage",
        goal_xy_m=goal_xy_m,
    )
    for records, label, color in ((restored, "coverage restored", "#1f77b4"),
                                  (missing, "missing CCTV coverage", "#d62728")):
        timed = [(record["simulation_time_s"], record["costmap_diagnostics"])
                 for record in records if record.get("costmap_diagnostics")]
        axes[2].plot(
            [item[0] for item in timed],
            [sum(float(item[1]["cost_after"][row][column])
                 for row, column in selected) for item in timed],
            label=f"{label} input", color=color,
        )
    truth = {round(float(item["simulation_time_s"]), 6): item
             for item in missing_ground_truth}
    timed_missing = [record for record in missing if record.get("costmap_diagnostics")]
    axes[2].step(
        [record["simulation_time_s"] for record in timed_missing],
        [sum(float(truth[round(float(record["simulation_time_s"]), 6)]
                   ["ground_truth_density"][row][column]) for row, column in selected)
         for record in timed_missing],
        label="privileged exact crowd", color="#2ca02c", linestyle="--", where="post",
    )
    axes[2].legend(fontsize=8); axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel("Aggregate crowd cost / count")
    axes[2].set_title(f"Uncovered central region ({len(cells)} cells)")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)


def render_freshness_case_figure(path: Path, correct: list[dict[str, Any]],
                                 direct_only: list[dict[str, Any]],
                                 erroneous: list[dict[str, Any]], cells: list[list[int]],
                                 goal_xy_m: tuple[float, float]) -> None:
    """Render intention-direct and freshness-mediated three-branch evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _render_paired_paths_and_separation(
        axes[0], axes[1], correct, erroneous,
        first_label="correct intention", second_label="incorrect intention",
        goal_xy_m=goal_xy_m,
    )
    direct_xy = np.asarray([record["position_xy_m"] for record in direct_only], dtype=float)
    axes[0].plot(direct_xy[:, 0], direct_xy[:, 1], label="wrong intention, no reset",
                 color="#ff7f0e", linestyle="--", linewidth=2.0)
    axes[0].legend(fontsize=8)
    for records, label, color, style in (
        (correct, "correct intention", "#1f77b4", "-"),
        (direct_only, "wrong intention, no reset", "#ff7f0e", "--"),
        (erroneous, "wrong intention + freshness reset", "#d62728", "-"),
    ):
        timed = [(record["simulation_time_s"], record["costmap_diagnostics"])
                 for record in records if record.get("costmap_diagnostics")]
        axes[2].plot(
            [item[0] for item in timed],
            [float(np.mean([item[1]["freshness"][cell[0]][cell[1]]
                            for cell in cells])) for item in timed],
            label=f"{label} mean", color=color,
            linestyle=style,
        )
    axes[2].legend(fontsize=8); axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("Simulation time (s)"); axes[2].set_ylabel("Mean freshness")
    axes[2].set_title(f"Affected intention cells ({len(cells)} cells)")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)


def render_behavior_failure_figure(
    path: Path,
    normal: list[dict[str, Any]],
    failing: list[dict[str, Any]],
    normal_behavior: dict[str, Any],
    failing_behavior: dict[str, Any],
    goal_xy_m: tuple[float, float],
) -> None:
    """Render successful navigation against stuck/oscillating DRL behavior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    _render_paired_paths_and_separation(
        axes[0], axes[1], normal, failing,
        first_label="normal DRL", second_label="failing DRL",
        goal_xy_m=goal_xy_m,
    )

    goal = np.asarray(goal_xy_m, dtype=float)
    for records, label, color in (
        (normal, "normal DRL", "#1f77b4"),
        (failing, "failing DRL", "#d62728"),
    ):
        time = np.asarray([item["simulation_time_s"] for item in records], dtype=float)
        xy = np.asarray([item["position_xy_m"] for item in records], dtype=float)
        distance = np.linalg.norm(xy - goal, axis=1)
        axes[2].plot(time, distance, label=label, color=color, linewidth=2.0)
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel("Goal distance (m)")
    axes[2].set_title("Progress toward goal")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    failing_time = np.asarray(
        [item["simulation_time_s"] for item in failing], dtype=float
    )
    failing_action = np.asarray(
        [item.get("action", (0.0, 0.0)) for item in failing], dtype=float
    )
    axes[3].plot(failing_time, failing_action[:, 0], label="linear velocity",
                 color="#2ca02c", linewidth=1.8)
    axes[3].plot(failing_time, failing_action[:, 1], label="angular velocity",
                 color="#d62728", linewidth=1.8)
    stuck = failing_behavior.get("first_stuck_window")
    oscillation = failing_behavior.get("first_oscillation_window")
    if stuck is not None:
        axes[3].axvspan(stuck["start_time_s"], stuck["end_time_s"],
                        color="#ff7f0e", alpha=0.18, label="stuck window")
    if oscillation is not None:
        axes[3].axvspan(oscillation["start_time_s"], oscillation["end_time_s"],
                        color="#9467bd", alpha=0.18, label="oscillation window")
    axes[3].axhline(0.0, color="#7f7f7f", linewidth=0.8)
    axes[3].set_xlabel("Simulation time (s)")
    axes[3].set_ylabel("Executed command")
    axes[3].set_title(
        "Failing DRL controls\n"
        f"stuck={failing_behavior['stuck_detected']}, "
        f"oscillation={failing_behavior['oscillation_detected']}"
    )
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.25)
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160); plt.close(figure)
