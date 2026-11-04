"""Aggregate successful Stage 6 runs without double-counting seed repeats."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std_population": float(statistics.pstdev(values)),
    }


def aggregate_runs(root: Path) -> dict[str, Any]:
    selected: dict[int, tuple[Path, dict[str, Any]]] = {}
    excluded = []
    for path in sorted(root.glob("matched-seed*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "pass":
            excluded.append({"path": str(path.parent), "reason": "status_not_pass"})
            continue
        seed = int(summary["seed"])
        previous = selected.get(seed)
        if previous is not None:
            excluded.append({"path": str(previous[0].parent), "reason": "older_seed_repeat"})
        selected[seed] = (path, summary)
    if not selected:
        raise ValueError(f"no successful Stage 6 runs below {root}")
    rows = []
    for seed, (path, summary) in sorted(selected.items()):
        comparison = summary["comparison"]
        drl, reference = comparison["drl_cost"], comparison["reference_cost"]
        rows.append({
            "seed": seed, "artifact": str(path.parent),
            "drl_success": bool(summary["drl_success"]),
            "reference_success": bool(summary["reference_success"]),
            "drl": drl, "reference": reference,
            "delta_j": float(comparison["delta_j"]),
            "inefficient": bool(comparison["inefficient"]),
            "outcome_class": comparison["outcome_class"],
            "planning_calls": int(summary["planning_calls"]),
            "infeasible_planning_calls": int(summary["infeasible_planning_calls"]),
        })
    metrics = ("travel_time_s", "path_length_m", "social_violation_integral_s",
               "smoothness", "total")
    return {
        "schema_version": 1, "scenario": "office", "unique_episode_count": len(rows),
        "seeds": [row["seed"] for row in rows], "episodes": rows,
        "drl_success_rate": sum(row["drl_success"] for row in rows) / len(rows),
        "reference_success_rate": sum(row["reference_success"] for row in rows) / len(rows),
        "drl_collision_rate": sum(row["drl"]["collision_events"] > 0 for row in rows) / len(rows),
        "reference_collision_rate": sum(row["reference"]["collision_events"] > 0
                                        for row in rows) / len(rows),
        "drl_metrics": {metric: _stats([float(row["drl"][metric]) for row in rows])
                        for metric in metrics},
        "reference_metrics": {
            metric: _stats([float(row["reference"][metric]) for row in rows])
            for metric in metrics
        },
        "delta_j": _stats([row["delta_j"] for row in rows]),
        "inefficient_episode_fraction": sum(row["inefficient"] for row in rows) / len(rows),
        "outcome_counts": {
            outcome: sum(row["outcome_class"] == outcome for row in rows)
            for outcome in ("both_successful", "drl_success_reference_unsuccessful",
                            "drl_unsuccessful_reference_success", "both_unsuccessful")
        },
        "reference_infeasible_planning_rate": (
            sum(row["infeasible_planning_calls"] for row in rows)
            / sum(row["planning_calls"] for row in rows)
        ),
        "excluded_runs": excluded,
    }
