"""Generate baseline/congestion-aware routes and replanning events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .evaluation import PathMetrics, evaluate_path
from .graph import load_zone_graph
from .planner import PlanningWeights, shortest_path


def _load_time_records(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["timestamp"]][row["zone_id"]] = row
    return dict(grouped)


def _signature(path: list[str]) -> str:
    return ">".join(path)


def _write_path_segments(
    writer: csv.writer,
    graph,
    timestamp: str,
    robot_id: str,
    path_type: str,
    path: list[str],
    planning_cost: float,
    metrics: PathMetrics,
) -> None:
    for sequence, (first, second) in enumerate(zip(path, path[1:])):
        a, b = graph.nodes[first], graph.nodes[second]
        distance = graph.edge_distance(first, second)
        share = distance / metrics.distance_m if metrics.distance_m else 0
        writer.writerow([
            timestamp, robot_id, path_type, f"{robot_id}-{path_type}-{sequence:03d}", sequence,
            f"{a.x:.6f}", f"{a.y:.6f}", f"{b.x:.6f}", f"{b.y:.6f}",
            f"{distance:.6f}", f"{planning_cost * share:.6f}",
            f"{metrics.planned_eta_s * share:.6f}",
        ])


def run(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    graph = load_zone_graph(Path(inputs["zones"]))
    observed = _load_time_records(Path(inputs["observed_congestion"]))
    ground_truth = _load_time_records(Path(inputs["ground_truth_congestion"]))
    analytics_report = json.loads(Path(inputs["analytics_report"]).read_text(encoding="utf-8"))
    density_reference = float(analytics_report["aggregation"]["medium_upper_exclusive"])
    weights = PlanningWeights(
        congestion_weight=float(config["congestion_weight"]),
        density_reference=density_reference,
        density_cost_cap=float(config["density_cost_cap"]),
        state_penalties={key: float(value) for key, value in config["state_penalties"].items()},
    )
    timestamps = sorted(set(observed) & set(ground_truth))
    interval = int(config["replan_interval_bins"])
    selected = timestamps[::interval]
    if timestamps and selected[-1] != timestamps[-1]:
        selected.append(timestamps[-1])

    paths_file = output / "paths.csv"
    metrics_file = output / "path_metrics.csv"
    events_file = output / "replanning_events.csv"
    metrics_rows: list[dict[str, object]] = []
    event_rows: list[list[object]] = []
    previous: dict[str, tuple[list[str], PathMetrics, dict[str, dict[str, str]]]] = {}
    with paths_file.open("w", newline="", encoding="utf-8") as path_stream:
        path_writer = csv.writer(path_stream)
        path_writer.writerow([
            "timestamp", "robot_id", "path_type", "segment_id", "sequence",
            "x1", "y1", "x2", "y2", "distance_m", "congestion_cost", "eta_segment_s",
        ])
        for timestamp in selected:
            observed_at = observed[timestamp]
            truth_at = ground_truth[timestamp]
            for robot in config["robots"]:
                robot_id = robot["robot_id"]
                start, goal = robot["start_zone_id"], robot["goal_zone_id"]
                baseline_path, baseline_cost = shortest_path(graph, start, goal)
                aware_path, aware_cost = shortest_path(graph, start, goal, observed_at, weights)
                for path_type, path, cost in (
                    ("BASELINE", baseline_path, baseline_cost),
                    ("CONGESTION_AWARE", aware_path, aware_cost),
                ):
                    metrics = evaluate_path(
                        graph, path, observed_at, truth_at, weights,
                        nominal_speed_m_s=float(config["nominal_speed_m_s"]),
                        actual_density_slowdown=float(config["actual_density_slowdown"]),
                    )
                    _write_path_segments(path_writer, graph, timestamp, robot_id, path_type, path, cost, metrics)
                    metrics_rows.append({
                        "timestamp": timestamp, "robot_id": robot_id, "path_type": path_type,
                        "start_zone_id": start, "goal_zone_id": goal,
                        "path_signature": _signature(path), "planning_cost": cost,
                        **metrics.__dict__,
                    })
                    if path_type == "CONGESTION_AWARE":
                        if robot_id in previous and previous[robot_id][0] != path:
                            old_path, old_metrics, old_observed = previous[robot_id]
                            changed_zones = set(old_path) | set(path)
                            def change_score(zone_id: str) -> float:
                                old = old_observed[zone_id]
                                new = observed_at[zone_id]
                                old_density = float(old["density_per_m2"] or 0)
                                new_density = float(new["density_per_m2"] or 0)
                                state_change = old["observation_state"] != new["observation_state"]
                                return abs(new_density - old_density) + (density_reference if state_change else 0)
                            trigger_zone = max(sorted(changed_zones), key=change_score)
                            trigger = observed_at[trigger_zone]
                            trigger_type = (
                                "OBSERVATION_STATE_CHANGE"
                                if old_observed[trigger_zone]["observation_state"] != trigger["observation_state"]
                                else "CONGESTION_CHANGE"
                            )
                            event_id = "E-" + hashlib.sha1(
                                f"{timestamp}:{robot_id}:{_signature(path)}".encode()
                            ).hexdigest()[:10]
                            event_rows.append([
                                timestamp, event_id, robot_id, trigger_type, trigger_zone,
                                trigger["source"], _signature(old_path), _signature(path),
                                f"{metrics.planned_eta_s - old_metrics.planned_eta_s:.6f}",
                                f"{metrics.distance_m - old_metrics.distance_m:.6f}",
                                f"Route changed after {trigger_type.lower().replace('_', ' ')} in {trigger_zone}",
                            ])
                        previous[robot_id] = (path, metrics, observed_at)

    metric_fields = [
        "timestamp", "robot_id", "path_type", "start_zone_id", "goal_zone_id",
        "path_signature", "planning_cost", "distance_m", "planned_eta_s",
        "actual_simulated_eta_s", "ground_truth_exposure", "unknown_distance_m",
        "stale_distance_m", "segment_count",
    ]
    with metrics_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics_rows)
    with events_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "timestamp", "event_id", "robot_id", "trigger_type", "trigger_zone_id",
            "source_sensor_id", "old_path_signature", "new_path_signature", "eta_delta_s",
            "distance_delta_m", "reason",
        ])
        writer.writerows(event_rows)

    paired: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in metrics_rows:
        paired[(str(row["timestamp"]), str(row["robot_id"]))][str(row["path_type"])] = row

    def summarize(robot_id: str | None) -> dict[str, int | float | str]:
        comparisons = []
        for (_, pair_robot), pair in paired.items():
            if robot_id is not None and pair_robot != robot_id:
                continue
            baseline = pair["BASELINE"]
            aware = pair["CONGESTION_AWARE"]
            comparisons.append((baseline, aware))
        count = len(comparisons)
        distance_deltas = [float(a["distance_m"]) - float(b["distance_m"]) for b, a in comparisons]
        eta_deltas = [
            float(a["actual_simulated_eta_s"]) - float(b["actual_simulated_eta_s"])
            for b, a in comparisons
        ]
        exposure_deltas = [
            float(a["ground_truth_exposure"]) - float(b["ground_truth_exposure"])
            for b, a in comparisons
        ]
        return {
            "robot_id": robot_id or "ALL",
            "comparison_count": count,
            "different_route_count": sum(b["path_signature"] != a["path_signature"] for b, a in comparisons),
            "aware_longer_count": sum(delta > 1e-9 for delta in distance_deltas),
            "actual_eta_better_count": sum(delta < 0 for delta in eta_deltas),
            "exposure_better_count": sum(delta < 0 for delta in exposure_deltas),
            "mean_distance_delta_m": sum(distance_deltas) / count if count else 0.0,
            "mean_actual_eta_delta_s": sum(eta_deltas) / count if count else 0.0,
            "mean_exposure_delta": sum(exposure_deltas) / count if count else 0.0,
        }

    summaries = [summarize(None)] + [summarize(robot["robot_id"]) for robot in config["robots"]]
    summary_fields = list(summaries[0])
    with (output / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    report = {
        "planning_id": config["planning_id"],
        "scenario_id": config["scenario_id"],
        "notes": config["notes"],
        "graph": {"node_count": len(graph.nodes), "directed_edge_count": sum(map(len, graph.adjacency.values()))},
        "planning_timestamps": len(selected),
        "robot_count": len(config["robots"]),
        "path_comparisons": len(metrics_rows) // 2,
        "replanning_event_count": len(event_rows),
        "density_reference": density_reference,
        "comparison_summary": summaries[0],
    }
    (output / "planning_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
