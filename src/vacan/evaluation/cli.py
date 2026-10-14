"""Run the ATC paired sensing/navigation evaluation retained for Step 8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import bootstrap_mean_ci, mean, sensing_time_metrics


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def group_zone_rows(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_rows(path):
        grouped[row["timestamp"]][row["zone_id"]] = row
    return dict(grouped)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean(value: float, digits: int = 8) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def sensing_evaluation(
    ground_truth_path: Path,
    observed_path: Path,
    zones_path: Path,
    fixed_states: set[str],
    collaborative_states: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    zones = json.loads(zones_path.read_text(encoding="utf-8"))["features"]
    area = {feature["properties"]["zone_id"]: float(feature["properties"]["area_m2"]) for feature in zones}
    ground_truth = group_zone_rows(ground_truth_path)
    observed = group_zone_rows(observed_path)
    timestamps = sorted(set(ground_truth) & set(observed))
    time_rows: list[dict[str, Any]] = []
    raw: dict[str, dict[str, float]] = {}
    condition_metrics: dict[str, list[dict[str, float]]] = {"fixed_only": [], "collaborative_fresh": []}
    for timestamp in timestamps:
        fixed = sensing_time_metrics(ground_truth[timestamp], observed[timestamp], area, fixed_states)
        collaborative = sensing_time_metrics(
            ground_truth[timestamp], observed[timestamp], area, collaborative_states
        )
        condition_metrics["fixed_only"].append(fixed)
        condition_metrics["collaborative_fresh"].append(collaborative)
        raw[timestamp] = {
            key: collaborative[key] - fixed[key]
            for key in ("coverage_fraction", "hotspot_recall", "density_mae")
            if not math.isnan(fixed[key]) and not math.isnan(collaborative[key])
        }
        time_rows.append({
            "timestamp": timestamp,
            "fixed_coverage_fraction": clean(fixed["coverage_fraction"]),
            "collaborative_coverage_fraction": clean(collaborative["coverage_fraction"]),
            "coverage_delta": clean(collaborative["coverage_fraction"] - fixed["coverage_fraction"]),
            "fixed_hotspot_recall": clean(fixed["hotspot_recall"]),
            "collaborative_hotspot_recall": clean(collaborative["hotspot_recall"]),
            "hotspot_recall_delta": clean(collaborative["hotspot_recall"] - fixed["hotspot_recall"]),
            "fixed_density_mae": clean(fixed["density_mae"]),
            "collaborative_density_mae": clean(collaborative["density_mae"]),
            "actual_high_zone_count": int(fixed["actual_high_count"]),
        })

    summary: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {}
    for condition, records in condition_metrics.items():
        tp = sum(item["true_positive_count"] for item in records)
        actual = sum(item["actual_high_count"] for item in records)
        predicted = sum(item["predicted_high_count"] for item in records)
        aggregate[condition] = {
            "coverage_fraction": mean(item["coverage_fraction"] for item in records),
            "unavailable_fraction": mean(item["unavailable_fraction"] for item in records),
            "hotspot_recall": tp / actual if actual else math.nan,
            "hotspot_precision": tp / predicted if predicted else math.nan,
            "density_mae": mean(item["density_mae"] for item in records),
            "density_rmse": mean(item["density_rmse"] for item in records),
        }
    for metric in aggregate["fixed_only"]:
        fixed_value = aggregate["fixed_only"][metric]
        collaborative_value = aggregate["collaborative_fresh"][metric]
        summary.append({
            "metric": metric,
            "fixed_only": clean(fixed_value),
            "collaborative_fresh": clean(collaborative_value),
            "delta": clean(collaborative_value - fixed_value),
        })
    return time_rows, summary, raw


def planning_evaluation(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    pairs: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_rows(path):
        pairs[(row["timestamp"], row["robot_id"])][row["path_type"]] = row
    output: list[dict[str, Any]] = []
    blocks: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (timestamp, robot), pair in sorted(pairs.items()):
        baseline, aware = pair["BASELINE"], pair["CONGESTION_AWARE"]
        values = {
            "distance_delta_m": float(aware["distance_m"]) - float(baseline["distance_m"]),
            "actual_eta_delta_s": float(aware["actual_simulated_eta_s"]) - float(baseline["actual_simulated_eta_s"]),
            "exposure_delta": float(aware["ground_truth_exposure"]) - float(baseline["ground_truth_exposure"]),
        }
        output.append({
            "timestamp": timestamp,
            "robot_id": robot,
            **{key: clean(value) for key, value in values.items()},
            "route_changed": int(aware["path_signature"] != baseline["path_signature"]),
            "eta_better": int(values["actual_eta_delta_s"] < 0),
            "exposure_better": int(values["exposure_delta"] < 0),
            "aware_longer": int(values["distance_delta_m"] > 0),
        })
        for metric, value in values.items():
            blocks[timestamp][metric].append(value)
        blocks[timestamp]["route_changed_rate"].append(float(aware["path_signature"] != baseline["path_signature"]))
        blocks[timestamp]["eta_win_rate"].append(float(values["actual_eta_delta_s"] < 0))
        blocks[timestamp]["exposure_win_rate"].append(float(values["exposure_delta"] < 0))
        blocks[timestamp]["aware_longer_rate"].append(float(values["distance_delta_m"] > 0))
    block_means = {
        metric: {timestamp: mean(values[metric]) for timestamp, values in blocks.items()}
        for metric in next(iter(blocks.values()))
    }
    return output, block_means


def replanning_summary(path: Path) -> list[dict[str, Any]]:
    rows = read_rows(path)
    output: list[dict[str, Any]] = []
    for field in ("trigger_type", "robot_id", "source_sensor_id"):
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row[field]] += 1
        for value, count in sorted(counts.items()):
            output.append({
                "group": field,
                "value": value,
                "count": count,
                "fraction": clean(count / len(rows)),
            })
    return output


def bootstrap_rows(
    sensing_raw: dict[str, dict[str, float]],
    planning_blocks: dict[str, dict[str, float]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    sensing_names = {
        "coverage_fraction": ("sensing", "coverage_delta", "higher_is_better"),
        "hotspot_recall": ("sensing", "hotspot_recall_delta", "higher_is_better"),
        "density_mae": ("sensing", "density_mae_delta", "lower_is_better"),
    }
    for offset, (source, (family, name, direction)) in enumerate(sensing_names.items()):
        values = {timestamp: metrics[source] for timestamp, metrics in sensing_raw.items() if source in metrics}
        estimate, low, high = bootstrap_mean_ci(values, samples=samples, confidence=confidence, seed=seed + offset)
        results.append({"family": family, "metric": name, "estimate": clean(estimate), "ci_low": clean(low), "ci_high": clean(high), "confidence": confidence, "direction": direction})
    planning_direction = {
        "distance_delta_m": "lower_is_better",
        "actual_eta_delta_s": "lower_is_better",
        "exposure_delta": "lower_is_better",
        "route_changed_rate": "descriptive",
        "eta_win_rate": "higher_is_better",
        "exposure_win_rate": "higher_is_better",
        "aware_longer_rate": "lower_is_better",
    }
    for offset, (metric, values) in enumerate(planning_blocks.items(), start=100):
        estimate, low, high = bootstrap_mean_ci(values, samples=samples, confidence=confidence, seed=seed + offset)
        results.append({"family": "planning", "metric": metric, "estimate": clean(estimate), "ci_low": clean(low), "ci_high": clean(high), "confidence": confidence, "direction": planning_direction[metric]})
    return results


def svg_line_figure(time_rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 960, 360
    left, right, top, bottom = 70, 930, 35, 300
    fixed = [float(row["fixed_coverage_fraction"]) for row in time_rows]
    collaborative = [float(row["collaborative_coverage_fraction"]) for row in time_rows]
    def points(values: list[float]) -> str:
        return " ".join(
            f"{left + index * (right-left)/(len(values)-1):.2f},{bottom - value*(bottom-top):.2f}"
            for index, value in enumerate(values)
        )
    grid = []
    for tick in range(0, 11, 2):
        value = tick / 10
        y = bottom - value * (bottom - top)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#d9e1e5"/><text x="58" y="{y+4:.2f}" text-anchor="end">{value:.1f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text{{font:12px sans-serif;fill:#34444d}} .title{{font-size:17px;font-weight:700}} .axis{{font-size:11px}}</style>
<rect width="100%" height="100%" fill="white"/><text class="title" x="{left}" y="22">Fresh observation coverage over time</text>
{''.join(grid)}<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#34444d"/>
<polyline points="{points(fixed)}" fill="none" stroke="#5aa7ff" stroke-width="2"/>
<polyline points="{points(collaborative)}" fill="none" stroke="#0e9f8f" stroke-width="2.5"/>
<line x1="680" y1="20" x2="710" y2="20" stroke="#5aa7ff" stroke-width="2"/><text x="717" y="24">Fixed only</text>
<line x1="795" y1="20" x2="825" y2="20" stroke="#0e9f8f" stroke-width="2.5"/><text x="832" y="24">Collaborative fresh</text>
<text transform="translate(17 220) rotate(-90)">Walkable-area fraction</text><text x="430" y="340">One-minute bins (09:33–20:26 JST)</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def svg_planning_figure(paired: list[dict[str, Any]], path: Path) -> None:
    metrics = [
        ("ETA", sum(int(row["eta_better"]) for row in paired)),
        ("Exposure", sum(int(row["exposure_better"]) for row in paired)),
        ("Distance not longer", sum(not int(row["aware_longer"]) for row in paired)),
    ]
    total = len(paired)
    bars = []
    for index, (label, count) in enumerate(metrics):
        y = 90 + index * 80
        fraction = count / total
        bars.append(f'<text x="170" y="{y+18}" text-anchor="end">{label}</text><rect x="190" y="{y}" width="600" height="28" rx="4" fill="#e4eaed"/><rect x="190" y="{y}" width="{600*fraction:.2f}" height="28" rx="4" fill="#0e9f8f"/><text x="{200+600*fraction:.2f}" y="{y+19}" fill="#102126">{fraction:.1%} ({count}/{total})</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="360" viewBox="0 0 960 360">
<style>text{{font:14px sans-serif;fill:#34444d}} .title{{font-size:18px;font-weight:700}} .note{{font-size:11px;fill:#687982}}</style>
<rect width="100%" height="100%" fill="white"/><text class="title" x="55" y="35">Congestion-aware planning outcomes</text>
<text class="note" x="55" y="55">Share of 396 paired robot-time comparisons with the preferred outcome</text>{''.join(bars)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(sensing: list[dict[str, Any]], bootstrap: list[dict[str, Any]], path: Path) -> None:
    sensing_lines = ["| Metric | Fixed only | Collaborative fresh | Delta |", "|---|---:|---:|---:|"]
    sensing_lines.extend(f'| {row["metric"]} | {row["fixed_only"]} | {row["collaborative_fresh"]} | {row["delta"]} |' for row in sensing)
    planning_lines = ["| Planning metric | Estimate | 95% CI |", "|---|---:|---:|"]
    planning_lines.extend(
        f'| {row["metric"]} | {row["estimate"]} | [{row["ci_low"]}, {row["ci_high"]}] |'
        for row in bootstrap if row["family"] == "planning"
    )
    path.write_text("# ATC Step 8 foundation evaluation tables\n\n## Sensing reconstruction\n\n" + "\n".join(sensing_lines) + "\n\n## Planning bootstrap estimates\n\n" + "\n".join(planning_lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = {key: Path(value) for key, value in config["inputs"].items()}
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    time_rows, sensing_summary, sensing_raw = sensing_evaluation(
        inputs["ground_truth_congestion"], inputs["observed_congestion"], inputs["zones"],
        set(config["conditions"]["fixed_only_states"]),
        set(config["conditions"]["collaborative_fresh_states"]),
    )
    planning_paired, planning_blocks = planning_evaluation(inputs["path_metrics"])
    replan_rows = replanning_summary(inputs["replanning_events"])
    bootstrap = config["bootstrap"]
    intervals = bootstrap_rows(
        sensing_raw, planning_blocks, samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["confidence"]), seed=int(bootstrap["seed"]),
    )
    write_csv(output / "sensing_timeseries.csv", time_rows)
    write_csv(output / "sensing_summary.csv", sensing_summary)
    write_csv(output / "planning_paired.csv", planning_paired)
    write_csv(output / "replanning_summary.csv", replan_rows)
    write_csv(output / "bootstrap_intervals.csv", intervals)
    svg_line_figure(time_rows, output / "figure_sensing_coverage.svg")
    svg_planning_figure(planning_paired, output / "figure_planning_outcomes.svg")
    markdown_table(sensing_summary, intervals, output / "tables.md")
    paper_output = Path(config["paper_output_directory"])
    paper_output.mkdir(parents=True, exist_ok=True)
    for name in ("figure_sensing_coverage.svg", "figure_planning_outcomes.svg", "tables.md"):
        shutil.copy2(output / name, paper_output / name)
    manifest = {
        "evaluation_id": config["evaluation_id"],
        "scenario_id": config["scenario_id"],
        "method": {"bootstrap": bootstrap, "conditions": config["conditions"]},
        "counts": {
            "sensing_timestamps": len(time_rows),
            "planning_pairs": len(planning_paired),
            "replanning_events": sum(int(row["count"]) for row in replan_rows if row["group"] == "trigger_type"),
        },
        "inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in inputs.items()},
        "outputs": sorted(path.name for path in output.iterdir() if path.name != "manifest.json"),
        "paper_outputs": sorted(str(path) for path in paper_output.iterdir()),
        "notes": config["notes"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
