"""Run the configured Step 3 ATC analytics pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vacan.datasets.atc.map import load_ros_map

from .congestion import aggregate_congestion
from .coverage import build_observed_congestion, write_coverage_plan
from .zones import filter_zones_by_trajectory_support, generate_grid_zones, write_zones


def run(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    ros_map = load_ros_map(Path(inputs["map_yaml"]))
    candidate_zones = generate_grid_zones(
        ros_map,
        zone_size_m=float(config["zone_size_m"]),
        minimum_walkable_fraction=float(config["minimum_walkable_fraction"]),
        traversable_states=tuple(config.get("traversable_occupancy_states", ["FREE"])),
    )
    zones, zone_selection = filter_zones_by_trajectory_support(
        Path(inputs["trajectories"]),
        candidate_zones,
        minimum_rows_per_zone=int(config.get("minimum_trajectory_rows_per_zone", 1)),
    )
    write_zones(zones, output / "zones.geojson")
    classification = config.get("density_classification")
    if classification is None:
        thresholds = config["density_thresholds"]
        empirical_quantiles = None
    else:
        reference = classification["absolute_reference_thresholds"]
        thresholds = {
            "low_upper_exclusive": reference[0],
            "medium_upper_exclusive": reference[1],
        }
        empirical_quantiles = (
            float(classification["low_to_medium_quantile"]),
            float(classification["medium_to_high_quantile"]),
        ) if classification["mode"] == "empirical_positive_quantiles" else None
    aggregation = aggregate_congestion(
        Path(inputs["trajectories"]),
        output / "ground_truth_congestion.csv",
        zones,
        time_bin_seconds=int(config["time_bin_seconds"]),
        low_upper_exclusive=float(thresholds["low_upper_exclusive"]),
        medium_upper_exclusive=float(thresholds["medium_upper_exclusive"]),
        empirical_positive_quantiles=empirical_quantiles,
    )
    plan = build_observed_congestion(
        output / "ground_truth_congestion.csv",
        output / "observed_congestion.csv",
        output / "coverage_summary.csv",
        output / "observations.csv",
        zones.zones,
        fixed_area_fraction=float(config["fixed_coverage_area_fraction"]),
        selection_key=str(config["analysis_id"]),
        robot_zones_per_bin=int(config["robot_zones_per_bin"]),
        stale_after_seconds=int(config["stale_after_seconds"]),
    )
    write_coverage_plan(plan, output / "coverage_plan.json")
    report = {
        "analysis_id": config["analysis_id"],
        "scenario_id": config["scenario_id"],
        "assumptions": config["notes"],
        "zone_selection": zone_selection,
        "aggregation": aggregation,
        "coverage": {
            "fixed_zone_count": len(plan["fixed_zone_ids"]),
            "blind_zone_count": len(plan["blind_zone_ids"]),
            "robot_observation_event_count": plan["robot_observation_event_count"],
        },
    }
    (output / "analytics_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
