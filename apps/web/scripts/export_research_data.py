"""Export a compact, browser-safe bundle from Step 3 and Step 4 artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYTICS = PROJECT_ROOT / "data/processed/atc/atc-20121114/analytics"
PLANNING = PROJECT_ROOT / "data/processed/atc/atc-20121114/planning"
CAVIAR = PROJECT_ROOT / "data/processed/caviar/walkbyshop1"
OUTPUT = WEB_ROOT / "source-data/dashboard.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def grouped(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows(path):
        result[row["timestamp"]][row["zone_id"]] = row
    return dict(result)


def number(value: str, digits: int = 6) -> float | None:
    return None if value == "" else round(float(value), digits)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def caviar_payload(atc_zones: list[dict[str, object]]) -> dict[str, object]:
    """Export aggregate CAVIAR evidence without raw paths or person-level records."""
    collection = json.loads((CAVIAR / "zones.geojson").read_text(encoding="utf-8"))
    camera_order = ["LISBON_CORRIDOR", "LISBON_FRONT"]
    counts: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    frame_count = 0
    for row in rows(CAVIAR / "zone_counts.csv"):
        frame_index = int(row["frame"])
        counts[row["zone_id"]][frame_index][row["camera_id"]] = int(
            row["pedestrian_count"]
        )
        frame_count = max(frame_count, frame_index + 1)

    zones = []
    for feature in collection["features"]:
        zone_id = feature["properties"]["zone_id"]
        ring = feature["geometry"]["coordinates"][0]
        observations = []
        totals = {camera: 0 for camera in camera_order}
        best_frame = 0
        best_score = (-1, -1, -1, 0)
        best_counts = [0, 0]
        for frame_index in range(frame_count):
            values = [counts[zone_id][frame_index].get(camera, 0) for camera in camera_order]
            for camera, value in zip(camera_order, values):
                totals[camera] += value
            if sum(values):
                observations.append([frame_index, *values])
            score = (sum(values), int(all(value > 0 for value in values)), min(values), -frame_index)
            if score > best_score:
                best_score = score
                best_frame = frame_index
                best_counts = values
        zones.append({
            "id": zone_id,
            "polygon": ring,
            "totalCounts": [totals[camera] for camera in camera_order],
            "peakCount": best_score[0],
            "bestFrame": best_frame,
            "bestCounts": best_counts,
            "observations": observations,
        })

    zones.sort(key=lambda zone: zone["id"])

    # Demo-only cross-dataset mapping: normalize each ATC centroid into the
    # 3×10 CAVIAR grid, then snap to the nearest zone with video evidence.
    atc_x = [float(zone["centroid"][0]) for zone in atc_zones]
    atc_y = [float(zone["centroid"][1]) for zone in atc_zones]
    x_min, x_max = min(atc_x), max(atc_x)
    y_min, y_max = min(atc_y), max(atc_y)
    observed_caviar = []
    for zone in zones:
        grid = zone["id"].replace("CAVIAR-R", "").split("-C")
        if zone["observations"]:
            observed_caviar.append((int(grid[0]), int(grid[1]), zone))
    demo_mappings = {}
    for atc_zone in atc_zones:
        x, y = (float(value) for value in atc_zone["centroid"])
        target_column = round(((x - x_min) / (x_max - x_min)) * 2)
        target_row = round(((y - y_min) / (y_max - y_min)) * 9)
        _, _, caviar_zone = min(
            observed_caviar,
            key=lambda item: (
                (item[0] - target_row) ** 2 + (item[1] - target_column) ** 2,
                item[0], item[1],
            ),
        )
        stable_seed = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(str(atc_zone["id"]))
        )
        selected_observation = caviar_zone["observations"][
            stable_seed % len(caviar_zone["observations"])
        ]
        camera = 0 if selected_observation[1] >= selected_observation[2] else 1
        frame_index = selected_observation[0]
        demo_mappings[atc_zone["id"]] = {
            "caviarZone": caviar_zone["id"],
            "camera": camera,
            "frame": frame_index,
            "method": "normalized-centroid-nearest-observed-zone",
        }
    return {
        "scenario": "CAVIAR Lisbon · WalkByShop1",
        "frameCount": frame_count,
        "fps": 25,
        "zoneCount": len(zones),
        "cameras": [
            {"id": "LISBON_CORRIDOR", "label": "Corridor view", "media": "/local-media/caviar/corridor.mp4"},
            {"id": "LISBON_FRONT", "label": "Front view", "media": "/local-media/caviar/front.mp4"},
        ],
        "zones": zones,
        "demoMappings": demo_mappings,
        "mappingMode": "DEMO_ONLY_NOT_SPATIALLY_ALIGNED",
        "localOnly": True,
        "notice": "Demo only: CAVIAR is not spatially aligned with the ATC venue. Raw CCTV remains local and read-only.",
    }


def main() -> None:
    collection = json.loads((ANALYTICS / "zones.geojson").read_text(encoding="utf-8"))
    zones = []
    for feature in collection["features"]:
        properties = feature["properties"]
        ring = feature["geometry"]["coordinates"][0]
        points = ring[:-1]
        zones.append({
            "id": properties["zone_id"],
            "row": properties["grid_row"],
            "column": properties["grid_column"],
            "area": round(float(properties["area_m2"]), 4),
            "walkableFraction": round(float(properties["walkable_fraction"]), 4),
            "polygon": ring,
            "centroid": [
                round(sum(point[0] for point in points) / len(points), 4),
                round(sum(point[1] for point in points) / len(points), 4),
            ],
        })
    zones.sort(key=lambda zone: zone["id"])
    zone_ids = [zone["id"] for zone in zones]

    ground_truth = grouped(ANALYTICS / "ground_truth_congestion.csv")
    observed = grouped(ANALYTICS / "observed_congestion.csv")
    coverage = {row["timestamp"]: row for row in rows(ANALYTICS / "coverage_summary.csv")}
    timestamps = sorted(set(ground_truth) & set(observed))
    frames = []
    for timestamp in timestamps:
        truth_at, observed_at = ground_truth[timestamp], observed[timestamp]
        coverage_at = coverage[timestamp]
        truth_density = [float(truth_at[zone_id]["density_per_m2"]) for zone_id in zone_ids]
        weighted_mean = sum(
            density * zone["area"] for density, zone in zip(truth_density, zones)
        ) / sum(zone["area"] for zone in zones)
        frames.append({
            "t": timestamp,
            "g": [round(value, 6) for value in truth_density],
            "d": [round(weighted_mean, 6), round(percentile(truth_density, 0.9), 6)],
            "o": [[
                number(observed_at[zone_id]["density_per_m2"]),
                observed_at[zone_id]["congestion_level"][0],
                observed_at[zone_id]["observation_state"][0],
                number(observed_at[zone_id]["confidence"]),
                number(observed_at[zone_id]["observation_age_s"], 0),
                observed_at[zone_id]["source"],
            ] for zone_id in zone_ids],
            "c": [number(coverage_at[key], 3) for key in (
                "fixed_area_m2", "robot_recovered_area_m2", "stale_area_m2",
                "unknown_area_m2", "total_walkable_area_m2",
            )],
        })

    metrics_rows = rows(PLANNING / "path_metrics.csv")
    planning: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in metrics_rows:
        key = f'{row["timestamp"]}|{row["robot_id"]}'
        planning[key][row["path_type"]] = {
            "zones": row["path_signature"].split(">"),
            "distance": number(row["distance_m"], 3),
            "plannedEta": number(row["planned_eta_s"], 3),
            "actualEta": number(row["actual_simulated_eta_s"], 3),
            "exposure": number(row["ground_truth_exposure"], 5),
            "unknownDistance": number(row["unknown_distance_m"], 3),
            "staleDistance": number(row["stale_distance_m"], 3),
            "cost": number(row["planning_cost"], 3),
        }

    events = []
    for row in rows(PLANNING / "replanning_events.csv"):
        events.append({
            "t": row["timestamp"], "id": row["event_id"], "robot": row["robot_id"],
            "type": row["trigger_type"], "zone": row["trigger_zone_id"],
            "source": row["source_sensor_id"], "oldPath": row["old_path_signature"].split(">"),
            "newPath": row["new_path_signature"].split(">"),
            "etaDelta": number(row["eta_delta_s"], 3),
            "distanceDelta": number(row["distance_delta_m"], 3), "reason": row["reason"],
        })

    analytics_report = json.loads((ANALYTICS / "analytics_report.json").read_text(encoding="utf-8"))
    planning_report = json.loads((PLANNING / "planning_report.json").read_text(encoding="utf-8"))
    summaries = rows(PLANNING / "comparison_summary.csv")
    payload = {
        "meta": {
            "title": "VACaN · Indoor Crowd & Robot Navigation",
            "scenario": analytics_report["scenario_id"],
            "source": "ATC Shopping Center · 2012-11-14",
            "zoneCount": len(zones), "timeBinCount": len(frames), "binSeconds": 60,
            "thresholds": [
                analytics_report["aggregation"]["low_upper_exclusive"],
                analytics_report["aggregation"]["medium_upper_exclusive"],
            ],
            "classification": analytics_report["aggregation"]["threshold_mode"],
            "fixedZoneCount": planning_report["graph"]["node_count"] - analytics_report["coverage"]["blind_zone_count"],
            "blindZoneCount": analytics_report["coverage"]["blind_zone_count"],
            "disclaimer": "Congestion levels are within-day empirical quantiles; ETA is a simulation proxy.",
        },
        "zoneIds": zone_ids, "zones": zones, "frames": frames,
        "planningTimestamps": sorted({row["timestamp"] for row in metrics_rows}),
        "robots": sorted({row["robot_id"] for row in metrics_rows}),
        "planning": planning, "events": events, "comparisonSummary": summaries,
        "caviar": caviar_payload(zones),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "size_bytes": OUTPUT.stat().st_size, "frames": len(frames)}))


if __name__ == "__main__":
    main()
