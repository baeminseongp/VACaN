"""Deterministic observation masking and information-freshness fusion."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from vacan.models import ObservationState

from .congestion import CONGESTION_HEADER
from .zones import Zone


def select_fixed_zones(
    zones: tuple[Zone, ...], target_area_fraction: float, selection_key: str
) -> tuple[str, ...]:
    if not 0 <= target_area_fraction <= 1:
        raise ValueError("target_area_fraction must be in [0, 1]")
    total_area = sum(zone.walkable_area_m2 for zone in zones)
    target_area = total_area * target_area_fraction
    ordered = sorted(
        zones,
        key=lambda zone: hashlib.sha256(f"{selection_key}:{zone.zone_id}".encode()).digest(),
    )
    selected: list[str] = []
    selected_area = 0.0
    for zone in ordered:
        if selected_area >= target_area:
            break
        selected.append(zone.zone_id)
        selected_area += zone.walkable_area_m2
    return tuple(sorted(selected))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_observed_congestion(
    ground_truth_path: Path,
    output_path: Path,
    coverage_summary_path: Path,
    observations_path: Path,
    zones: tuple[Zone, ...],
    *,
    fixed_area_fraction: float,
    selection_key: str,
    robot_zones_per_bin: int,
    stale_after_seconds: int,
) -> dict[str, object]:
    fixed = set(select_fixed_zones(zones, fixed_area_fraction, selection_key))
    blind = sorted(zone.zone_id for zone in zones if zone.zone_id not in fixed)
    area = {zone.zone_id: zone.walkable_area_m2 for zone in zones}
    centroid = {zone.zone_id: zone.centroid for zone in zones}
    last_robot: dict[str, tuple[datetime, dict[str, str]]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_count = 0

    with ground_truth_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output, coverage_summary_path.open("w", newline="", encoding="utf-8") as summary, observations_path.open(
        "w", newline="", encoding="utf-8"
    ) as observations:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(output, fieldnames=CONGESTION_HEADER)
        writer.writeheader()
        summary_writer = csv.writer(summary)
        summary_writer.writerow([
            "timestamp", "fixed_area_m2", "robot_recovered_area_m2", "stale_area_m2",
            "unknown_area_m2", "total_walkable_area_m2",
        ])
        observation_writer = csv.writer(observations)
        observation_writer.writerow([
            "timestamp", "sensor_id", "sensor_type", "x", "y", "heading_deg",
            "visible_zone_ids", "coverage_wkt", "confidence",
        ])

        current_timestamp: str | None = None
        time_rows: list[dict[str, str]] = []
        time_index = 0

        def flush_time() -> None:
            nonlocal time_index, event_count
            if current_timestamp is None:
                return
            current_time = _parse_time(current_timestamp)
            if blind and robot_zones_per_bin > 0:
                observed_now = {
                    blind[(time_index * robot_zones_per_bin + offset) % len(blind)]
                    for offset in range(min(robot_zones_per_bin, len(blind)))
                }
            else:
                observed_now = set()
            ground_truth_by_zone = {row["zone_id"]: row for row in time_rows}
            for zone_id in sorted(observed_now):
                row = ground_truth_by_zone[zone_id]
                last_robot[zone_id] = (current_time, row.copy())
                x, y = centroid[zone_id]
                observation_writer.writerow([
                    current_timestamp, "ROBOT-SIM-01", "ROBOT", f"{x:.6f}", f"{y:.6f}",
                    "0", zone_id, "", "1.000000",
                ])
                event_count += 1

            area_by_state = {state.value: 0.0 for state in ObservationState}
            for truth in time_rows:
                zone_id = truth["zone_id"]
                if zone_id in fixed:
                    row = truth.copy()
                    row.update({"source": "CCTV_MASK", "confidence": "1.000000"})
                    state = ObservationState.FIXED_COVERED
                elif zone_id in last_robot:
                    last_time, last_value = last_robot[zone_id]
                    age = (current_time - last_time).total_seconds()
                    row = last_value.copy()
                    row["timestamp"] = current_timestamp
                    row["last_observed"] = last_time.isoformat().replace("+00:00", "Z")
                    row["observation_age_s"] = f"{age:.0f}"
                    row["source"] = "ROBOT-SIM-01"
                    row["confidence"] = f"{math.exp(-age / stale_after_seconds):.6f}"
                    state = (
                        ObservationState.ROBOT_RECOVERED
                        if age <= stale_after_seconds
                        else ObservationState.STALE
                    )
                else:
                    row = {key: "" for key in CONGESTION_HEADER}
                    row.update({
                        "timestamp": current_timestamp,
                        "zone_id": zone_id,
                        "congestion_level": "UNKNOWN",
                        "source": "NONE",
                    })
                    state = ObservationState.UNKNOWN
                row["observation_state"] = state.value
                writer.writerow(row)
                area_by_state[state.value] += area[zone_id]
            summary_writer.writerow([
                current_timestamp,
                f"{area_by_state['FIXED_COVERED']:.6f}",
                f"{area_by_state['ROBOT_RECOVERED']:.6f}",
                f"{area_by_state['STALE']:.6f}",
                f"{area_by_state['UNKNOWN']:.6f}",
                f"{sum(area.values()):.6f}",
            ])
            time_index += 1

        for row in reader:
            timestamp = row["timestamp"]
            if current_timestamp is not None and timestamp != current_timestamp:
                flush_time()
                time_rows = []
            current_timestamp = timestamp
            time_rows.append(row)
        flush_time()

    plan = {
        "selection": "deterministic_sha256_order",
        "selection_key": selection_key,
        "requested_fixed_area_fraction": fixed_area_fraction,
        "fixed_zone_ids": sorted(fixed),
        "blind_zone_ids": blind,
        "robot_zones_per_bin": robot_zones_per_bin,
        "stale_after_seconds": stale_after_seconds,
        "robot_observation_event_count": event_count,
    }
    return plan


def write_coverage_plan(plan: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
