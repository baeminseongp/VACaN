"""Streaming temporal aggregation of trajectory snapshots into zone density."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

from vacan.models import congestion_level

from .zones import ZoneIndex


CONGESTION_HEADER = [
    "timestamp", "zone_id", "pedestrian_count", "density_per_m2",
    "congestion_level", "observation_state", "source", "confidence",
    "last_observed", "observation_age_s",
]


def _iso_timestamp(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must be in [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def aggregate_congestion(
    trajectory_path: Path,
    output_path: Path,
    zone_index: ZoneIndex,
    *,
    time_bin_seconds: int,
    low_upper_exclusive: float,
    medium_upper_exclusive: float,
    empirical_positive_quantiles: tuple[float, float] | None = None,
) -> dict[str, int | float]:
    """Average simultaneous per-frame counts; do not count unique visitors per bin."""
    if time_bin_seconds <= 0:
        raise ValueError("time_bin_seconds must be positive")
    frame_counts: dict[int, int] = {}
    count_sums: dict[tuple[int, str], int] = {}
    current_time: float | None = None
    current_people: dict[str, set[str]] = {}
    input_rows = outside_zone_rows = frame_total = 0

    def flush_frame() -> None:
        nonlocal frame_total
        if current_time is None:
            return
        bin_start = math.floor(current_time / time_bin_seconds) * time_bin_seconds
        frame_counts[bin_start] = frame_counts.get(bin_start, 0) + 1
        frame_total += 1
        for zone_id, person_ids in current_people.items():
            key = (bin_start, zone_id)
            count_sums[key] = count_sums.get(key, 0) + len(person_ids)

    with trajectory_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            input_rows += 1
            sample_time = float(row["source_time_s"])
            if current_time is not None and sample_time < current_time:
                raise ValueError("trajectory rows must be sorted by source_time_s")
            if current_time is None or sample_time != current_time:
                flush_frame()
                current_time = sample_time
                current_people = {}
            zone = zone_index.zone_for_point(float(row["x_m"]), float(row["y_m"]))
            if zone is None:
                outside_zone_rows += 1
                continue
            current_people.setdefault(zone.zone_id, set()).add(row["person_id"])
    flush_frame()

    density_by_key: dict[tuple[int, str], tuple[float, float]] = {}
    for bin_start, frame_count in frame_counts.items():
        for zone in zone_index.zones:
            mean_count = count_sums.get((bin_start, zone.zone_id), 0) / frame_count
            density_by_key[(bin_start, zone.zone_id)] = (
                mean_count,
                mean_count / zone.walkable_area_m2,
            )
    threshold_mode = "absolute"
    if empirical_positive_quantiles is not None:
        low_quantile, high_quantile = empirical_positive_quantiles
        if not 0 <= low_quantile < high_quantile <= 1:
            raise ValueError("empirical quantiles must be ordered in [0, 1]")
        positive = sorted(density for _, density in density_by_key.values() if density > 0)
        if positive:
            low_upper_exclusive = _quantile(positive, low_quantile)
            medium_upper_exclusive = _quantile(positive, high_quantile)
            threshold_mode = "empirical_positive_quantiles"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CONGESTION_HEADER)
        for bin_start in sorted(frame_counts):
            frame_count = frame_counts[bin_start]
            timestamp = _iso_timestamp(bin_start)
            for zone in zone_index.zones:
                mean_count, density = density_by_key[(bin_start, zone.zone_id)]
                level = congestion_level(density, low_upper_exclusive, medium_upper_exclusive)
                writer.writerow([
                    timestamp, zone.zone_id, f"{mean_count:.6f}", f"{density:.6f}",
                    level.value, "FIXED_COVERED", "ATC_FULL_GROUND_TRUTH", "1.000000",
                    timestamp, "0",
                ])
    return {
        "input_rows": input_rows,
        "outside_selected_zone_rows": outside_zone_rows,
        "frame_count": frame_total,
        "time_bin_count": len(frame_counts),
        "zone_count": len(zone_index.zones),
        "threshold_mode": threshold_mode,
        "low_upper_exclusive": low_upper_exclusive,
        "medium_upper_exclusive": medium_upper_exclusive,
    }
