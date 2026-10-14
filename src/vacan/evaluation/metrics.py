"""Dependency-free paired metrics and bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else math.nan


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    block_values: Mapping[str, float], *, samples: int, confidence: float, seed: int
) -> tuple[float, float, float]:
    """Return mean and percentile CI by resampling timestamp blocks."""
    values = list(block_values.values())
    if not values:
        return math.nan, math.nan, math.nan
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    generator = random.Random(seed)
    size = len(values)
    bootstrapped = [
        sum(values[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    tail = (1 - confidence) / 2
    return mean(values), percentile(bootstrapped, tail), percentile(bootstrapped, 1 - tail)


def sensing_time_metrics(
    ground_truth: Mapping[str, Mapping[str, str]],
    observed: Mapping[str, Mapping[str, str]],
    area_by_zone: Mapping[str, float],
    valid_states: set[str],
) -> dict[str, float]:
    total_area = sum(area_by_zone.values())
    available = [zone for zone, row in observed.items() if row["observation_state"] in valid_states]
    available_area = sum(area_by_zone[zone] for zone in available)
    errors = [
        float(observed[zone]["density_per_m2"]) - float(ground_truth[zone]["density_per_m2"])
        for zone in available
    ]
    actual_high = {zone for zone, row in ground_truth.items() if row["congestion_level"] == "HIGH"}
    predicted_high = {
        zone for zone in available if observed[zone]["congestion_level"] == "HIGH"
    }
    true_positive = len(actual_high & predicted_high)
    return {
        "coverage_fraction": available_area / total_area,
        "unavailable_fraction": 1 - available_area / total_area,
        "hotspot_recall": true_positive / len(actual_high) if actual_high else math.nan,
        "hotspot_precision": true_positive / len(predicted_high) if predicted_high else math.nan,
        "density_mae": mean(abs(error) for error in errors),
        "density_rmse": math.sqrt(mean(error * error for error in errors)),
        "available_zone_count": float(len(available)),
        "actual_high_count": float(len(actual_high)),
        "predicted_high_count": float(len(predicted_high)),
        "true_positive_count": float(true_positive),
    }
