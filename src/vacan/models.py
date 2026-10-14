"""Canonical enums and deterministic state transitions."""

from __future__ import annotations

from enum import StrEnum


class SensorType(StrEnum):
    CCTV = "CCTV"
    ROBOT = "ROBOT"


class ObservationState(StrEnum):
    FIXED_COVERED = "FIXED_COVERED"
    ROBOT_RECOVERED = "ROBOT_RECOVERED"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class CongestionLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


def observation_state(
    sensor_type: SensorType | None,
    observation_age_s: float | None,
    stale_after_s: float,
) -> ObservationState:
    """Classify coverage without confusing missing information with free space."""
    if sensor_type is None or observation_age_s is None:
        return ObservationState.UNKNOWN
    if observation_age_s < 0:
        raise ValueError("observation_age_s cannot be negative")
    if stale_after_s <= 0:
        raise ValueError("stale_after_s must be positive")
    if observation_age_s > stale_after_s:
        return ObservationState.STALE
    if sensor_type is SensorType.CCTV:
        return ObservationState.FIXED_COVERED
    return ObservationState.ROBOT_RECOVERED


def congestion_level(
    density_per_m2: float | None,
    low_upper_exclusive: float,
    medium_upper_exclusive: float,
) -> CongestionLevel:
    """Map continuous density to a configurable display classification."""
    if density_per_m2 is None:
        return CongestionLevel.UNKNOWN
    if density_per_m2 < 0:
        raise ValueError("density_per_m2 cannot be negative")
    if not 0 <= low_upper_exclusive < medium_upper_exclusive:
        raise ValueError("density thresholds must be ordered and non-negative")
    if density_per_m2 < low_upper_exclusive:
        return CongestionLevel.LOW
    if density_per_m2 < medium_upper_exclusive:
        return CongestionLevel.MEDIUM
    return CongestionLevel.HIGH
