"""Route metrics evaluated against observations and full ground truth."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import ZoneGraph
from .planner import PlanningWeights, normalized_density


@dataclass(frozen=True)
class PathMetrics:
    distance_m: float
    planned_eta_s: float
    actual_simulated_eta_s: float
    ground_truth_exposure: float
    unknown_distance_m: float
    stale_distance_m: float
    segment_count: int


def evaluate_path(
    graph: ZoneGraph,
    path: list[str],
    observed: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, str]],
    weights: PlanningWeights,
    *,
    nominal_speed_m_s: float,
    actual_density_slowdown: float,
) -> PathMetrics:
    distance_total = planned_eta = actual_eta = exposure = unknown = stale = 0.0
    for first, second in zip(path, path[1:]):
        distance = graph.edge_distance(first, second)
        observed_density = (
            normalized_density(observed[first], weights)
            + normalized_density(observed[second], weights)
        ) / 2
        true_density = (
            float(ground_truth[first]["density_per_m2"])
            + float(ground_truth[second]["density_per_m2"])
        ) / 2
        true_normalized = min(true_density / weights.density_reference, weights.density_cost_cap)
        state_penalty = (
            weights.state_penalties[observed[first]["observation_state"]]
            + weights.state_penalties[observed[second]["observation_state"]]
        ) / 2
        distance_total += distance
        planned_eta += distance / nominal_speed_m_s * (1 + observed_density + state_penalty)
        actual_eta += distance / nominal_speed_m_s * (1 + actual_density_slowdown * true_normalized)
        exposure += distance * true_density
        if "UNKNOWN" in {observed[first]["observation_state"], observed[second]["observation_state"]}:
            unknown += distance
        if "STALE" in {observed[first]["observation_state"], observed[second]["observation_state"]}:
            stale += distance
    return PathMetrics(
        distance_m=distance_total,
        planned_eta_s=planned_eta,
        actual_simulated_eta_s=actual_eta,
        ground_truth_exposure=exposure,
        unknown_distance_m=unknown,
        stale_distance_m=stale,
        segment_count=max(0, len(path) - 1),
    )
