"""Dependency-free Dijkstra planner with congestion and uncertainty costs."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .graph import ZoneGraph


@dataclass(frozen=True)
class PlanningWeights:
    congestion_weight: float
    density_reference: float
    density_cost_cap: float
    state_penalties: dict[str, float]


def normalized_density(record: dict[str, str], weights: PlanningWeights) -> float:
    value = record.get("density_per_m2", "")
    if not value:
        return 0.0
    if weights.density_reference <= 0:
        raise ValueError("density_reference must be positive")
    return min(float(value) / weights.density_reference, weights.density_cost_cap)


def node_factor(record: dict[str, str], weights: PlanningWeights) -> float:
    state = record["observation_state"]
    if state not in weights.state_penalties:
        raise ValueError(f"missing state penalty for {state}")
    return (
        1.0
        + weights.congestion_weight * normalized_density(record, weights)
        + weights.state_penalties[state]
    )


def shortest_path(
    graph: ZoneGraph,
    start: str,
    goal: str,
    observations: dict[str, dict[str, str]] | None = None,
    weights: PlanningWeights | None = None,
) -> tuple[list[str], float]:
    if start not in graph.nodes or goal not in graph.nodes:
        raise ValueError("start and goal must exist in the zone graph")
    queue: list[tuple[float, str]] = [(0.0, start)]
    costs = {start: 0.0}
    previous: dict[str, str] = {}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current == goal:
            break
        for neighbor, distance in graph.adjacency[current]:
            edge_cost = distance
            if observations is not None and weights is not None:
                first_factor = node_factor(observations[current], weights)
                second_factor = node_factor(observations[neighbor], weights)
                edge_cost *= (first_factor + second_factor) / 2
            candidate = cost + edge_cost
            if candidate < costs.get(neighbor, math.inf):
                costs[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    if goal not in costs:
        raise ValueError(f"no path from {start} to {goal}")
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path, costs[goal]
