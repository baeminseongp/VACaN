"""Build a connected four-neighbour graph from regular analysis zones."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GraphNode:
    zone_id: str
    grid_row: int
    grid_column: int
    x: float
    y: float


@dataclass(frozen=True)
class ZoneGraph:
    nodes: dict[str, GraphNode]
    adjacency: dict[str, tuple[tuple[str, float], ...]]

    def edge_distance(self, first: str, second: str) -> float:
        for neighbor, distance in self.adjacency[first]:
            if neighbor == second:
                return distance
        raise KeyError(f"zones are not adjacent: {first}, {second}")


def _ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    points = ring[:-1] if ring and ring[0] == ring[-1] else ring
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def load_zone_graph(path: Path) -> ZoneGraph:
    collection = json.loads(path.read_text(encoding="utf-8"))
    nodes: dict[str, GraphNode] = {}
    cells: dict[tuple[int, int], str] = {}
    for feature in collection["features"]:
        properties = feature["properties"]
        zone_id = properties["zone_id"]
        x, y = _ring_centroid(feature["geometry"]["coordinates"][0])
        node = GraphNode(
            zone_id=zone_id,
            grid_row=int(properties["grid_row"]),
            grid_column=int(properties["grid_column"]),
            x=x,
            y=y,
        )
        nodes[zone_id] = node
        cells[(node.grid_row, node.grid_column)] = zone_id
    adjacency_lists: dict[str, list[tuple[str, float]]] = {zone_id: [] for zone_id in nodes}
    for zone_id, node in nodes.items():
        for row_delta, column_delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_id = cells.get((node.grid_row + row_delta, node.grid_column + column_delta))
            if neighbor_id is None:
                continue
            neighbor = nodes[neighbor_id]
            distance = math.hypot(neighbor.x - node.x, neighbor.y - node.y)
            adjacency_lists[zone_id].append((neighbor_id, distance))
    return ZoneGraph(
        nodes=nodes,
        adjacency={key: tuple(sorted(value)) for key, value in adjacency_lists.items()},
    )
