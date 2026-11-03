"""Deterministic Stage 5 curriculum population construction."""

from __future__ import annotations

import copy
import math
from typing import Any


def _distance_xy(first: list[float], second: list[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]),
                      float(first[1]) - float(second[1]))


def resolve_pedestrian_population(
    stage1: dict[str, Any], routes: dict[str, Any], target_count: int,
    minimum_spawn_separation_m: float = 0.8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Subset or expand people using only existing NavMesh-validated route points."""
    if target_count < 0:
        raise ValueError("target pedestrian count cannot be negative")
    result_stage1 = copy.deepcopy(stage1)
    result_routes = copy.deepcopy(routes)
    base_people = list(stage1.get("pedestrians", []))
    if not base_people and target_count:
        raise ValueError("cannot expand an empty pedestrian configuration")
    if target_count <= len(base_people):
        selected = base_people[:target_count]
        selected_ids = {person["id"] for person in selected}
        result_stage1["pedestrians"] = copy.deepcopy(selected)
        result_routes["pedestrians"] = {
            key: copy.deepcopy(value)
            for key, value in routes["pedestrians"].items() if key in selected_ids
        }
        return result_stage1, result_routes

    people = copy.deepcopy(base_people)
    cached_people = copy.deepcopy(routes["pedestrians"])
    used_spawns = [list(cached_people[person["id"]]["spawn_xyz_m"]) for person in base_people]
    extras = target_count - len(base_people)
    scenario = str(routes["scenario"])
    for extra_index in range(extras):
        source = base_people[extra_index % len(base_people)]
        source_cache = routes["pedestrians"][source["id"]]
        path = source_cache["cycle_paths_xyz_m"]["0"]
        if len(path) < 3:
            raise ValueError(f"route for {source['id']} is too short to add a pedestrian")
        fraction = 0.2 + 0.6 * (extra_index + 0.5) / extras
        target_index = round(fraction * (len(path) - 1))
        candidate_indices = sorted(
            range(1, len(path) - 1), key=lambda index: abs(index - target_index)
        )
        spawn_index = next(
            (
                index for index in candidate_indices
                if min(_distance_xy(path[index], used) for used in used_spawns)
                >= minimum_spawn_separation_m
            ),
            None,
        )
        if spawn_index is None:
            raise ValueError(
                f"no separated spawn found on validated route for {source['id']}"
            )
        new_id = f"{scenario}_{len(base_people) + extra_index:02d}"
        spawn = list(path[spawn_index])
        new_person = copy.deepcopy(source)
        new_person["id"] = new_id
        new_person["spawn"] = spawn
        new_cache = copy.deepcopy(source_cache)
        new_cache["spawn_xyz_m"] = spawn
        new_cache["route_offset"] = 0
        new_cache["initial_path_xyz_m"] = copy.deepcopy(path[spawn_index:])
        people.append(new_person)
        cached_people[new_id] = new_cache
        used_spawns.append(spawn)
    result_stage1["pedestrians"] = people
    result_routes["pedestrians"] = cached_people
    return result_stage1, result_routes
