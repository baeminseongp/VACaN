"""Event-aware Stage 7 interventions for congestion and pedestrian intention."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

from vacan.mapping.dynamic_costmap import Cell, GridGeometry


@dataclass(frozen=True)
class InterventionSpec:
    intervention_id: str
    factor: str
    intervention_type: str
    start_time_s: float
    end_time_s: float
    cells: tuple[Cell, ...] = ()
    pedestrian_id: str | None = None
    value: float | None = None
    delta: float | None = None
    destination_grid: Cell | None = None
    measurement_convention: str | None = None
    ground_truth_reference: str | None = None
    event_id: str | None = None

    def active(self, now: float, tolerance: float = 1e-6) -> bool:
        return self.start_time_s - tolerance <= now <= self.end_time_s + tolerance

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cells"] = [list(cell) for cell in self.cells]
        if self.destination_grid is not None:
            result["destination_grid"] = list(self.destination_grid)
        return result


def _point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        x1, y1 = map(float, first)
        x2, y2 = map(float, second)
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def resolve_scope(scope: dict[str, Any], geometry: GridGeometry) -> tuple[Cell, ...]:
    cells = {tuple(map(int, cell)) for cell in scope.get("cells", [])}
    rows, columns = geometry.shape
    rectangle = scope.get("rectangle_xy")
    polygon = scope.get("polygon_xy")
    if rectangle is not None:
        x0, y0, x1, y1 = map(float, rectangle)
        for row in range(rows):
            for column in range(columns):
                x, y = geometry.cell_center((row, column))
                if x0 <= x < x1 and y0 <= y < y1:
                    cells.add((row, column))
    if polygon is not None:
        if len(polygon) < 3:
            raise ValueError("polygon intervention scope needs at least three vertices")
        for row in range(rows):
            for column in range(columns):
                if _point_in_polygon(geometry.cell_center((row, column)), polygon):
                    cells.add((row, column))
    for row, column in cells:
        if not (0 <= row < rows and 0 <= column < columns):
            raise ValueError(f"intervention cell {(row, column)} is outside grid {geometry.shape}")
    return tuple(sorted(cells))


def parse_intervention(raw: dict[str, Any], geometry: GridGeometry) -> InterventionSpec:
    factor = str(raw.get("factor"))
    kind = str(raw.get("type"))
    if factor not in {"congestion", "intention"}:
        raise ValueError("Stage 7 factor must be congestion or intention")
    supported = {
        "congestion": {"overestimate", "underestimate", "ground_truth_correction",
                       "full_information", "no_op"},
        "intention": {"incorrect_destination", "missing", "spurious",
                      "ground_truth_correction", "full_information", "no_op"},
    }
    if kind not in supported[factor]:
        raise ValueError(f"unsupported {factor} intervention type {kind}")
    start = float(raw["start_time_s"])
    end = float(raw.get("end_time_s", start))
    if start < 0 or end < start:
        raise ValueError("invalid intervention time interval")
    destination = raw.get("destination_grid")
    return InterventionSpec(
        intervention_id=str(raw["id"]), factor=factor, intervention_type=kind,
        start_time_s=start, end_time_s=end,
        cells=resolve_scope(raw.get("scope", {}), geometry),
        pedestrian_id=None if raw.get("pedestrian_id") is None else str(raw["pedestrian_id"]),
        value=None if raw.get("value") is None else float(raw["value"]),
        delta=None if raw.get("delta") is None else float(raw["delta"]),
        destination_grid=None if destination is None else tuple(map(int, destination)),
        measurement_convention=raw.get("measurement_convention"),
        ground_truth_reference=raw.get("ground_truth_reference"),
        event_id=raw.get("event_id"),
    )


class InterventionManager:
    """Modify Stage 3 source events while retaining an immutable audit log."""

    def __init__(self, specs: list[InterventionSpec], geometry: GridGeometry,
                 cctv_cells: set[Cell]) -> None:
        identifiers = [item.intervention_id for item in specs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("intervention identifiers must be unique")
        for spec in specs:
            if set(spec.cells) & cctv_cells and spec.intervention_type != "no_op":
                raise ValueError(f"{spec.intervention_id}: interventions cannot overwrite CCTV cells")
            if spec.factor == "congestion" and not spec.cells and spec.intervention_type not in {
                    "no_op", "full_information"}:
                raise ValueError(f"{spec.intervention_id}: congestion intervention needs cells")
            if spec.intervention_type in {"incorrect_destination", "ground_truth_correction"} \
                    and spec.factor == "intention" and spec.destination_grid is None:
                raise ValueError(f"{spec.intervention_id}: destination_grid is required")
            if spec.intervention_type == "spurious" and (not spec.pedestrian_id or not spec.event_id):
                raise ValueError("spurious intention requires identified pedestrian_id and event_id")
            if spec.factor == "intention" and spec.intervention_type in {
                    "incorrect_destination", "ground_truth_correction", "spurious"}:
                if spec.destination_grid is None:
                    raise ValueError(f"{spec.intervention_id}: intention destination is required")
                row, column = spec.destination_grid
                rows, columns = geometry.shape
                if not (0 <= row < rows and 0 <= column < columns):
                    raise ValueError(f"{spec.intervention_id}: intention destination is outside grid")
                if spec.destination_grid in cctv_cells:
                    raise ValueError(f"{spec.intervention_id}: intention cannot overwrite CCTV")
        self.specs = list(specs)
        self.geometry = geometry
        self.cctv_cells = set(cctv_cells)
        self.logs: list[dict[str, Any]] = []
        self._applied_once: set[str] = set()
        self._full_intention_destinations: dict[str, Cell] = {}

    def modify_congestion(self, now: float, observations: list[dict[str, Any]],
                          visibility_mask: list[list[bool]],
                          ground_truth_counts: list[list[int]]) -> tuple[dict[Cell, float], list[dict[str, Any]]]:
        counts: dict[Cell, int] = {}
        for observation in observations:
            cell = tuple(observation["observed_grid"])
            counts[cell] = counts.get(cell, 0) + 1
        overrides: dict[Cell, float] = {}
        tick_logs = []
        for spec in self.specs:
            if spec.factor != "congestion" or not spec.active(now) or spec.intervention_type == "no_op":
                continue
            if spec.intervention_type == "full_information":
                affected = []
                for row in range(self.geometry.shape[0]):
                    for column in range(self.geometry.shape[1]):
                        cell = (row, column)
                        if cell not in self.cctv_cells:
                            overrides[cell] = float(ground_truth_counts[row][column])
                            affected.append([row, column])
                tick_logs.append({
                    "intervention_id": spec.intervention_id, "factor": "congestion",
                    "type": "full_information", "simulation_time_s": float(now),
                    "observation_source": "privileged_full_grid_count",
                    "affected_grid_cells": affected, "anchor_reset": True,
                })
                continue
            for cell in spec.cells:
                row, column = cell
                if not visibility_mask[row][column] and cell not in overrides:
                    continue
                original = float(overrides.get(cell, counts.get(cell, 0)))
                if spec.intervention_type == "ground_truth_correction":
                    if spec.measurement_convention != "full_grid_count":
                        raise ValueError("only explicitly labeled full_grid_count correction is supported")
                    modified = float(ground_truth_counts[row][column])
                elif spec.value is not None:
                    modified = spec.value
                else:
                    magnitude = abs(float(spec.delta if spec.delta is not None else 1.0))
                    modified = original + magnitude if spec.intervention_type == "overestimate" \
                        else original - magnitude
                modified = max(0.0, float(modified))
                overrides[cell] = modified
                event = {
                    "intervention_id": spec.intervention_id, "factor": "congestion",
                    "type": spec.intervention_type, "simulation_time_s": float(now),
                    "grid_cell": list(cell), "observation_source": "robot_fov",
                    "original_observed_count": original, "modified_count": modified,
                    "ground_truth_count": int(ground_truth_counts[row][column]),
                    "measurement_convention": spec.measurement_convention or "noisy_fov_assignment",
                    "resulting_costmap_delta": modified - original, "anchor_reset": True,
                }
                tick_logs.append(event)
        self.logs.extend(copy.deepcopy(tick_logs))
        return overrides, tick_logs

    def modify_intentions(self, now: float, events: list[dict[str, Any]],
                          observations: list[dict[str, Any]],
                          pedestrians: list[dict[str, Any]] | None = None) \
            -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        modified_events = [copy.deepcopy(event) for event in events]
        tick_logs = []
        observations_by_id = {str(item["id"]): item for item in observations}
        for spec in self.specs:
            if spec.factor != "intention" or not spec.active(now) or spec.intervention_type == "no_op":
                continue
            if spec.intervention_type == "full_information":
                modified_events = []
                for person in pedestrians or []:
                    destination_xyz = person.get("current_destination_xyz_m")
                    if destination_xyz is None:
                        raise ValueError("full-information intention requires controller destination")
                    destination = self.geometry.world_to_grid(
                        float(destination_xyz[0]), float(destination_xyz[1])
                    )
                    identifier = str(person["id"])
                    if destination is None or destination in self.cctv_cells:
                        continue
                    if self._full_intention_destinations.get(identifier) == destination:
                        continue
                    self._full_intention_destinations[identifier] = destination
                    modified_events.append({
                        "event_id": f"gt:{identifier}:{destination[0]},{destination[1]}:{now:.6f}",
                        "pedestrian_id": identifier, "destination_grid": list(destination),
                        "contribution": 1, "source": "pegasus_controller_destination",
                    })
                    tick_logs.append({
                        "intervention_id": spec.intervention_id, "factor": "intention",
                        "type": "full_information", "simulation_time_s": float(now),
                        "event_id": modified_events[-1]["event_id"],
                        "pedestrian_id": identifier,
                        "modified_destination_grid": list(destination),
                        "ground_truth_reference": "pegasus_controller_current_destination",
                        "modified_contribution": 1,
                        "affected_grid_cells": [list(destination)], "anchor_reset": True,
                    })
                continue
            matched = [event for event in modified_events
                       if spec.pedestrian_id is None or event["pedestrian_id"] == spec.pedestrian_id]
            if spec.intervention_type == "spurious":
                once_key = f"{spec.intervention_id}:{spec.event_id}"
                if once_key in self._applied_once:
                    continue
                self._applied_once.add(once_key)
                event = {
                    "event_id": str(spec.event_id), "pedestrian_id": str(spec.pedestrian_id),
                    "destination_grid": list(spec.destination_grid or spec.cells[0]),
                    "contribution": 1, "source": "counterfactual_spurious",
                }
                modified_events.append(event)
                matched = [event]
            for event in list(matched):
                original_destination = list(event["destination_grid"])
                original_contribution = int(event.get("contribution", 1))
                if spec.intervention_type == "missing":
                    modified_events.remove(event)
                    new_destination, new_contribution = None, 0
                elif spec.intervention_type in {"incorrect_destination", "ground_truth_correction"}:
                    event["destination_grid"] = list(spec.destination_grid)
                    event["source"] = f"counterfactual_{spec.intervention_type}"
                    new_destination, new_contribution = list(spec.destination_grid), original_contribution
                else:
                    new_destination, new_contribution = list(event["destination_grid"]), 1
                observation = observations_by_id.get(event["pedestrian_id"], {})
                delta_by_cell = {f"{original_destination[0]},{original_destination[1]}":
                                 -original_contribution}
                if new_destination is not None:
                    key = f"{new_destination[0]},{new_destination[1]}"
                    delta_by_cell[key] = delta_by_cell.get(key, 0) + new_contribution
                log = {
                    "intervention_id": spec.intervention_id, "factor": "intention",
                    "type": spec.intervention_type, "simulation_time_s": float(now),
                    "event_id": event.get("event_id"), "pedestrian_id": event["pedestrian_id"],
                    "original_observed_position_xy_m": observation.get("position_xy_m"),
                    "original_observed_velocity_xy_m_s": observation.get("velocity_xy_m_s"),
                    "original_destination_grid": original_destination,
                    "modified_destination_grid": new_destination,
                    "ground_truth_reference": spec.ground_truth_reference,
                    "original_contribution": original_contribution,
                    "modified_contribution": new_contribution,
                    "affected_grid_cells": [cell for cell in (original_destination, new_destination)
                                            if cell is not None],
                    "resulting_costmap_delta_by_cell": delta_by_cell,
                    "anchor_reset": new_contribution > 0,
                }
                tick_logs.append(log)
        self.logs.extend(copy.deepcopy(tick_logs))
        return modified_events, tick_logs


def decision_costmap_intervention(costmap: list[list[float]], diagnostics: dict[str, Any],
                                  ground_truth: list[list[int]], spec: InterventionSpec, *,
                                  ground_truth_pedestrians: list[dict[str, Any]] | None = None,
                                  cctv_cells: set[Cell] | None = None,
                                  geometry: GridGeometry | None = None) \
        -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Apply a same-tick decision intervention without mutating recorded input."""
    result = copy.deepcopy(costmap)
    logs: list[dict[str, Any]] = []
    if spec.intervention_type == "no_op":
        return result, logs
    if spec.factor == "congestion":
        if spec.intervention_type == "full_information":
            for row in range(len(result)):
                for column in range(len(result[row])):
                    if (row, column) not in (cctv_cells or set()):
                        result[row][column] = float(ground_truth[row][column])
            return result, [{"factor": "congestion", "type": "full_information"}]
        observations = diagnostics.get("observations", [])
        counts: dict[Cell, int] = {}
        for observation in observations:
            cell = tuple(observation["observed_grid"])
            counts[cell] = counts.get(cell, 0) + 1
        for row, column in spec.cells:
            original = float(counts.get((row, column), 0))
            if spec.intervention_type == "ground_truth_correction":
                if spec.measurement_convention != "full_grid_count":
                    raise ValueError("decision correction must be labeled full_grid_count")
                modified = float(ground_truth[row][column])
            elif spec.value is not None:
                modified = spec.value
            else:
                amount = abs(float(spec.delta if spec.delta is not None else 1.0))
                modified = original + amount if spec.intervention_type == "overestimate" else original - amount
            # The recorded final cell may already include same-tick intention
            # additions, which occur after direct replacement. Preserve them.
            result[row][column] = max(
                0.0, float(result[row][column]) + float(modified) - original
            )
            logs.append({"grid_cell": [row, column], "original": original,
                         "modified": result[row][column], "factor": "congestion"})
        return result, logs
    if spec.intervention_type == "full_information":
        if geometry is None:
            raise ValueError("decision full-information intention needs grid geometry")
        for person in ground_truth_pedestrians or []:
            destination_xyz = person.get("current_destination_xyz_m")
            if destination_xyz is None:
                continue
            destination = geometry.world_to_grid(float(destination_xyz[0]), float(destination_xyz[1]))
            if destination is not None and destination not in (cctv_cells or set()):
                result[destination[0]][destination[1]] += 1.0
                logs.append({"factor": "intention", "type": "full_information",
                             "pedestrian_id": person["id"],
                             "modified_destination": list(destination)})
        return result, logs
    events = diagnostics.get("intention_events", [])
    if spec.pedestrian_id is not None and not any(
            event["pedestrian_id"] == spec.pedestrian_id for event in events):
        if geometry is not None:
            for person in ground_truth_pedestrians or []:
                if str(person["id"]) != spec.pedestrian_id:
                    continue
                destination_xyz = person.get("current_destination_xyz_m")
                destination = None if destination_xyz is None else geometry.world_to_grid(
                    float(destination_xyz[0]), float(destination_xyz[1]))
                if destination is not None:
                    events = [{"pedestrian_id": spec.pedestrian_id,
                               "destination_grid": list(destination)}]
                break
    for event in events:
        if spec.pedestrian_id is not None and event["pedestrian_id"] != spec.pedestrian_id:
            continue
        old = tuple(event["destination_grid"])
        if spec.intervention_type == "missing":
            result[old[0]][old[1]] -= 1.0
            new = None
        else:
            new = spec.destination_grid
            result[old[0]][old[1]] -= 1.0
            result[new[0]][new[1]] += 1.0
        logs.append({"pedestrian_id": event["pedestrian_id"], "original_destination": list(old),
                     "modified_destination": None if new is None else list(new), "factor": "intention"})
    return result, logs
