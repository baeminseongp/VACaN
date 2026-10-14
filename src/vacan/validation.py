"""Dependency-free validation for canonical Step 1 scenario fixtures."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import CongestionLevel, ObservationState


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def validate_scenario(directory: Path, contract_path: Path | None = None) -> ValidationResult:
    directory = directory.resolve()
    if contract_path is None:
        contract_path = Path(__file__).resolve().parents[2] / "schemas" / "contracts.json"
    contract = _load_json(contract_path)
    errors: list[str] = []
    optional_tables = set(contract.get("optional_tables", []))
    observation_states = {item.value for item in ObservationState}
    congestion_levels = {item.value for item in CongestionLevel}

    zones_path = directory / "zones.geojson"
    if not zones_path.exists():
        errors.append("missing zones.geojson")
        zone_ids: set[str] = set()
    else:
        zones = _load_json(zones_path)
        if zones.get("type") != "FeatureCollection":
            errors.append("zones.geojson must be a FeatureCollection")
        features = zones.get("features", [])
        zone_ids = {str(item.get("properties", {}).get("zone_id", "")) for item in features}
        if "" in zone_ids or len(zone_ids) != len(features):
            errors.append("zone_id values must be present and unique")
        for item in features:
            properties = item.get("properties", {})
            if float(properties.get("area_m2", 0)) <= 0:
                errors.append(f"zone {properties.get('zone_id')} must have positive area_m2")

    loaded: dict[str, list[dict[str, str]]] = {}
    for filename, required_header in contract["tables"].items():
        path = directory / filename
        if not path.exists():
            if filename not in optional_tables:
                errors.append(f"missing {filename}")
            continue
        header, rows = _load_rows(path)
        if header != required_header:
            errors.append(f"{filename} header does not match contract {contract['contract_version']}")
        loaded[filename] = rows

    for index, row in enumerate(loaded.get("congestion.csv", []), start=2):
        zone_id = row.get("zone_id", "")
        if zone_id not in zone_ids:
            errors.append(f"congestion.csv:{index} references unknown zone {zone_id}")
        state = row.get("observation_state", "")
        level = row.get("congestion_level", "")
        if state not in observation_states:
            errors.append(f"congestion.csv:{index} has invalid observation_state {state}")
        if level not in congestion_levels:
            errors.append(f"congestion.csv:{index} has invalid congestion_level {level}")
        density = row.get("density_per_m2", "")
        count = row.get("pedestrian_count", "")
        if state == ObservationState.UNKNOWN:
            if density or count or level != CongestionLevel.UNKNOWN:
                errors.append(f"congestion.csv:{index} UNKNOWN must not contain count/density")
        elif not density or not count:
            errors.append(f"congestion.csv:{index} observed state requires count and density")
        confidence = row.get("confidence", "")
        if confidence and not 0 <= float(confidence) <= 1:
            errors.append(f"congestion.csv:{index} confidence must be in [0, 1]")

    for filename in ("observations.csv", "robots.csv"):
        for index, row in enumerate(loaded.get(filename, []), start=2):
            visible = row.get("visible_zone_ids", "")
            if visible:
                for zone_id in visible.split("|"):
                    if zone_id not in zone_ids:
                        errors.append(f"{filename}:{index} references unknown zone {zone_id}")

    for index, row in enumerate(loaded.get("replanning_events.csv", []), start=2):
        zone_id = row.get("trigger_zone_id", "")
        if zone_id and zone_id not in zone_ids:
            errors.append(f"replanning_events.csv:{index} references unknown zone {zone_id}")

    return ValidationResult(tuple(errors))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m vacan.validation SCENARIO_DIRECTORY", file=sys.stderr)
        return 2
    result = validate_scenario(Path(args[0]))
    if not result.ok:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: {Path(args[0])} matches the canonical data contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
