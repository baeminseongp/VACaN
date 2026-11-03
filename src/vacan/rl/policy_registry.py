"""Environment-scoped Stage 5 policy metadata and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


POLICY_METADATA_FILENAME = "policy-metadata.json"
POLICY_SCOPE = "single_environment"
SCENARIOS = {"office", "hospital", "warehouse"}


def _candidate_metadata_paths(checkpoint: Path) -> list[Path]:
    checkpoint = checkpoint.resolve()
    return [
        checkpoint.parent / POLICY_METADATA_FILENAME,
        checkpoint.parent / "summary.json",
        checkpoint.parent.parent / POLICY_METADATA_FILENAME,
        checkpoint.parent.parent / "summary.json",
    ]


def checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    """Read policy metadata, including summaries from pre-registry runs."""
    for path in _candidate_metadata_paths(checkpoint):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        scenario = data.get("scenario")
        if scenario in SCENARIOS:
            parent_value = data.get("parent_checkpoint", data.get("resumed_from"))
            if parent_value:
                parent = Path(parent_value)
                if not parent.is_file():
                    raise ValueError(
                        f"cannot verify policy lineage because parent checkpoint is missing: {parent}"
                    )
                parent_metadata = checkpoint_metadata(parent)
                if parent_metadata["scenario"] != scenario:
                    raise ValueError(
                        f"checkpoint lineage crosses environments: {parent_metadata['scenario']} "
                        f"parent -> {scenario} child ({checkpoint})"
                    )
            return {
                "scenario": scenario,
                "policy_scope": data.get("policy_scope", POLICY_SCOPE),
                "source": str(path),
                "parent_checkpoint": str(parent_value) if parent_value else None,
            }
    raise ValueError(
        f"cannot establish environment ownership for checkpoint {checkpoint}; "
        f"expected {POLICY_METADATA_FILENAME} or summary.json beside its run"
    )


def validate_checkpoint_scenario(checkpoint: Path, scenario: str) -> dict[str, Any]:
    """Reject accidental cross-environment policy loading."""
    if scenario not in SCENARIOS:
        raise ValueError(f"invalid policy scenario: {scenario}")
    metadata = checkpoint_metadata(checkpoint)
    if metadata["scenario"] != scenario:
        raise ValueError(
            f"checkpoint belongs to {metadata['scenario']}, not {scenario}: {checkpoint}"
        )
    if metadata["policy_scope"] != POLICY_SCOPE:
        raise ValueError(
            f"checkpoint is not an environment-specific policy: {checkpoint}"
        )
    return metadata


def write_policy_metadata(path: Path, *, scenario: str, phase: str, seed: int,
                          training_steps: int, checkpoint: Path,
                          parent_checkpoint: Path | None) -> dict[str, Any]:
    """Write the ownership record used for later resume/evaluation validation."""
    if scenario not in SCENARIOS:
        raise ValueError(f"invalid policy scenario: {scenario}")
    data = {
        "schema_version": 1,
        "policy_scope": POLICY_SCOPE,
        "scenario": scenario,
        "phase": phase,
        "seed": int(seed),
        "training_steps": int(training_steps),
        "checkpoint": str(checkpoint.resolve()),
        "parent_checkpoint": str(parent_checkpoint.resolve()) if parent_checkpoint else None,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data
