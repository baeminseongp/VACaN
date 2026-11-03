"""Atomic checkpoint persistence with reproducibility metadata."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CheckpointMetadata:
    run_id: str
    step: int
    seed: int
    created_at: str


class CheckpointManager:
    def __init__(self, directory: str | Path, keep_last: int = 5) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        self.directory = Path(directory)
        self.keep_last = keep_last

    def save(
        self,
        run_id: str,
        step: int,
        seed: int,
        writer: Callable[[Path], Any],
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{run_id}-step-{step:09d}.ckpt"
        temporary = target.with_suffix(".tmp")
        writer(temporary)
        os.replace(temporary, target)
        metadata = CheckpointMetadata(run_id, step, seed, datetime.now(UTC).isoformat())
        metadata_path = target.with_suffix(".json")
        metadata_tmp = metadata_path.with_suffix(".json.tmp")
        metadata_tmp.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
        os.replace(metadata_tmp, metadata_path)
        self._prune(run_id)
        return target

    def _prune(self, run_id: str) -> None:
        checkpoints = sorted(self.directory.glob(f"{run_id}-step-*.ckpt"))
        for checkpoint in checkpoints[: -self.keep_last]:
            checkpoint.unlink(missing_ok=True)
            checkpoint.with_suffix(".json").unlink(missing_ok=True)
