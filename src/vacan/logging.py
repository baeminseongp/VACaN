"""Structured JSON logging setup."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("run_id", "episode", "step", "seed", "component"):
            if hasattr(record, key):
                event[key] = getattr(record, key)
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, sort_keys=True)


def configure_logging(level: str = "INFO", output: str | Path | None = None) -> None:
    handler: logging.Handler
    if output is None:
        handler = logging.StreamHandler()
    else:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
