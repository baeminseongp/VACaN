"""Streaming normalization of headerless ATC trajectory rows."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_HEADER = [
    "timestamp", "source_time_s", "person_id", "x_m", "y_m", "z_m",
    "speed_m_s", "motion_angle_rad", "facing_angle_rad", "source_dataset",
]


def normalize_trajectories(
    source_path: Path,
    output_path: Path,
    *,
    max_rows: int | None = None,
) -> int:
    """Normalize an ATC CSV while retaining the source clock for auditability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.reader(source)
        writer = csv.writer(destination)
        writer.writerow(OUTPUT_HEADER)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 8:
                raise ValueError(f"{source_path}:{line_number}: expected 8 columns, got {len(row)}")
            try:
                source_time = float(row[0])
                person_id = int(row[1])
                x_m, y_m, z_m = (float(value) / 1000 for value in row[2:5])
                speed_m_s = float(row[5]) / 1000
                motion_angle, facing_angle = float(row[6]), float(row[7])
            except ValueError as error:
                raise ValueError(f"{source_path}:{line_number}: invalid numeric value") from error
            timestamp = datetime.fromtimestamp(source_time, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            writer.writerow([
                timestamp, format(source_time, ".6f"), person_id,
                format(x_m, ".6f"), format(y_m, ".6f"), format(z_m, ".6f"),
                format(speed_m_s, ".6f"), format(motion_angle, ".6f"),
                format(facing_angle, ".6f"), "ATC",
            ])
            written += 1
            if max_rows is not None and written >= max_rows:
                break
    return written
