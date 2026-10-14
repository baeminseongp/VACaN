"""Evaluate robot-centric person observations from JRDB 2D labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _occlusion(label: dict[str, Any]) -> Any:
    attributes = label.get("attributes") or {}
    return attributes.get("occlusion", label.get("occlusion", "unknown"))


def _severely_occluded(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return int(value) >= 2
    normalized = str(value).lower().replace("-", "_").replace(" ", "_")
    return normalized in {"2", "3", "severely_occluded", "fully_occluded"}


def _horizontal_occupancy(labels: list[dict[str, Any]], width: float) -> float:
    intervals: list[tuple[float, float]] = []
    for label in labels:
        box = label.get("box") or {}
        if {"cx", "w"} <= box.keys():
            left = float(box["cx"]) - float(box["w"]) / 2
            right = float(box["cx"]) + float(box["w"]) / 2
        elif {"x", "w"} <= box.keys():
            left, right = float(box["x"]), float(box["x"]) + float(box["w"])
        else:
            continue
        intervals.append((max(0.0, left), min(width, right)))
    covered = 0.0
    end = 0.0
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if left > end:
            covered += right - left
        elif right > end:
            covered += right - end
        end = max(end, right)
    return covered / width


def _frame_number(name: str, fallback: int) -> int:
    digits = "".join(character for character in Path(name).stem if character.isdigit())
    return int(digits) if digits else fallback


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_jrdb(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = _resolve(config_path, config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    image_width = float(config.get("panorama_width_px", 3760))
    frame_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []

    for sequence in config["sequences"]:
        annotation_path = _resolve(config_path, sequence["annotations"])
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        labels_by_frame = payload.get("labels")
        if not isinstance(labels_by_frame, dict):
            raise ValueError(f"JRDB labels must be a frame-to-label mapping: {annotation_path}")
        tracks: dict[str, list[int]] = defaultdict(list)
        seen: set[str] = set()
        ordered = sorted(labels_by_frame.items(), key=lambda item: _frame_number(item[0], 0))
        for fallback, (frame_name, labels) in enumerate(ordered):
            if not isinstance(labels, list):
                raise ValueError(f"frame labels must be a list: {frame_name}")
            frame_id = _frame_number(frame_name, fallback)
            track_ids = [str(label.get("label_id", f"anonymous:{index}")) for index, label in enumerate(labels)]
            new_tracks = sum(track_id not in seen for track_id in track_ids)
            for track_id in track_ids:
                seen.add(track_id)
                tracks[track_id].append(frame_id)
            severe = sum(_severely_occluded(_occlusion(label)) for label in labels)
            frame_rows.append({
                "dataset_id": config["dataset_id"],
                "sequence_id": sequence["sequence_id"],
                "sensor_motion": sequence.get("sensor_motion", "UNKNOWN"),
                "frame_id": frame_id,
                "person_count": len(labels),
                "visible_person_count": len(labels) - severe,
                "severely_occluded_count": severe,
                "new_track_count": new_tracks,
                "active_track_count": len(track_ids),
                "panorama_occupancy_fraction": f"{_horizontal_occupancy(labels, image_width):.8f}",
            })
        for track_id, frames in tracks.items():
            span = max(frames) - min(frames) + 1
            continuity_rows.append({
                "dataset_id": config["dataset_id"],
                "sequence_id": sequence["sequence_id"],
                "sensor_motion": sequence.get("sensor_motion", "UNKNOWN"),
                "track_id": track_id,
                "observed_frame_count": len(set(frames)),
                "frame_span": span,
                "continuity": f"{len(set(frames)) / span:.8f}",
            })
        inputs.append({
            "sequence_id": sequence["sequence_id"],
            "annotations": str(annotation_path),
            "sha256": _sha256(annotation_path),
            "frame_count": len(ordered),
            "sensor_motion": sequence.get("sensor_motion", "UNKNOWN"),
        })

    summary_rows: list[dict[str, Any]] = []
    motions = sorted({row["sensor_motion"] for row in frame_rows})
    for motion in ["ALL", *motions]:
        rows = frame_rows if motion == "ALL" else [row for row in frame_rows if row["sensor_motion"] == motion]
        tracks = continuity_rows if motion == "ALL" else [row for row in continuity_rows if row["sensor_motion"] == motion]
        counts = [float(row["person_count"]) for row in rows]
        visible = [
            float(row["visible_person_count"]) / float(row["person_count"])
            for row in rows
            if float(row["person_count"]) > 0
        ]
        continuity = [float(row["continuity"]) for row in tracks]
        occupancy = [float(row["panorama_occupancy_fraction"]) for row in rows]
        metrics = {
            "frame_count": len(rows),
            "mean_person_count": _mean(counts),
            "p95_person_count": _percentile(counts, 0.95),
            "max_person_count": max(counts),
            "mean_visible_fraction": _mean(visible),
            "mean_track_continuity": _mean(continuity),
            "mean_panorama_occupancy_fraction": _mean(occupancy),
            "new_tracks_per_frame": sum(float(row["new_track_count"]) for row in rows) / len(rows),
        }
        summary_rows.extend({
            "dataset_id": config["dataset_id"],
            "sensor_motion": motion,
            "metric": name,
            "value": f"{value:.8f}" if isinstance(value, float) else value,
        } for name, value in metrics.items())

    _write_csv(output / "frame_metrics.csv", frame_rows)
    _write_csv(output / "track_continuity.csv", continuity_rows)
    _write_csv(output / "summary_metrics.csv", summary_rows)
    report = {
        "dataset_id": config["dataset_id"],
        "evaluation_family": "mobile_robot_sensing",
        "sequence_count": len(config["sequences"]),
        "frame_count": len(frame_rows),
        "track_count": len(continuity_rows),
        "inputs": inputs,
        "outputs": ["frame_metrics.csv", "track_continuity.csv", "summary_metrics.csv"],
        "license": config["license"],
        "limitations": config["limitations"],
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
