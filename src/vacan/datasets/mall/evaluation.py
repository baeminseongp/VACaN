"""Evaluate structured VLM crowd estimates against Mall head-point GT."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


LEVELS = ("LOW", "MEDIUM", "HIGH")


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_point_counts(path: Path) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for row in _read_csv(path):
        counts[int(row["frame_id"])] += 1
    return dict(counts)


def load_mat_counts(path: Path) -> dict[int, int]:
    try:
        import numpy as np
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError("Mall .mat input requires the optional 'mall' dependencies") from error
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    frames = payload.get("frame")
    if frames is None:
        raise ValueError("mall_gt.mat does not contain the expected 'frame' variable")
    counts: dict[int, int] = {}
    for index, frame in enumerate(np.atleast_1d(frames), start=1):
        locations = getattr(frame, "loc", None)
        if locations is None:
            raise ValueError(f"Mall frame {index} does not contain a loc field")
        array = np.asarray(locations)
        counts[index] = 1 if array.ndim == 1 and array.size else int(array.shape[0])
    return counts


def _level(count: int, medium_min: int, high_min: int) -> str:
    if count >= high_min:
        return "HIGH"
    if count >= medium_min:
        return "MEDIUM"
    return "LOW"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _macro_f1(actual: list[str], predicted: list[str]) -> float:
    scores = []
    for level in LEVELS:
        tp = sum(a == level and p == level for a, p in zip(actual, predicted))
        fp = sum(a != level and p == level for a, p in zip(actual, predicted))
        fn = sum(a == level and p != level for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evaluate_mall(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ground_truth_path = _resolve(config_path, config["ground_truth"])
    prediction_path = _resolve(config_path, config["vlm_predictions"])
    output = _resolve(config_path, config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    counts = load_mat_counts(ground_truth_path) if ground_truth_path.suffix.lower() == ".mat" else load_point_counts(ground_truth_path)
    predictions = {int(row["frame_id"]): row for row in _read_csv(prediction_path)}
    if set(counts) != set(predictions):
        missing = sorted(set(counts) - set(predictions))
        extra = sorted(set(predictions) - set(counts))
        raise ValueError(f"VLM/GT frame mismatch; missing={missing[:5]}, extra={extra[:5]}")
    medium_min = int(config["congestion_thresholds"]["medium_min_count"])
    high_min = int(config["congestion_thresholds"]["high_min_count"])
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    actual_levels: list[str] = []
    predicted_levels: list[str] = []
    for frame_id in sorted(counts):
        prediction = predictions[frame_id]
        actual_count = counts[frame_id]
        predicted_count = int(prediction["predicted_count"])
        actual_level = _level(actual_count, medium_min, high_min)
        predicted_level = prediction["predicted_level"].upper()
        if predicted_level not in LEVELS:
            raise ValueError(f"invalid predicted_level at frame {frame_id}: {predicted_level}")
        error = predicted_count - actual_count
        evidence_complete = bool(prediction.get("description", "").strip() and prediction.get("grounded_evidence", "").strip())
        rows.append({
            "dataset_id": config["dataset_id"],
            "frame_id": frame_id,
            "ground_truth_count": actual_count,
            "predicted_count": predicted_count,
            "signed_error": error,
            "absolute_error": abs(error),
            "ground_truth_level": actual_level,
            "predicted_level": predicted_level,
            "level_correct": int(actual_level == predicted_level),
            "evidence_complete": int(evidence_complete),
        })
        errors.append(float(error))
        actual_levels.append(actual_level)
        predicted_levels.append(predicted_level)
    metrics = {
        "frame_count": len(rows),
        "count_mae": sum(abs(value) for value in errors) / len(errors),
        "count_rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "count_bias": sum(errors) / len(errors),
        "within_5_count_fraction": sum(abs(value) <= 5 for value in errors) / len(errors),
        "congestion_level_accuracy": sum(a == p for a, p in zip(actual_levels, predicted_levels)) / len(rows),
        "congestion_level_macro_f1": _macro_f1(actual_levels, predicted_levels),
        "evidence_complete_fraction": sum(row["evidence_complete"] for row in rows) / len(rows),
    }
    summary_rows = [{
        "dataset_id": config["dataset_id"],
        "metric": metric,
        "value": f"{value:.8f}" if isinstance(value, float) else value,
    } for metric, value in metrics.items()]
    _write_csv(output / "frame_metrics.csv", rows)
    _write_csv(output / "summary_metrics.csv", summary_rows)
    report = {
        "dataset_id": config["dataset_id"],
        "evaluation_family": "dense_crowd_vlm",
        "frame_count": len(rows),
        "metrics": metrics,
        "thresholds": config["congestion_thresholds"],
        "inputs": {
            "ground_truth": {"path": str(ground_truth_path), "sha256": _sha256(ground_truth_path)},
            "vlm_predictions": {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
        },
        "outputs": ["frame_metrics.csv", "summary_metrics.csv"],
        "license": config["license"],
        "limitations": config["limitations"],
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
