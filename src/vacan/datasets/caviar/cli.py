"""Project CAVIAR annotations to a shared ground plane and evaluate zone counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from .annotations import iter_annotations, xml_frame_count
from .homography import Homography, fit_homography, rmse


def inside(point: tuple[float, float], bounds: dict[str, float]) -> bool:
    x, y = point
    return bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]


def zone_id(point: tuple[float, float], bounds: dict[str, float], size: float) -> str:
    column = min(int((point[0] - bounds["x_min"]) // size), math.ceil((bounds["x_max"] - bounds["x_min"]) / size) - 1)
    row = min(int((point[1] - bounds["y_min"]) // size), math.ceil((bounds["y_max"] - bounds["y_min"]) / size) - 1)
    return f"CAVIAR-R{row:02d}-C{column:02d}"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def zone_features(bounds: dict[str, float], size: float) -> list[dict[str, Any]]:
    features = []
    rows = math.ceil((bounds["y_max"] - bounds["y_min"]) / size)
    columns = math.ceil((bounds["x_max"] - bounds["x_min"]) / size)
    for row in range(rows):
        for column in range(columns):
            x0 = bounds["x_min"] + column * size
            x1 = min(x0 + size, bounds["x_max"])
            y0 = bounds["y_min"] + row * size
            y1 = min(y0 + size, bounds["y_max"])
            identifier = f"CAVIAR-R{row:02d}-C{column:02d}"
            features.append({
                "type": "Feature",
                "properties": {"zone_id": identifier, "area_m2": (x1 - x0) * (y1 - y0)},
                "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
            })
    return features


def write_projection_figure(
    path: Path,
    frame: int,
    cameras: list[str],
    bounds: dict[str, float],
    size: float,
    projected_rows: list[dict[str, Any]],
) -> None:
    scale = 38
    plot_width = (bounds["x_max"] - bounds["x_min"]) * scale
    plot_height = (bounds["y_max"] - bounds["y_min"]) * scale
    panel_x = [210, 560]
    colors = ["#247ba0", "#f28e2b"]
    shapes: list[str] = []
    for index, camera in enumerate(cameras):
        x0 = panel_x[index]
        y0 = 70
        shapes.append(f'<text x="{x0 + plot_width / 2:.1f}" y="45" text-anchor="middle" class="camera">{camera}</text>')
        shapes.append(f'<rect x="{x0}" y="{y0}" width="{plot_width:.1f}" height="{plot_height:.1f}" fill="#f7fafb" stroke="#50636d"/>')
        columns = math.ceil((bounds["x_max"] - bounds["x_min"]) / size)
        rows = math.ceil((bounds["y_max"] - bounds["y_min"]) / size)
        for column in range(1, columns):
            x = x0 + column * size * scale
            shapes.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + plot_height:.1f}" stroke="#cbd6db"/>')
        for row in range(1, rows):
            y = y0 + row * size * scale
            shapes.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_width:.1f}" y2="{y:.1f}" stroke="#cbd6db"/>')
        points = [row for row in projected_rows if row["camera_id"] == camera and int(row["frame"]) == frame and int(row["within_common_evaluation"])]
        for row in points:
            x = x0 + (float(row["ground_x_m"]) - bounds["x_min"]) * scale
            y = y0 + plot_height - (float(row["ground_y_m"]) - bounds["y_min"]) * scale
            shapes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colors[index]}" fill-opacity=".78" stroke="white" stroke-width="2"/>')
        shapes.append(f'<text x="{x0 + plot_width / 2:.1f}" y="{y0 + plot_height + 25:.1f}" text-anchor="middle">{len(points)} projected people in shared support</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">
<style>text{{font:12px sans-serif;fill:#34444d}} .title{{font-size:18px;font-weight:700}} .camera{{font-size:14px;font-weight:700}}</style>
<rect width="100%" height="100%" fill="white"/><text x="25" y="28" class="title">CAVIAR frame {frame}: CCTV bottom-center → shared ground zones</text>
{''.join(shapes)}<text x="25" y="505">Grid: 1 m · common calibrated support: x 0–2.9 m, y 0–9.75 m · view-specific annotations</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = Path(config["retrieval_root"])
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    common_bounds = {key: float(value) for key, value in config["common_evaluation_bounds_m"].items()}
    size = float(config["zone_size_m"])
    zones = zone_features(common_bounds, size)
    (output / "zones.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": zones}, indent=2) + "\n", encoding="utf-8")
    zone_ids = [feature["properties"]["zone_id"] for feature in zones]

    projected_rows: list[dict[str, Any]] = []
    projection_metrics: list[dict[str, Any]] = []
    homographies: dict[str, Any] = {}
    counts: dict[str, dict[int, dict[str, int]]] = {}
    frame_counts: dict[str, int] = {}
    input_files: dict[str, dict[str, str | int]] = {}

    for view in config["views"]:
        camera = view["camera_id"]
        annotation_path = source_root / view["annotations"]
        video_path = source_root / view["video"]
        image_points = [tuple(map(float, item["image_px"])) for item in view["correspondences"]]
        ground_points = [(float(item["ground_cm"][0]) / 100, float(item["ground_cm"][1]) / 100) for item in view["correspondences"]]
        forward = fit_homography(image_points, ground_points)
        inverse = fit_homography(ground_points, image_points)
        fitted_ground = [forward.project(point) for point in image_points]
        fitted_image = [inverse.project(point) for point in ground_points]
        calibration_world_rmse = rmse(fitted_ground, ground_points)
        calibration_image_rmse = rmse(fitted_image, image_points)
        frame_count = xml_frame_count(annotation_path)
        frame_counts[camera] = frame_count
        counts[camera] = {frame: {zone: 0 for zone in zone_ids} for frame in range(frame_count)}
        annotations = 0
        within_calibration = 0
        within_common = 0
        roundtrip_squared = 0.0
        calibration_bounds = {key: float(value) for key, value in view["calibration_bounds_m"].items()}
        for annotation in iter_annotations(annotation_path):
            annotations += 1
            foot = annotation.bottom_center
            ground = forward.project(foot)
            reconstructed = inverse.project(ground)
            roundtrip_squared += math.dist(foot, reconstructed) ** 2
            in_calibration = inside(ground, calibration_bounds)
            in_common = inside(ground, common_bounds)
            within_calibration += int(in_calibration)
            within_common += int(in_common)
            identifier = zone_id(ground, common_bounds, size) if in_common else ""
            if identifier:
                counts[camera][annotation.frame][identifier] += 1
            projected_rows.append({
                "camera_id": camera, "frame": annotation.frame, "object_id": annotation.object_id,
                "bbox_xc_px": f"{annotation.xc:.3f}", "bbox_yc_px": f"{annotation.yc:.3f}",
                "bbox_width_px": f"{annotation.width:.3f}", "bbox_height_px": f"{annotation.height:.3f}",
                "foot_u_px": f"{foot[0]:.3f}", "foot_v_px": f"{foot[1]:.3f}",
                "ground_x_m": f"{ground[0]:.6f}", "ground_y_m": f"{ground[1]:.6f}",
                "within_calibration": int(in_calibration), "within_common_evaluation": int(in_common), "zone_id": identifier,
            })
        projection_metrics.extend([
            {"camera_id": camera, "metric": "calibration_control_world_rmse_m", "value": f"{calibration_world_rmse:.12f}", "sample_count": 4, "interpretation": "in_sample"},
            {"camera_id": camera, "metric": "calibration_control_image_rmse_px", "value": f"{calibration_image_rmse:.12f}", "sample_count": 4, "interpretation": "in_sample"},
            {"camera_id": camera, "metric": "annotation_roundtrip_rmse_px", "value": f"{math.sqrt(roundtrip_squared / annotations):.12f}", "sample_count": annotations, "interpretation": "numerical_consistency"},
            {"camera_id": camera, "metric": "within_calibration_fraction", "value": f"{within_calibration / annotations:.8f}", "sample_count": annotations, "interpretation": "support_check"},
            {"camera_id": camera, "metric": "within_common_evaluation_fraction", "value": f"{within_common / annotations:.8f}", "sample_count": annotations, "interpretation": "support_check"},
        ])
        homographies[camera] = {"image_to_ground": forward.values, "ground_to_image": inverse.values, "correspondences": view["correspondences"]}
        input_files[camera] = {
            "video": str(video_path), "video_sha256": digest(video_path),
            "annotations": str(annotation_path), "annotation_sha256": digest(annotation_path),
            "xml_frame_count": frame_count, "annotation_count": annotations,
        }

    cameras = [view["camera_id"] for view in config["views"]]
    if len(set(frame_counts.values())) != 1:
        raise ValueError(f"view frame counts do not match: {frame_counts}")
    total_frames = next(iter(frame_counts.values()))
    zone_rows: list[dict[str, Any]] = []
    for camera in cameras:
        for frame in range(total_frames):
            for identifier in zone_ids:
                zone_rows.append({"camera_id": camera, "frame": frame, "zone_id": identifier, "pedestrian_count": counts[camera][frame][identifier]})

    total_differences: list[float] = []
    zone_differences: list[float] = []
    exact_frames = 0
    exact_zone_frames = 0
    frame_rows: list[dict[str, Any]] = []
    for frame in range(total_frames):
        first_total = sum(counts[cameras[0]][frame].values())
        second_total = sum(counts[cameras[1]][frame].values())
        difference = first_total - second_total
        total_differences.append(difference)
        exact_frames += int(difference == 0)
        frame_rows.append({
            "frame": frame,
            f"{cameras[0]}_count": first_total,
            f"{cameras[1]}_count": second_total,
            "signed_difference": difference,
            "absolute_difference": abs(difference),
        })
        for identifier in zone_ids:
            zone_difference = counts[cameras[0]][frame][identifier] - counts[cameras[1]][frame][identifier]
            zone_differences.append(zone_difference)
            exact_zone_frames += int(zone_difference == 0)
    zone_metrics = [
        {"metric": "cross_view_total_count_mae_per_frame", "value": f"{sum(abs(value) for value in total_differences) / len(total_differences):.8f}", "sample_count": total_frames},
        {"metric": "cross_view_total_count_rmse_per_frame", "value": f"{math.sqrt(sum(value * value for value in total_differences) / len(total_differences)):.8f}", "sample_count": total_frames},
        {"metric": "cross_view_total_count_exact_fraction", "value": f"{exact_frames / total_frames:.8f}", "sample_count": total_frames},
        {"metric": "cross_view_zone_count_mae", "value": f"{sum(abs(value) for value in zone_differences) / len(zone_differences):.8f}", "sample_count": len(zone_differences)},
        {"metric": "cross_view_zone_count_rmse", "value": f"{math.sqrt(sum(value * value for value in zone_differences) / len(zone_differences)):.8f}", "sample_count": len(zone_differences)},
        {"metric": "cross_view_zone_count_exact_fraction", "value": f"{exact_zone_frames / len(zone_differences):.8f}", "sample_count": len(zone_differences)},
    ]
    write_csv(output / "projected_detections.csv", list(projected_rows[0]), projected_rows)
    write_csv(output / "zone_counts.csv", ["camera_id", "frame", "zone_id", "pedestrian_count"], zone_rows)
    write_csv(output / "frame_count_comparison.csv", list(frame_rows[0]), frame_rows)
    write_csv(output / "projection_metrics.csv", ["camera_id", "metric", "value", "sample_count", "interpretation"], projection_metrics)
    write_csv(output / "zone_count_metrics.csv", ["metric", "value", "sample_count"], zone_metrics)
    (output / "homographies.json").write_text(json.dumps(homographies, indent=2) + "\n", encoding="utf-8")
    representative_frame = max(
        range(total_frames),
        key=lambda frame: (
            min(sum(counts[camera][frame].values()) for camera in cameras),
            sum(sum(counts[camera][frame].values()) for camera in cameras),
        ),
    )
    write_projection_figure(
        output / "figure_ground_projection.svg", representative_frame, cameras,
        common_bounds, size, projected_rows,
    )
    paper_output = Path(config["paper_output_directory"])
    paper_output.mkdir(parents=True, exist_ok=True)
    for name in ("figure_ground_projection.svg", "projection_metrics.csv", "zone_count_metrics.csv"):
        shutil.copy2(output / name, paper_output / name)
    report = {
        "dataset_id": config["dataset_id"], "camera_count": len(cameras), "frame_count_per_camera": total_frames,
        "shared_zone_count": len(zone_ids), "projected_annotation_count": len(projected_rows),
        "representative_frame": representative_frame,
        "projection_metrics": projection_metrics, "zone_count_metrics": zone_metrics,
        "inputs": input_files,
        "paper_outputs": sorted(str(path) for path in paper_output.iterdir()),
        "notes": config["notes"],
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
