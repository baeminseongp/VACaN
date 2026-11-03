#!/usr/bin/env python3
"""Render Stage 3 summary and one eight-layer PNG per costmap tick."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vacan-mpl")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def read_jsonl_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def render_frame(policy, diagnostics, config, output, cost_max, cctv_max, dpi):
    origin_x, origin_y = config["grid"]["origin_xy"]
    resolution = float(config["grid"]["resolution_m"])
    shape = np.asarray(policy["costmap"]).shape
    figure, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    static = np.asarray(policy["walkable_mask"], dtype=float)
    static[np.asarray(policy["static_obstacle_mask"], dtype=bool)] = -1.0
    cctv_counts = np.zeros(shape)
    for key, value in diagnostics["cctv_counts"].items():
        row, column = map(int, key.split(","))
        cctv_counts[row, column] = value
    layers = [
        (static, "Static layout", "coolwarm", -1.0, 1.0),
        (cctv_counts, "CCTV exact counts", "Blues", 0.0, cctv_max),
        (np.asarray(diagnostics["robot_visibility_mask"]), "Nova Carter / FOV", "Greens", 0.0, 1.0),
        (np.asarray(policy["costmap"]), "Dynamic crowd cost", "magma", 0.0, cost_max),
    ]
    for axis, (layer, title, cmap, low, high) in zip(axes[0], layers, strict=True):
        image = axis.imshow(layer, origin="lower", cmap=cmap, interpolation="nearest",
                            vmin=low, vmax=max(high, 1e-12))
        axis.set(title=title, xlabel="column", ylabel="row")
        figure.colorbar(image, ax=axis, fraction=0.046)
    robot_position = diagnostics["robot"]["position_xyz_m"]
    axes[0, 2].scatter((robot_position[0] - origin_x) / resolution,
                       (robot_position[1] - origin_y) / resolution,
                       marker="*", s=70, color="black")
    observed_axis = axes[1, 0]
    observed_axis.imshow(np.asarray(policy["walkable_mask"]), origin="lower", cmap="Greys", alpha=0.2)
    for item in diagnostics["observations"]:
        x = (item["position_xy_m"][0] - origin_x) / resolution
        y = (item["position_xy_m"][1] - origin_y) / resolution
        observed_axis.quiver(x, y, item["velocity_xy_m_s"][0], item["velocity_xy_m_s"][1],
                             color="tab:blue", angles="xy", scale_units="xy", scale=1)
    observed_axis.set(title="Observed people / velocity", xlabel="column", ylabel="row")
    predicted_axis = axes[1, 1]
    predicted_axis.imshow(np.asarray(policy["costmap"]), origin="lower", cmap="Greys", alpha=0.25)
    predicted = [item["predicted_destination_grid"] for item in diagnostics["observations"]
                 if item.get("predicted_destination_grid") is not None]
    if predicted:
        predicted_axis.scatter([cell[1] for cell in predicted], [cell[0] for cell in predicted],
                               marker="x", s=45, color="tab:orange")
    predicted_axis.set(title="Predicted destination", xlabel="column", ylabel="row")
    event_axis = axes[1, 2]
    incoming = np.asarray(diagnostics["intention_incoming_count"])
    image = event_axis.imshow(incoming, origin="lower", cmap="Reds", interpolation="nearest",
                              vmin=0, vmax=max(1, int(incoming.max())))
    event_axis.set(title=f"New intention events (n={len(diagnostics['intention_events'])})",
                   xlabel="column", ylabel="row")
    figure.colorbar(image, ax=event_axis, fraction=0.046)
    freshness_axis = axes[1, 3]
    image = freshness_axis.imshow(np.asarray(diagnostics["freshness"]), origin="lower",
                                  cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
    freshness_axis.set(title="Freshness / decay", xlabel="column", ylabel="row")
    figure.colorbar(image, ax=freshness_axis, fraction=0.046)
    figure.suptitle(f"{policy['scenario']} tick t={policy['simulation_time_s']:.3f}s, "
                    f"CCTV mean={diagnostics['cctv_mean']:.4f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path)
    args = parser.parse_args()
    policies = read_jsonl_gzip(args.directory / "policy_costmap.jsonl.gz")
    diagnostics = read_jsonl_gzip(args.directory / "diagnostics.jsonl.gz")
    if len(policies) != len(diagnostics) or not policies:
        raise ValueError("policy and diagnostic streams must be nonempty and aligned")
    config = yaml.safe_load((args.directory / "scenario.yaml").read_text(encoding="utf-8"))
    cost_max = max(1.0, max(max(map(max, record["costmap"])) for record in policies))
    cctv_max = max(1.0, max(max(record["cctv_counts"].values()) for record in diagnostics))
    render_frame(policies[-1], diagnostics[-1], config, args.output, cost_max, cctv_max, 160)
    if args.frames_dir is None:
        return
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256()
    for name in ("policy_costmap.jsonl.gz", "diagnostics.jsonl.gz"):
        with gzip.open(args.directory / name, "rb") as stream:
            input_hash.update(stream.read())
    input_hash.update((args.directory / "scenario.yaml").read_bytes())
    fingerprint = input_hash.hexdigest()
    expected = [
        {
            "frame_index": index,
            "simulation_time_s": float(policy["simulation_time_s"]),
            "filename": f"frame-{index:04d}-t{float(policy['simulation_time_s']):08.3f}s.png",
        }
        for index, policy in enumerate(policies)
    ]
    manifest_path = args.frames_dir / "frames.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        same_index = existing.get("frames") == expected
        all_present = same_index and all(
            (args.frames_dir / frame["filename"]).is_file() for frame in expected
        )
        # Accept the pre-fingerprint manifest once because Stage 3 replay is
        # deterministic; rewrite it immediately with the current input hash.
        compatible_hash = existing.get("input_sha256") in (None, fingerprint)
        if all_present and compatible_hash:
            existing["input_sha256"] = fingerprint
            manifest_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
            print(f"[Stage 3 visualization] {policies[0]['scenario']}: "
                  f"reused {len(expected)} cached frames", flush=True)
            return
    total = len(expected)
    for index, (policy, diagnostic) in enumerate(zip(policies, diagnostics, strict=True)):
        filename = expected[index]["filename"]
        render_frame(policy, diagnostic, config, args.frames_dir / filename,
                     cost_max, cctv_max, 100)
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print(f"[Stage 3 visualization] {policies[0]['scenario']}: "
                  f"{index + 1}/{total} frames", flush=True)
    manifest_path.write_text(
        json.dumps({"scenario": policies[0]["scenario"], "input_sha256": fingerprint,
                    "frames": expected},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
