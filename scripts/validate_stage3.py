#!/usr/bin/env python3
"""Validate Stage 3 costmap artifacts and privileged-state separation."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path


def read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def validate_episode(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    policy = read_gzip_jsonl(directory / "policy_costmap.jsonl.gz")
    diagnostics = read_gzip_jsonl(directory / "diagnostics.jsonl.gz")
    privileged = read_gzip_jsonl(directory / "privileged_ground_truth.jsonl.gz")
    if summary.get("status") != "pass" or not policy or len(policy) != len(diagnostics):
        raise AssertionError(f"{directory}: incomplete Stage 3 output")
    if len(policy) != len(privileged) or len(policy) != summary["ticks"]:
        raise AssertionError(f"{directory}: stream lengths differ")
    expected_shape = tuple(summary["costmap_shape"])
    previous_time = None
    cctv_means = []
    for index, (nav, diag, truth) in enumerate(zip(policy, diagnostics, privileged, strict=True)):
        if "ground_truth_count" in nav or "privileged" in nav:
            raise AssertionError(f"{directory}: policy stream exposes privileged Ground Truth")
        if "ground_truth_count" not in truth:
            raise AssertionError(f"{directory}: privileged stream is incomplete")
        now = float(nav["simulation_time_s"])
        if previous_time is not None:
            interval = now - previous_time
            if not math.isclose(interval, 0.2, abs_tol=0.017):
                raise AssertionError(f"{directory}: non-5Hz interval {interval}")
        previous_time = now
        rows = nav["costmap"]
        if (len(rows), len(rows[0])) != expected_shape:
            raise AssertionError(f"{directory}: costmap shape mismatch")
        if any(not isinstance(value, (int, float)) for row in rows for value in row):
            raise AssertionError(f"{directory}: nonnumeric cost")
        counts = list(diag["cctv_counts"].values())
        expected_mean = sum(counts) / len(counts)
        if not math.isclose(diag["cctv_mean"], expected_mean, abs_tol=1e-12):
            raise AssertionError(f"{directory}: CCTV mean excludes or miscounts cells")
        cctv_means.append(diag["cctv_mean"])
        mask = nav["cctv_coverage_mask"]
        for row in range(expected_shape[0]):
            for column in range(expected_shape[1]):
                freshness = diag["freshness"][row][column]
                if not 0.0 <= freshness <= 1.0:
                    raise AssertionError(f"{directory}: freshness outside [0,1]")
                if mask[row][column] and diag["last_update_type"][row][column] != "cctv":
                    raise AssertionError(f"{directory}: CCTV provenance overwritten")
    if not (directory / "costmap.png").is_file():
        raise AssertionError(f"{directory}: visualization missing")
    frame_manifest = json.loads((directory / "frames" / "frames.json").read_text(encoding="utf-8"))
    frames = frame_manifest.get("frames", [])
    if len(frames) != len(policy):
        raise AssertionError(f"{directory}: expected one visualization per costmap tick")
    for index, frame in enumerate(frames):
        if frame["frame_index"] != index or not (directory / "frames" / frame["filename"]).is_file():
            raise AssertionError(f"{directory}: missing frame {index}")
        if not math.isclose(frame["simulation_time_s"], policy[index]["simulation_time_s"]):
            raise AssertionError(f"{directory}: frame timestamp mismatch")
    return {
        "ticks": len(policy), "shape": list(expected_shape),
        "visualization_frames": len(frames),
        "cctv_mean_range": [min(cctv_means), max(cctv_means)],
        "observations": summary["observations"], "intention_events": summary["intention_events"],
        "processing_ms_per_tick": summary["processing_ms_per_tick"],
    }


def uncompressed_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return path.read_bytes()


def compare_replay(primary: Path, repeated: Path) -> None:
    for scenario in ("office", "hospital", "warehouse"):
        for name in (
            "policy_costmap.jsonl.gz", "diagnostics.jsonl.gz",
            "privileged_ground_truth.jsonl.gz", "intention_events.jsonl",
        ):
            if uncompressed_bytes(primary / scenario / name) != uncompressed_bytes(repeated / scenario / name):
                raise AssertionError(f"nondeterministic replay: {scenario}/{name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("artifacts/stage3"))
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    results = {item["scenario"]: validate_episode(args.output / item["path"])
               for item in manifest["episodes"]}
    if set(results) != {"office", "hospital", "warehouse"}:
        raise AssertionError(f"scenario coverage missing: {sorted(results)}")
    if args.replay_output:
        compare_replay(args.output, args.replay_output)
    print(json.dumps({"status": "PASS", "episodes": results,
                      "deterministic_replay": args.replay_output is not None}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
