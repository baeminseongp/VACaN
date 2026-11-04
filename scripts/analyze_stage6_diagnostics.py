#!/usr/bin/env python3
"""Create non-causal Stage 6 candidate diagnostic intervals from saved logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vacan.reference.comparison import group_candidate_intervals, information_error_candidates


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        candidates = information_error_candidates(
            _jsonl(run / "drl-trajectory.jsonl"), _jsonl(run / "ground-truth-drl.jsonl"),
            crowd_error_threshold=1.0, intention_error_threshold=1.0,
            speed_change_threshold_m_s=0.2, angular_change_threshold_rad_s=0.5,
            local_penalty_threshold=0.5,
        )
        intervals = group_candidate_intervals(candidates, maximum_gap_s=0.25)
        output = {"schema_version": 1, "causal_interpretation": False,
                  "candidate_count": len(candidates), "candidate_interval_count": len(intervals),
                  "intervals": intervals, "candidates": candidates}
        (run / "diagnostic-candidates.json").write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"run": str(run), "candidate_count": len(candidates),
                          "candidate_interval_count": len(intervals)}, sort_keys=True))


if __name__ == "__main__":
    main()
