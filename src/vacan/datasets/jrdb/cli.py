"""Command-line entry point for the JRDB Step 7 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_jrdb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    report = evaluate_jrdb(parser.parse_args().config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
