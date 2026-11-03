"""Central YAML configuration loading and validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    required = {"project", "reproducibility", "simulation", "logging", "checkpoints"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    factors = config["project"].get("research_factors")
    if factors != ["congestion", "intention", "freshness"]:
        raise ValueError("research_factors must preserve congestion, intention, freshness")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a VACaN YAML configuration")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    config = load_config(args.path)
    print(yaml.safe_dump(config, sort_keys=True), end="")


if __name__ == "__main__":
    main()
