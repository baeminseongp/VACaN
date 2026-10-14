"""Explicit, license-gated downloader for the two ATC development resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


RESOURCES = {
    "map": ("https://dil.atr.jp/crest2010_HRI/ATC_dataset/ATC-map.zip", "ATC-map.zip"),
    "sample": ("https://dil.atr.jp/crest2010_HRI/ATC_dataset/atc-20121114.7z", "atc-20121114.7z"),
}


def download(resource: str, destination_directory: Path) -> dict[str, object]:
    url, filename = RESOURCES[resource]
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / filename
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    expected_size: int | None = None
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:  # noqa: S310
        content_length = response.headers.get("Content-Length")
        expected_size = int(content_length) if content_length else None
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
            size += len(block)
    if expected_size is not None and size != expected_size:
        raise OSError(f"incomplete download: expected {expected_size} bytes, received {size}")
    partial.replace(destination)
    record = {
        "resource": resource,
        "url": url,
        "filename": filename,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }
    (destination_directory / f"{filename}.download.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", choices=sorted(RESOURCES), required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/media/engineer/DATA1/VACaN/datasets/atc/downloads"),
    )
    parser.add_argument("--accept-research-only-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_research_only_license:
        parser.error("review the official terms and pass --accept-research-only-license")
    record = download(args.resource, args.destination)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
