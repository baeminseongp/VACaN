"""Transcode read-only CAVIAR MPEG-1 videos for local browser playback."""

from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path("/media/engineer/DATA1/datasets/CAVIAR/raw/Lisbon")
OUTPUT_ROOT = PROJECT_ROOT / "data/processed/caviar/walkbyshop1/media"
SOURCES = {
    "corridor.mp4": DATASET_ROOT / "WalkByShop1cor/WalkByShop1cor.mpg",
    "front.mp4": DATASET_ROOT / "WalkByShop1front/WalkByShop1front.mpg",
}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        if not source.is_file():
            raise SystemExit(f"Missing CAVIAR video: {source}")
        output = OUTPUT_ROOT / name
        if output.is_file() and output.stat().st_mtime >= source.stat().st_mtime:
            print(f"Ready {output}")
            continue
        temporary = output.with_suffix(".working.mp4")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-i", str(source), "-an", "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(output)
        print(f"Prepared {output}")


if __name__ == "__main__":
    main()
