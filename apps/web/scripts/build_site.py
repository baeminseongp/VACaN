"""Create the buildless static Sites artifact."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
DIST = ROOT / "dist"


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(STATIC, DIST)
    data_directory = DIST / "data"
    data_directory.mkdir(parents=True)
    dashboard = ROOT / "source-data/dashboard.json"
    shutil.copy2(dashboard, data_directory / "dashboard.json")
    digest = hashlib.sha256()
    for path in (STATIC / "app.js", STATIC / "styles.css", dashboard):
        digest.update(path.read_bytes())
    version = digest.hexdigest()[:12]
    index = DIST / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace("__BUILD_VERSION__", version),
        encoding="utf-8",
    )
    print(f"Built {DIST} (version {version})")


if __name__ == "__main__":
    main()
