# VACaN Visual Analytics

Buildless static dashboard for the Step 5 linked-view prototype.

```bash
python3 scripts/export_research_data.py
python3 scripts/build_site.py
```

The export contains only zone-level derived data, route summaries, CAVIAR
zone-count evidence, and event provenance. Raw trajectories and identifiable
imagery are excluded.

The resulting `dist/` directory is the Sites static deployment artifact.

To inspect the downloaded CAVIAR videos from the main spatial map, run from the
repository root:

```bash
make web-serve
```

Then open <http://127.0.0.1:8000> and click an ATC grid cell in **Spatial
situation**. A deterministic, demo-only CAVIAR proxy appears in the adjacent
**Evidence inspector**. The mapping normalizes ATC centroids into the CAVIAR
grid and snaps to the nearest CAVIAR zone containing evidence; it is not a real
cross-venue calibration. `web-media` converts the read-only MPEG-1 sources into
browser-compatible local MP4 derivatives. The local server exposes only those
two allow-listed files and supports byte ranges for seeking. Neither the source
videos nor the MP4 derivatives are copied into `dist/` or published.
