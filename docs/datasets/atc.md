# ATC Shopping Center adapter

## Scope and source

The adapter targets the official ATC pedestrian tracking dataset published by
ATR Interaction Science Laboratories. The official page reports approximately
900 m² of coverage, 92 collection days, headerless eight-column trajectory
CSV, and a ROS `map_server` occupancy grid in the same coordinate system.

The source is research-only and requires citation. Source archives are ignored
by Git and are not redistributed. Exact source URLs, access date, and citation
are recorded in `configs/datasets/atc.json`.

## Reproducible flow

```text
official archive (immutable)
  -> manual download and extraction
  -> /media/engineer/DATA1/datasets/ATC (read-only is supported)
  -> map/trajectory normalization
  -> /home/engineer/VACaN/data/processed/atc/atc-20121114 (writable derivatives)
  -> spatial alignment report
```

Download is intentionally gated by `--accept-research-only-license`. The
downloader supports only the official map and single-day development sample;
the nine multi-GB archives are not automated in this phase. In this workspace,
the user-managed extracted source is located at `/media/engineer/DATA1/datasets/ATC`.

```bash
PYTHONPATH=src python -m vacan.datasets.atc.download \
  --resource map --accept-research-only-license
PYTHONPATH=src python -m vacan.datasets.atc.download \
  --resource sample --accept-research-only-license
```

Example archive extraction is kept explicit because availability differs by platform:

```bash
unzip ATC-map.zip -d /media/engineer/DATA1/datasets/ATC
bsdtar -xf atc-20121114.7z -C /media/engineer/DATA1/datasets/ATC
```

Normalize and validate:

```bash
PYTHONPATH=src python -m vacan.datasets.atc.cli map \
  /media/engineer/DATA1/datasets/ATC/ATC-map/localization_grid.yaml \
  /home/engineer/VACaN/data/processed/atc/atc-20121114/map

PYTHONPATH=src python -m vacan.datasets.atc.cli trajectory \
  /media/engineer/DATA1/datasets/ATC/atc-20121114.csv \
  /home/engineer/VACaN/data/processed/atc/atc-20121114/trajectories.csv

PYTHONPATH=src python -m vacan.datasets.atc.cli align \
  /media/engineer/DATA1/datasets/ATC/ATC-map/localization_grid.yaml \
  /home/engineer/VACaN/data/processed/atc/atc-20121114/trajectories.csv \
  /home/engineer/VACaN/data/processed/atc/atc-20121114/alignment.json
```

## Conversion decisions

- Millimeters and mm/s become meters and m/s.
- Unix seconds become UTC ISO 8601 while the original numeric time is retained.
- The PGM is interpreted with ROS `map_server` threshold semantics.
- Image row order is inverted before applying the map origin and yaw.
- Consecutive same-state cells are emitted as run polygons to keep web GeoJSON
  smaller than one-feature-per-pixel output.
- Alignment reports the fraction of trajectory samples inside the rotated map
  bounds. It does not silently discard outliers.

## Acceptance criterion

The fixture requires 100% inside-map coordinates. For the official sample, the
initial warning threshold is 95%. A lower value blocks downstream zone-density
work until units, origin, map selection, and outliers have been inspected.

## ATC occupancy encoding note

The downloaded PGM contains predominantly ROS `UNKNOWN` pixels and occupied
pixels, with no normal free-valued cells. Step 3 therefore treats non-occupied
cells as candidate traversable space, then retains only 5 m zones supported by
at least 1,000 trajectory rows. This is an explicit ATC-specific analysis-region
rule, not a claim that all unknown map cells are physically walkable.
