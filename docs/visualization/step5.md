# Step 5 — Linked visual analytics dashboard

## Goal

Step 5 turns the Step 3 congestion/coverage products and Step 4 planning
products into one linked, browser-based research prototype. It is a static
dashboard: the analytical CSV/GeoJSON files remain authoritative and the web
bundle is a disposable view artifact.

## Views and interactions

- **Spatial situation:** density or observation-coverage state by zone, with
  baseline and congestion-aware paths overlaid.
- **Evidence inspector:** observed density, relative level, observation age,
  confidence, source, and the held-out density proxy for the selected zone.
- **Temporal context:** minute-level numeric crowd density in `person/m²`, with
  an area-weighted mean, P90 zone density, empirical congestion bands, replan
  markers, and direct time selection. P90 means 90% of zones are at or below
  that density and makes localized congestion visible without letting a single
  maximum-value zone dominate the chart.
- **Path comparison:** distance, simulated ETA, congestion exposure, and
  stale/unknown distance for a selected robot.
- The time slider, playback, robot selector, zone selection, and event list
  update all views together.

## Data boundary

`scripts/export_research_data.py` creates `source-data/dashboard.json`. The
export includes 70 aggregate zones, 654 minute frames, route signatures, and
replanning events. It deliberately excludes individual IDs and raw x/y
trajectory samples. `UNKNOWN` remains distinct from zero congestion.

Ground-truth density is included only as an aggregate, held-out evaluation
proxy. Relative Low/Medium/High labels use within-day empirical quantiles;
they are not general safety thresholds. ETA remains a simulation proxy.

## Build and inspect

```bash
make web-build
python3 -m http.server 8000 --directory apps/web/dist
```

Open `http://localhost:8000`. The build has no package-manager or runtime
dependency. Run `make test` to check dimensions, linked-view alignment,
planning pairs, and excluded trajectory fields.

## Step 5 acceptance check

- Switching Density/Coverage changes the map legend and zone colors.
- Slider, playback, and numeric density-chart clicks synchronize the four views.
- Selecting a zone exposes provenance and freshness without person-level data.
- Selecting a robot or replanning event updates both route alternatives.
- Baseline and congestion-aware metrics remain explicitly comparable.
