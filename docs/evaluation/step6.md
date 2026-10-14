# Step 6 — CAVIAR real-video projection validation

## Scope

Step 6 validates the CCTV-to-ground-to-zone path using the real Lisbon shopping
centre pair `WalkByShop1cor` and `WalkByShop1front`. Both MPEG streams contain
2,360 decoded frames at 384×288 and 25 fps; the matching XML files also contain
2,360 frame records.

For each hand-labeled person box, the adapter uses its bottom center as a ground
contact proxy, applies the official view-specific homography, and assigns a
shared 1 m ground zone. The common evaluation support is the intersection of
the published calibration rectangles: x=0–2.9 m and y=0–9.75 m, divided into 30
zones.

## Projection metrics

Each view has exactly four official point correspondences, which exactly
determine an eight-degree-of-freedom homography. Consequently, the reported
control-point residual and algebraic round-trip error are approximately zero
by construction. They verify implementation consistency but are **not an
independent localization-accuracy estimate**.

The support check is more informative:

| Camera | Annotations | Inside view calibration | Inside common support |
|---|---:|---:|---:|
| Corridor | 11,348 | 11.31% | 10.05% |
| Front | 1,012 | 39.43% | 31.62% |

Points outside the published calibration support remain in
`projected_detections.csv` with an empty `zone_id`; they are not silently
clamped into edge zones. This avoids treating strong homography extrapolation
as validated ground position.

## Zone-count metrics

The two independently labeled synchronized views are compared only inside the
common calibration support over all 2,360 supplied frame indices.

| Metric | Result |
|---|---:|
| Total-count MAE per frame | 0.4326 persons |
| Total-count RMSE per frame | 0.8875 persons |
| Exact total-count agreement | 70.38% |
| Per-zone/frame MAE | 0.02064 persons |
| Per-zone/frame RMSE | 0.14839 persons |
| Exact per-zone/frame agreement | 98.01% |

These values measure cross-view annotation/projection consistency, not detector
accuracy. Differences can result from camera visibility, occlusion, distinct
annotation scope, and the duplicate/missing-frame caveat documented by CAVIAR.

## Reproduce

```bash
make caviar-evaluation
make test
```

For linked visual inspection of the projected zones and synchronized video:

```bash
make web-serve
```

Open `http://127.0.0.1:8000` and click an ATC cell in the existing **Spatial
situation** map. The adjacent **Evidence inspector** shows a deterministic
CAVIAR demonstration proxy and seeks to a representative annotated frame;
previous/next controls traverse other non-empty frames. The mapping normalizes
ATC zone centroids into the 3×10 CAVIAR grid and snaps to the nearest CAVIAR
zone with evidence. It is explicitly labeled **DEMO ONLY / NOT SPATIALLY
ALIGNED** and must not be reported as a real ATC camera calibration. Video is
local-only and is never included in the deployable `dist/` artifact.

The configuration is `configs/datasets/caviar.json`. Generated outputs are
written to `data/processed/caviar/walkbyshop1` and include homographies,
projected detections, zone counts, frame comparisons, metrics, input hashes,
and a ground-zone projection figure. Raw videos and XML remain read-only under
`/media/engineer/DATA1/datasets/CAVIAR`.

## Acceptance check

- Both annotation streams contain 2,360 frames and match their decoded videos.
- Official corridor/front correspondences reproduce their ground coordinates.
- Every in-support projected detection maps to exactly one shared zone.
- Out-of-support projections remain explicit and are excluded from zone counts.
- Projection residuals are labeled in-sample rather than held-out accuracy.
- Cross-view count results retain visibility and synchronization limitations.
