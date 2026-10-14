# Step 3: ATC congestion and observation-state analytics

## Reproducible command

```bash
make atc-analytics
```

Configuration is versioned in `configs/analytics/atc-step3.json`. Outputs are
written to the ignored directory
`data/processed/atc/atc-20121114/analytics/`.

## Method

1. Partition the metric ROS map into 5 m square candidates.
2. Treat non-occupied PGM cells as candidates because the supplied map has no
   normal ROS free-valued pixels.
3. Retain a candidate only when at least 1,000 trajectory rows support it.
4. Group frames into 60-second bins.
5. For every source timestamp, count simultaneous people per zone; average
   these frame counts within the bin.
6. Divide the mean simultaneous count by candidate traversable zone area.
7. Preserve continuous density and derive display levels from the 50th and
   85th percentiles of positive density.
8. Apply a deterministic 60% fixed-coverage evaluation mask.
9. Send one simulated robot to two blind zones per bin in round-robin order.
10. Mark robot values stale after 180 seconds without a revisit.

The CCTV mask is not a camera-calibration result, and the robot schedule is not
a navigation result. Those are controlled Step 3 inputs for testing the visual
analytics state model. Camera geometry and navigation are later components.

## Current result

| Measure | Result |
|---|---:|
| Candidate zones | 336 |
| Trajectory-supported zones | 70 |
| Analysis area | 1,663.365 m² |
| Source frames | 901,524 |
| 60-second bins | 654 |
| Ground-truth records | 45,780 |
| Fixed zones | 43 |
| Blind zones | 27 |
| Robot observation events | 1,308 |
| Fixed covered area | 61.19% |
| Initial unknown area | 35.96% |
| Final unknown area | 0% |

Empirical display thresholds are 0.007093 and 0.032803 persons/m². They yield
30,451 Low, 10,730 Medium, and 4,599 High records. These are relative analysis
levels, not universal crowd-safety thresholds. Absolute reference thresholds
of 0.5 and 1.5 persons/m² remain in the configuration for comparison.

All 27 blind zones exhibit robot-recovered and stale states. Unknown density is
empty, not zero; stale records retain their last observation, age, source, and
decayed confidence.

## Known limitations

- The ATC PGM does not directly identify walkable floor area; trajectory
  support defines the analysis region in this iteration.
- Regular 5 m zones are reproducible but not semantic rooms or corridors.
- The deterministic mask supports controlled coverage experiments but does not
  model occlusion or calibrated FoV.
- Quantile levels are useful for within-day exploration and should not be
  interpreted as safety categories across venues.
