# Visual artifact index and provenance

These files are compact copies or derivatives of local acceptance artifacts.
They are provided for implementation understanding and web mockups.

## Stage 2 trajectory views

| Environment | File | Original source |
|---|---|---|
| Office | `office-stage2-top-down.png` | `artifacts/stage2-isaac/train/office/episode-000/top_down.png` |
| Hospital | `hospital-stage2-top-down.png` | `artifacts/stage2-isaac/train/hospital/episode-000/top_down.png` |
| Warehouse | `warehouse-stage2-top-down.png` | `artifacts/stage2-isaac/train/warehouse/episode-000/top_down.png` |

These images summarize Nova Carter and pedestrian trajectories. They are not
camera images and do not represent the policy observation tensor.

## Stage 4 voxel-map views

| Environment | File | Original source |
|---|---|---|
| Office | `office-stage4-voxel-map.png` | `artifacts/stage4/office/voxel_map_3d.png` |
| Hospital | `hospital-stage4-voxel-map.png` | `artifacts/stage4/hospital/voxel_map_3d.png` |
| Warehouse | `warehouse-stage4-voxel-map.png` | `artifacts/stage4/warehouse/voxel_map_3d.png` |

The plots visualize occupancy extracted from official scene collision geometry.
Navigation uses the generated numeric map, not pixels sampled from these PNGs.

## Stage 3 costmap replay

`office-stage3-costmap-replay.mp4` is a 12.2-second, 5fps H.264 replay made from
61 ordered files under `artifacts/stage3/office/frames/`. It is a derived
visualization, not simulator-camera footage. It can be regenerated with:

```bash
ffmpeg -framerate 5 -pattern_type glob \
  -i 'artifacts/stage3/office/frames/frame-*.png' \
  -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' \
  -c:v libx264 -crf 25 -preset slow -pix_fmt yuv420p \
  -movflags +faststart docs/media/office-stage3-costmap-replay.mp4
```

The authoritative layer values remain in the Stage 3 JSONL outputs. Use the
video only to understand temporal behavior or prototype playback controls.
