# Step 7 — Mobile sensing and dense-crowd/VLM extension

## Scope

Step 7 adds two independent evaluation tracks. JRDB evaluates observations made
from a robot-height mobile sensor. Mall evaluates image-level crowd counts and
structured VLM descriptions in a denser shopping-mall scene. Their records are
never fused with one another or projected onto the ATC floor map because the
datasets represent different venues and do not share a coordinate system.

The official JRDB site describes 60,000 annotated frames, 2.4 million 2D human
boxes, robot-centric 360-degree imagery, and both stationary and dynamic sensor
perspectives. Access requires an account. Mall contains 2,000 640×480 frames
with more than 60,000 annotated pedestrian head positions and is restricted to
research, non-commercial use.

## JRDB mobile-sensing track

Input is the official `labels_2d` JSON shape: a top-level `labels` mapping from
frame names to person-label lists. Each configured sequence must state its
sensor motion as `STATIONARY`, `DYNAMIC`, or `UNKNOWN`; motion is not guessed
from imagery.

The adapter produces:

- `frame_metrics.csv`: total and visible people, severe occlusion, new tracks,
  active tracks, and unioned horizontal panorama occupancy per frame;
- `track_continuity.csv`: observed frames divided by each track's frame span;
- `summary_metrics.csv`: separate stationary, dynamic, and overall summaries;
- `evaluation_report.json`: hashes, license, inputs, outputs, and limitations.

Panorama occupancy is an image-space observation-yield proxy. It is not floor
coverage, density per square metre, or a substitute for JRDB 3D evaluation.
Stationary and dynamic summaries are descriptive unless sequences are matched
under a separate experimental design.

## Mall dense-crowd/VLM track

Ground truth may be the official `mall_gt.mat` or a normalized head-point CSV
with `frame_id,x_px,y_px`. Reading the MATLAB file requires the optional Mall
dependency:

```bash
python -m pip install -e '.[mall]'
```

VLM predictions use this CSV contract:

```text
frame_id,predicted_count,predicted_level,description,grounded_evidence
```

The committed prompt at `configs/prompts/mall-vlm-counting.txt` forbids person
identification, sensitive-attribute inference, hidden-cause speculation, and
claims of metric floor density. Images and free-form model responses remain
outside Git and the public web bundle.

Metrics are count MAE, RMSE, bias, fraction within five people, congestion-level
accuracy, macro-F1, and evidence-field completeness. Low/Medium/High thresholds
are scenario display thresholds in configuration, not universal safety limits.

## Reproduce without licensed datasets

The fixtures mimic the official label structures but contain no JRDB or Mall
media. They validate the complete contracts and output separation:

```bash
make step7-fixture
make test
```

Fixture outputs are written independently to:

```text
data/processed/jrdb/step7-fixture/
data/processed/mall/step7-fixture/
```

They are implementation checks and must not be reported as benchmark results.

## Run with real data

1. Request and download JRDB from <https://jrdb.erc.monash.edu/> into
   `/media/engineer/DATA1/datasets/JRDB`.
2. Set the real sequence IDs, annotation paths, and verified motion labels in
   `configs/datasets/jrdb-step7.json`, then run `make jrdb-evaluation`.
3. Download Mall from
   <https://staff.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html> into
   `/media/engineer/DATA1/datasets/Mall`.
4. Produce a private structured prediction CSV under
   `data/interim/mall/vlm_predictions.csv`, verify thresholds, then run
   `make mall-evaluation`.

## Web publication boundary

Only aggregate `summary_metrics.csv` values and non-identifying provenance may
be added to the dashboard export. JRDB/Mall images, labels, raw VLM responses,
local paths, and per-person records remain excluded. A future web change must
first add an aggregate-only exporter and tests proving that these fields do not
enter `apps/web/dist`.

## Acceptance check

- JRDB and Mall each generate their own report and metrics tables.
- Input hashes and license statements are retained.
- Missing Mall prediction frames fail instead of being silently dropped.
- Sensor motion is explicit rather than inferred.
- No Step 7 imagery or person-level record is exported to the website.
- Fixture pipelines and the full repository test suite pass.
- Real-data benchmark claims remain blocked until the licensed datasets and
  real prediction outputs are supplied.
