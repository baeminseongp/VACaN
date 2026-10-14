# CAVIAR Lisbon dataset

## Local source

The read-only source root is `/media/engineer/DATA1/datasets/CAVIAR`. Step 6
uses the synchronized Lisbon `WalkByShop1` pair:

- `raw/Lisbon/WalkByShop1cor/WalkByShop1cor.mpg` and `cwbs1gt.xml`
- `raw/Lisbon/WalkByShop1front/WalkByShop1front.mpg` and `fwbs1gt.xml`
- `metadata/shoppingplane.png` and the archived official source page

`SHA256SUMS` records the retrieved bytes. The pipeline records the four primary
video/XML hashes again in its evaluation report.

## Source and terms

The official source is <https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/>.
Publication must acknowledge the EC Funded CAVIAR project/IST 2001 37540. The
source page describes the data as Creative Commons BY-SA. Raw video is not
committed or redistributed by this repository.

## Known data characteristics

- 384×288 MPEG video, decoded at 25 fps.
- 2,360 frames in each selected view and matching XML frame records.
- View-specific hand-labeled bounding boxes.
- Four image/ground correspondences per camera, with ground units published in
  centimetres and converted to metres by the adapter.
- The official page warns that the paired streams can contain occasional
  duplicate/missing frames; equal frame indices are the first-pass alignment.

## Derived-data boundary

Only boxes, bottom-center foot proxies, metric projections, aggregate zone
counts, metrics, and figures are written under `data/processed`. No identity
inference, face processing, or raw-frame redistribution is performed.
