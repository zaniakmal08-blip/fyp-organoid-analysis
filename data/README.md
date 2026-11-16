# Dataset organisation

The original collection is `2026 OCT Organoid toxicity ML images`. The saved audit reports 750 main TIFFs (747 RGB, 3 16-bit), all 2048 × 1536. `metadata/raw_inventory.csv` lists all files including additional subfolders and sizes. This inventory is not a biological annotation or a train/test split.

`pilot/` holds 20 selected images and six available mask/ROI/measurement sets locally. `metadata/pilot_manifest.json` records exact sizes and SHA-256 hashes. `metadata/annotation_manifest.csv` preserves the source annotation register; review status remains unreviewed unless explicitly updated after expert review.

Raw TIFFs and annotation payloads are ignored by Git. Restore them from authorised lab storage or the original FYP folder using `scripts/import_local_data.py`. Share approved data separately with authorised reviewers. No public redistribution permission is assumed. Do not upload the separate NC004 workbook or infer matches from similar well labels.

Before training, verify image/mask shape, binary labels, ROI agreement and review status. Keep the same well/specimen across days within one split. Record batch, acquisition settings, treatment, concentration, day, annotation source, reviewer and calibration. Read `DATA_NOTES.md` before using original CSV measurements.
