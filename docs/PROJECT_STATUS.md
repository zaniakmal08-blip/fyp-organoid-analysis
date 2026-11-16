# Project status — 16 November 2026

## Implemented

- Single-plane TIFF upload, normalisation and calibration confirmation.
- Tiny U-Net segmentation with aspect-preserving preprocessing.
- Predicted polygon, Fiji ROI and eight ImageJ measurements.
- CSV/JSON/overlay exports with provenance.
- CNN inspector and numerical verification scripts.
- Six-image pilot fitting and saved training-image comparison.

## Not yet established

- Independent reference annotations and expert-approved boundary convention.
- Reliable spatial calibration and repaired reference measurement CSVs.
- A group-separated train/validation/test split.
- Unseen-image segmentation or measurement accuracy.
- Mapping between the NC004 workbook and October toxicity TIFFs.
- A confirmed biological prediction target or final extension scope.

## Decisions for supervision

Confirm the measurable FYP objective and success criteria; agree the annotation boundary; request matching image/ROI/calibration references; establish whether prediction or microscope capture is feasible given labels, time and hardware. Keep possible halo/differentiation features separate from the organoid measurement task until suitable references exist.

## Latest repository milestone

Curated the existing prototype into portable code/data/model/documentation folders. This is an organisation milestone, not a new model-performance result.
