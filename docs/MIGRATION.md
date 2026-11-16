# Curation record

Created 16 November 2026 as a separate repository inside the original FYP folder. Originals were not moved or deleted.

| Original | Repository | Decision |
|---|---|---|
| training_pilot core app/training/visualizer | src/ | Keep active code; make paths portable |
| training_pilot/imagej_helper | src/imagej_helper | Keep measurement bridge |
| training_pilot/prototype_model | models/ | Keep selected small checkpoint and fitting records |
| training_pilot/verify_*.py | tests/ | Keep integration checks |
| annotation_pilot_20 | data/pilot + metadata | Copy locally; track manifest, not image payloads |
| dataset_review/audit_summary.json | data/metadata | Preserve source audit |
| full original image collection | data/metadata/raw_inventory.csv | Inventory without duplicate multi-GB copy |
| six_image_update/evaluation_round2 | experiments/six-image-pilot | Keep selected numerical evidence |
| one-off comparison/audit scripts, old checkpoints and large outputs | original folder | Preserve as history outside active repo |
| recordings, transcripts, workbook, CV and administrative forms | original folder | Exclude from repository |
| environment caches and transcription packages | original folder | Exclude reproducible dependencies |

The new setup guide supersedes old machine-specific launch instructions. The full 19-method OpenCV comparison described in earlier project discussion was not found in this folder and is not claimed as an included deliverable.
