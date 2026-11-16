# FYP project handover for Codex and Antigravity

Last reviewed: **16 November 2026**. This is a snapshot, not proof that future files or decisions remain unchanged. Read current code, Git state and newer progress entries before acting.

## 1. Identity and purpose

Owner: **Syazani Akmal**. Repository: https://github.com/zaniakmal08-blip/fyp-organoid-analysis (public).

Working topic: automated organoid image segmentation and quantitative 2D morphology measurement. The aim of the existing engineering prototype is to help identify an organoid boundary in a microscope image and reproduce the kinds of measurements obtained manually with Fiji/ImageJ. The final FYP title, exact scope and acceptance targets still need supervisor confirmation.

The intended repository audience is the student, supervisors and other reviewers. It is both an engineering workspace and a record of research progress. Its organisation was inspired by the team's Solar Filament competition repository; that is a separate project with different data, models and metrics.

## 2. Problem being addressed

A manual workflow involves opening a microscope image, confirming spatial scale, selecting or tracing the organoid and measuring the selected region. Tracing and calibration choices affect repeatability. The prototype automates part of this workflow while retaining visual inspection and calibration confirmation.

The target is eight morphometric quantities:

| Measurement | Meaning | Units after verified calibration |
|---|---|---|
| Area | Size of the selected 2D region | micrometres squared |
| Perimeter | Length of its boundary | micrometres |
| Circularity | Shape descriptor based on area and perimeter | dimensionless |
| Aspect ratio | Ratio of fitted major/minor axes | dimensionless |
| Roundness | Shape descriptor related to area and major axis | dimensionless |
| Solidity | Area relative to its convex hull | dimensionless |
| Feret | Maximum caliper diameter | micrometres |
| MinFeret | Minimum caliper diameter | micrometres |

These are image-region measurements. They do not directly determine 3D volume, neuron percentage, viability, differentiation or treatment response. Those interpretations require suitable independent evidence.

## 3. What exists now

The working pipeline is:

1. Load one single-plane TIFF and reject unsupported/invalid inputs.
2. Convert to grayscale and normalise using image intensity percentiles.
3. Resize while preserving aspect ratio and pad to the model input size.
4. Run a small U-Net for binary organoid segmentation.
5. Restore predictions to original image coordinates, select the principal region and derive an outline.
6. Use the Java helper and Fiji/ImageJ to obtain measurements and a Fiji-compatible ROI.
7. Export masks, overlays, measurements, provenance and a ZIP from a localhost browser app.

There is also a CNN inspector that displays operations, activations and numerical calculations. It is an educational/debugging tool, not an independent model-validation method.

Key implementation files are mapped in [the working guide](WORKING_GUIDE.md). The selected checkpoint is small (about 128 KB) and is tracked in Git. TinyUNet has 29,321 parameters. Current preprocessing is aspect-preserving; earlier square-resize baseline checkpoints are incompatible with the current app.

## 4. Dataset state and provenance

The original local collection is named `2026 OCT Organoid toxicity ML images`. The saved audit reports **750 main images**, all 2048 x 1536: 747 RGB and three 16-bit images. The collection has additional subfolders; `data/metadata/raw_inventory.csv` lists files beyond those 750 main images. Do not equate inventory rows with independent biological specimens.

The annotation pilot has **20 selected TIFFs**, with **six available mask/ROI/measurement sets**. The copied local pilot contains 38 payload files: 20 images plus six each of masks, ROIs and measurement CSVs. Filenames retain batch/plate/well, treatment, concentration, day and magnification information. Verify interpretation with the lab rather than treating filenames as biological ground truth.

The source annotation register is `data/metadata/annotation_manifest.csv`; expert review was pending at handover. `data/metadata/pilot_manifest.json` records sizes and hashes for the curated snapshot. That manifest is an integrity record, not a dynamic annotation database. If reviewed data are deliberately updated, preserve the prior dataset version and explicitly create/update the manifest after review; do not bypass a mismatch by silently changing hashes.

The full raw collection was not duplicated into the new repo. The original folder remains the archive. GitHub contains inventories and documentation, not the pilot TIFF/mask/ROI/CSV payloads. A clone therefore needs a separate authorised dataset transfer.

## 5. Current model and evidence

Selected checkpoint: `models/tiny_unet.pt`.

SHA-256 at handover: `f35b680a14671ea23703c7784be8d5bf8366e1262d797e9906b86de449fe5b44`.

All six annotated images were used for fitting. There is **no independent validation or test set**. Saved exported-mask comparison gives approximately **97.06% mean Dice on those same six images**. A04 is weaker at approximately **89.71% Dice**, with large perimeter and circularity errors. Do not report the mean as unseen-image accuracy.

Training scores in `models/training_metrics.json` and postprocessed exported-mask scores in `experiments/six-image-pilot/` refer to different output representations. Keep those labels distinct.

Historic selection notes describe 300 epochs, learning rate 0.001 and Dice-loss weight 3. The current training script warm-starts from the selected checkpoint by default. Running those settings today continues fitting; it is not an exact replay of the historical starting state.

## 6. Data problems that must remain visible

Read [DATA_NOTES.md](../data/DATA_NOTES.md) before using measurements:

- A04 and B07 original CSV sizes are consistent with pixel units, unlike calibrated earlier CSVs.
- A05's original CSV measures the entire mask canvas rather than the saved organoid ROI.
- Saved masks agreeing with ROI rasters establishes file consistency, not expert biological correctness.
- The recorded scale of approximately 0.61922572 micrometres/pixel is provisional and must not be assumed valid for every image.
- Some OME physical-size metadata are suspicious; the app requires operator confirmation rather than silently trusting them.
- The separate NC004 workbook has not been matched to the October toxicity TIFF collection. Similar well labels do not establish correspondence.

The numerical CSVs are not training targets for segmentation; images and masks are. Repair measurement references through reviewed Fiji measurements with verified calibration, while preserving originals and documenting changes.

## 7. Scope still requiring decisions

The technical supervision review explored organoid morphometry, a possible visible lighter/halo-like feature, and broader prediction/imaging possibilities. None of these discussions establishes a validated biological label or final extension requirement.

Questions for supervision:

1. Is the assessed objective segmentation plus morphometry, prediction, microscope capture, or a defined combination?
2. What is the correct target boundary and annotation tool/protocol?
3. Who can review matching TIFFs, ROIs and spatial calibration?
4. Which biological/sample groups should define independent evaluation?
5. What success criteria and error tolerances are meaningful for all eight measurements?
6. If prediction is required, what measured outcome and longitudinal labels exist?

Do not invent answers from this handover. Record actual decisions in dated technical meeting notes and update `PROJECT_STATUS.md`.

## 8. Verified engineering state

On 16 November 2026, the reorganised copy passed:

- Repository structure, Python syntax, Markdown-link and checkpoint-hash checks.
- Integrity verification for 38 local pilot files.
- 75 CNN pixel calculations across 25 operations (maximum absolute difference about 4.16e-6).
- Fiji integration checks against the original three reference measurements, prediction exports and image dimensions.
- Calibration scaling checks and rejection of blank images, stacks and PNG uploads.

The original GitHub repository-check workflow also passed. Evidence is in [VERIFICATION.md](VERIFICATION.md) and `results/verification-2026-11-16/`. Tests used an existing Windows environment; a fresh dependency installation was not verified. These tests do not validate the lab's calibration or model generalisation.

## 9. How to resume in Antigravity

Open the existing `fyp-organoid-analysis` folder as the project root. On the original computer it is under `C:\Users\User\Desktop\FYP`. There is no need to clone it again on that machine.

Do not assume Antigravity automatically loads Codex instructions. Explicitly tell the agent to read root `AGENTS.md` and this handover. Ask it to inspect current state before editing. Suggested first prompt:

> Continue my individual organoid-analysis FYP in this repository. First read AGENTS.md, docs/PROJECT_HANDOVER.md, docs/WORKING_GUIDE.md, docs/PROJECT_STATUS.md, docs/ROADMAP.md and data/DATA_NOTES.md. Inspect Git status and the relevant source code. Summarise the current state and unresolved decisions. My task for this session is: [describe the specific task]. Preserve original data and unrelated changes. Distinguish training results from independent validation, and update the relevant progress record when work is complete.

If you want changes published, append: "After checks pass and reviewing the staged files, commit and push these changes to GitHub." If you only want planning, say so explicitly.

On another machine, clone the public repo, install dependencies, configure Fiji and restore authorised data using [the setup guide](SETUP.md). Local environments, Fiji and ignored data do not arrive through Git.

## 10. Next useful milestone

Start with scope confirmation and trusted references. Then expand annotations, define group-separated splits, implement split-aware training/evaluation, compare baselines and evaluate all eight measurement errors. The existing training script automatically fits all available paired masks; merely adding images does not create a valid evaluation split.

Keep a clear chain from research question to data version, code revision, experiment, results and conclusion. Update this handover after meaningful changes to scope, architecture, dataset, selected checkpoint or verified results.
