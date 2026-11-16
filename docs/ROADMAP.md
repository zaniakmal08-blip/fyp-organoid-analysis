# Roadmap

- [x] Build local segmentation/measurement prototype.
- [x] Fit a six-image pilot and record its limitations.
- [x] Organise code, checkpoint, dataset manifests and research documentation.
- [ ] Confirm final scope and measurable objectives with supervisor.
- [ ] Agree annotation protocol and obtain reviewed Fiji ROIs.
- [ ] Verify calibration and repair reference measurements.
- [ ] Expand annotations and track reviewer status.
- [ ] Freeze splits by specimen/well (and batch when appropriate); keep related days together.
- [ ] Establish classical image-processing and learning baselines on identical splits.
- [ ] Evaluate segmentation and all eight measurement errors on unseen groups.
- [ ] Record failure cases, uncertainty and manual-review requirements.
- [ ] Assess biological prediction or imaging hardware extension only after scope/data confirmation.
- [ ] Prepare reproducible demonstration, report and presentation.

## Next up

Suggested order for the next working sessions. Items 1-3 need people, not code, and unblock everything else.

1. **Supervisor meeting: fix the scope.** Bring the six questions in section 7 of [PROJECT_HANDOVER.md](PROJECT_HANDOVER.md). Record the answers in `docs/meetings/`.
2. **Calibration.** Ask the lab for the true um/pixel at 10x (stage micrometer or acquisition software). Replace the provisional 0.619 value only with evidence.
3. **Annotation protocol.** Agree in writing what counts as the organoid boundary (include the halo or not), then have the six existing ROIs reviewed.
4. **Repair the three problem CSVs** (A04, A05, B07) by remeasuring the saved ROIs in Fiji as described in [DATA_NOTES.md](../data/DATA_NOTES.md). Keep the originals.
5. **Annotate the remaining 14 pilot images** so all 20 have masks and ROIs.
6. **Add a split to `src/train_prototype.py`**: read a split file listing train/validation/test image names (grouped by well so one organoid never appears on both sides), train on the training names only, and store the split in the checkpoint.
7. **Write `src/evaluate.py`**: for a checkpoint and a list of images, report Dice/IoU and the percentage error of all eight measurements per image, and save a CSV. This replaces the one-off comparison used for the six-image pilot.
8. **Classical baseline** (for example Otsu threshold + morphology) evaluated with the same script, so the U-Net has something to beat.
9. **Data augmentation** (flips, rotations, brightness) once the split exists; compare with and without it.
10. **Write up**: methods, results on unseen images, failure cases such as A04, limitations.

Engineering housekeeping that can be done at any time: choose a licence, add screenshots of the two apps to the README once image sharing is approved, and make `imagej_bridge.py` work on macOS/Linux if anyone else needs to run it.

Update checkboxes only with linked evidence. Use issues for specific tasks and dates agreed with the supervisor.
