# Six-image update: input checks and measurement caveats

All six image/mask pairs are 2048 x 1536. Masks are binary 0/255 and agree exactly with their saved Fiji ROI rasters. This establishes file/selection consistency, not expert biological validation.

The added training pairs are A04 H2O2 50uM day 5, A05 H2O2 100uM day 6, and B07 ROT 0nM day 1. Original TIFFs, ROIs, masks and manual measurement CSVs were preserved.

## Manual measurement issues

- A04 CSV area is 1708630, exactly its foreground pixel count. B07 CSV area is 481158, also exactly its foreground pixel count. Their recorded lengths/areas are consistent with pixel units rather than the calibrated units of the earlier three CSVs. Do not combine these raw size values as if all six CSVs share physical units.
- A05 CSV is labelled `Mask`. Its area is 3145728 = 2048 x 1536 (the entire canvas), perimeter 7168 = 2 x (2048+1536), and aspect ratio 1.333333 = 2048/1536. It is not a measurement of the saved organoid ROI. The mask and ROI themselves passed the alignment check and remain usable for segmentation training.
- The model trains on images and masks; numerical CSV measurements are not used as training targets.

For old/new numerical comparisons, saved ROIs are remeasured by ImageJ and converted using the same provisional recorded scale (0.61922571983322461 um/pixel) for every image. These derived references are stored separately, and are not presented as corrected RA measurements. Verify calibration with the lab before reporting physical units.

To repair the original reference workflow in Fiji: open the ORIGINAL TIFF, apply its saved organoid ROI, verify spatial calibration, ensure Limit to threshold is off, Measure, and save the eight values. Do not measure an unselected entire mask canvas.

## Scope

All six annotations are included in fitting, with no held-out test set. Training-set overlap and measurement agreement cannot be reported as unseen-image accuracy. The app and calculation inspector must load the new checkpoint before they use its six-image training set.
