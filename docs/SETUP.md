# Setup and usage

Run commands from the repository root. The full prototype currently targets Windows because the ImageJ bridge finds Fiji's bundled `java.exe`. Use a Python version supported by your PyTorch installation; the local verification environment is recorded in `requirements-tested.txt`. That file is an environment record, not a cross-platform lockfile.

1. Create and activate a virtual environment and install `requirements.txt` as shown in the main README.
2. Install Fiji with bundled Java from https://fiji.sc/ and set `$env:FIJI_HOME = 'C:\path\to\Fiji'` in the current PowerShell session.
3. Restore pilot files with `python scripts/import_local_data.py --source 'C:\path\to\original\FYP'`. This copies the 20 selected images and their available annotations and verifies them against the manifest. Existing differing files are never overwritten.
4. Start `python src/prototype_app.py`. Open http://127.0.0.1:8765. Upload a single-plane TIFF, confirm calibration, inspect the predicted boundary and export results.
5. Start `python src/network_visualizer.py` for the CNN inspector at http://127.0.0.1:8770.

Optional path overrides: `FYP_DATA_DIR` (folder with images/masks/rois/measurements), `FYP_MODEL_PATH`, `FYP_RESULTS_DIR`, and `FIJI_HOME`. Set environment variables before starting Python. Defaults use this repository's `data/pilot`, `models/tiny_unet.pt` and `results/local`.

## Verification

```powershell
python scripts/check_repository.py
python tests/verify_network_visualizer.py
python tests/verify_prototype.py
```

The last two need pilot images and the checkpoint; the measurement check also needs Fiji. They check software arithmetic, exports, calibration scaling and invalid inputs, not generalisation or biological validity. The measurement check uses the original three reference CSVs, not all six.

## Training

```powershell
python src/train_prototype.py --epochs 300 --learning-rate 0.001 --dice-weight 3
```

This fits all paired masks and warm-starts from the included model by default. It writes a candidate under `results/local/model_candidate`; it does not replace the selected model. This command continues training and is not an exact replay of the historic run. For a new random initialisation, pass a non-existing `--warm-start` path. Record the starting checkpoint, parameters and data hashes in the experiment log. Evaluate candidates before manually promoting one into `models/`.

`src/train_three.py` and `src/predict_image.py` retain the earlier square-resize baseline. Its checkpoints are not compatible with the measurement app's aspect-preserving preprocessing; use only for baseline comparison.
