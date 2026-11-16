# Setup and usage

Run every command from the repository root. The full workflow currently targets **Windows**, because the ImageJ bridge looks for Fiji's bundled `java.exe`.

## 1. Requirements

| Need | Why | Notes |
|---|---|---|
| Python 3.10 or newer | Runs everything | Verified locally on 3.14 (see `requirements-tested.txt`); CI uses 3.12 |
| [Fiji](https://fiji.sc/) with bundled Java | Computes the measurements | Not needed for the CNN inspector, training or unit tests |
| Pilot data | Examples, training, integration checks | Not in Git; restored separately (step 4) |

## 2. Install

```powershell
git clone https://github.com/zaniakmal08-blip/fyp-organoid-analysis.git
cd fyp-organoid-analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For development (tests, linter, formatter) install `requirements-dev.txt` instead.

Check that `python` is the environment you expect, with PyTorch available:

```powershell
python -c "import sys, torch; print(sys.executable, torch.__version__)"
```

## 3. Point the project at Fiji

For the current terminal only:

```powershell
$env:FIJI_HOME = 'C:\path\to\Fiji'
```

To set it permanently for your Windows account (then open a new terminal):

```powershell
setx FIJI_HOME "C:\path\to\Fiji"
```

The folder must contain `jars\ij-*.jar` and a `java\` folder.

## 4. Restore the pilot data

```powershell
python scripts/import_local_data.py --source 'C:\path\to\original\FYP'
```

This copies the 20 selected images and the available masks, ROIs and measurement CSVs from `<source>\annotation_pilot_20` into `data/pilot/`, checking every file against `data/metadata/pilot_manifest.json`. A file that differs from the manifest is never overwritten.

Expected result:

```text
data/pilot/
├── images/        <name>.tif
├── masks/         <name>.png   (0 = background, 255 = organoid)
├── rois/          <name>.roi   (Fiji ROI)
└── measurements/  <name>.csv   (manual Fiji measurements)
```

## 5. Run

| Tool | Command | Opens |
|---|---|---|
| Measurement app | `python src/prototype_app.py` | <http://127.0.0.1:8765> |
| CNN inspector | `python src/network_visualizer.py` | <http://127.0.0.1:8770> |

Both accept `--port N` and `--no-browser`. Stop a server with Ctrl+C. In VS Code or Antigravity, press **F5** and choose a configuration from `.vscode/launch.json`.

In the measurement app: upload a single-plane TIFF, confirm the µm/pixel calibration, inspect the predicted outline, then export. Outputs are saved under `results/local/predictions/`.

## 6. Configuration

Set these environment variables before starting Python to override the defaults in `src/project_paths.py`:

| Variable | Purpose | Default |
|---|---|---|
| `FIJI_HOME` | Fiji installation including Java | `tools/Fiji` |
| `FYP_DATA_DIR` | Folder containing `images/ masks/ rois/ measurements/` | `data/pilot` |
| `FYP_MODEL_PATH` | Checkpoint the apps load | `models/tiny_unet.pt` |
| `FYP_RESULTS_DIR` | Root for generated output | `results/local` |

## 7. Checks

| Command | Needs | Checks |
|---|---|---|
| `python -m pytest` | Dev requirements | Model layout, checkpoint loading, preprocessing, TIFF validation |
| `python scripts/check_repository.py` | Nothing extra | Python syntax, Markdown links, data manifest, checkpoint hash |
| `ruff check .` and `ruff format --check .` | Dev requirements | Code style |
| `python tests/verify_network_visualizer.py` | Pilot data | Inspector arithmetic against real PyTorch tensors |
| `python tests/verify_prototype.py` | Pilot data + Fiji | Manual-ROI agreement, exports, calibration scaling, invalid inputs |

The first three also run on GitHub Actions. All of these are software checks; none measures generalisation or biological validity. The Fiji check compares against the original three reference CSVs, not all six.

## 8. Training

```powershell
python src/train_prototype.py --epochs 300 --learning-rate 0.001 --dice-weight 3
```

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 200 | Training epochs |
| `--size` | 512 | Model input size (divisible by 4) |
| `--learning-rate` | 0.0004 | Initial Adam learning rate (cosine decay) |
| `--dice-weight` | 1.0 | Weight of the Dice term in the loss |
| `--warm-start` | `models/tiny_unet.pt` | Checkpoint to continue from |
| `--output` | `results/local/model_candidate` | Where the candidate is written |

This fits **all** paired masks and warm-starts from the selected model, so it continues training rather than replaying the historic run. For a fresh random initialisation pass a non-existing `--warm-start` path. The selected model in `models/` is never replaced automatically: evaluate the candidate, then promote it deliberately and update `models/README.md` and `experiments/log.csv`.

`src/train_three.py` and `src/predict_image.py` are the earlier square-resize baseline. Their checkpoints are not compatible with the measurement app; use them only for baseline comparison.

## 9. Troubleshooting

| Message | Cause and fix |
|---|---|
| `Fiji installation not found` | `FIJI_HOME` is unset or wrong in this terminal. Set it (step 3) and restart the app. |
| `ModuleNotFoundError: No module named 'torch'` | The wrong Python is active. Activate `.venv` or call the intended interpreter explicitly. |
| `Cannot start server ... Try --port 8766` | The port is in use, usually by an earlier run. Stop it or pass `--port`. |
| Example images fail to load | Pilot data has not been restored (step 4). Uploading your own TIFF still works. |
| `This prototype accepts single-plane TIFFs` | The file is a stack. Export one plane from Fiji first. |
| `Source missing or changed` during import | The source folder does not match the recorded snapshot. Do not edit the manifest to force it; check which copy is authoritative. |
| `Activate.ps1 cannot be loaded` | PowerShell script policy. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
