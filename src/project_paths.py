"""Portable locations; overrides are optional environment variables."""
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('FYP_DATA_DIR', str(REPO / 'data' / 'pilot'))).expanduser().resolve()
MODEL = Path(os.environ.get('FYP_MODEL_PATH', str(REPO / 'models' / 'tiny_unet.pt'))).expanduser().resolve()
RESULTS = Path(os.environ.get('FYP_RESULTS_DIR', str(REPO / 'results' / 'local'))).expanduser().resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
FIJI = Path(os.environ.get('FIJI_HOME', str(REPO / 'tools' / 'Fiji'))).expanduser().resolve()
