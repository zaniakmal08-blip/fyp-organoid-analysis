"""Central place for every folder and file location the project uses.

All paths are derived from the repository root, so the code works on any
computer without editing. Each location can be overridden with an environment
variable *before* starting Python:

=================  =============================================  ======================
Variable           Meaning                                        Default
=================  =============================================  ======================
``FYP_DATA_DIR``   Folder holding images/ masks/ rois/            ``data/pilot``
                   measurements/
``FYP_MODEL_PATH`` Selected model checkpoint                      ``models/tiny_unet.pt``
``FYP_RESULTS_DIR`` Root for generated (Git-ignored) outputs      ``results/local``
``FIJI_HOME``      Fiji installation including bundled Java       ``tools/Fiji``
=================  =============================================  ======================

Never hard-code an absolute workstation path elsewhere in the code; import the
constants below instead.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _location(variable: str, default: Path) -> Path:
    """Return the environment override for ``variable`` or the repository default."""
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


DATA = _location("FYP_DATA_DIR", REPO / "data" / "pilot")
MODEL = _location("FYP_MODEL_PATH", REPO / "models" / "tiny_unet.pt")
RESULTS = _location("FYP_RESULTS_DIR", REPO / "results" / "local")
FIJI = _location("FIJI_HOME", REPO / "tools" / "Fiji")

# Generated outputs always have somewhere to go.
RESULTS.mkdir(parents=True, exist_ok=True)
