"""Bridge to Fiji/ImageJ, which performs the actual morphometric measurements.

Python does not compute the eight measurements itself. It starts a headless
Java process running ``imagej_helper/MeasureOrganoid.java`` against the
``ij-*.jar`` shipped with Fiji, so values match what a researcher would obtain
manually in Fiji with Analyze > Measure.

Currently Windows-only: it looks for Fiji's bundled ``java.exe``.
"""

import json
import subprocess
from pathlib import Path

from project_paths import FIJI

HELPER = Path(__file__).resolve().parent / "imagej_helper" / "MeasureOrganoid.java"
TIMEOUT_SECONDS = 90

# The eight measurements, in export order. Keys match MeasureOrganoid.java.
FIELDS = [
    "Area",
    "Perimeter",
    "Circularity",
    "AspectRatio",
    "Roundness",
    "Solidity",
    "Feret",
    "MinFeret",
]
UNITS = {
    "Area": "um^2",
    "Perimeter": "um",
    "Feret": "um",
    "MinFeret": "um",
    "Circularity": "dimensionless",
    "AspectRatio": "dimensionless",
    "Roundness": "dimensionless",
    "Solidity": "dimensionless",
}


def imagej_paths():
    """Return ``(ij_jar, java_executable)`` from the Fiji installation.

    Raises ``RuntimeError`` if Fiji (with bundled Java) is not at ``FIJI_HOME``.
    """
    jars = sorted((FIJI / "jars").glob("ij-*.jar"))
    javas = list((FIJI / "java").rglob("bin/java.exe"))
    if not jars or not javas:
        raise RuntimeError(
            "Fiji installation not found. Set the FIJI_HOME environment variable to your "
            "Fiji directory (including bundled Java)."
        )
    return jars[-1], javas[0]


def measure_imagej(mask_path, output_dir, scale_x, scale_y, roi_input=None):
    """Measure one region with ImageJ and return the eight values as a dict.

    ``mask_path``   binary PNG (0/255) at original image size.
    ``output_dir``  receives ``measurements.json`` and ``predicted.roi``.
    ``scale_x/y``   calibration in um/pixel.
    ``roi_input``   optional region definition; without it the region is the
                    thresholded mask. A ``.csv`` of x,y polygon vertices or a
                    Fiji ``.roi`` file is used instead when given.
    """
    jar, java = imagej_paths()
    command = [
        str(java),
        "-Djava.awt.headless=true",
        "-cp",
        str(jar),
        str(HELPER),
        str(Path(mask_path).resolve()),
        str(Path(output_dir).resolve()),
        str(scale_x),
        str(scale_y),
    ]
    if roi_input:
        command.append(str(Path(roi_input).resolve()))
    result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    if result.returncode:
        raise RuntimeError("ImageJ measurement failed: " + result.stderr[-2000:])
    return json.loads((Path(output_dir) / "measurements.json").read_text())
