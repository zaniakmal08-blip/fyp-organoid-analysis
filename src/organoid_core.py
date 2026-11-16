"""The analysis pipeline: TIFF -> segmentation -> polygon -> ImageJ measurements.

This is the module to read first. ``analyse`` runs every step for one image:

1. validate and load the TIFF                      (``preprocessing.load_tiff``)
2. normalise and letterbox to the model input size (``preprocessing``)
3. predict a probability map with TinyUNet         (``model.TinyUNet``)
4. map it back to original coordinates, keep the largest region, fill holes
5. simplify the outline to a polygon
6. measure the polygon with Fiji/ImageJ            (``imagej_bridge``)
7. write the mask, overlay, CSV, JSON provenance and a ZIP

Names from the helper modules are re-exported here so existing imports such as
``from organoid_core import load_tiff`` keep working.
"""

import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from imagej_bridge import FIELDS, UNITS, imagej_paths, measure_imagej
from model import TinyUNet
from preprocessing import letterbox, load_tiff, normalize
from project_paths import FIJI, MODEL, RESULTS

__all__ = [
    "FIELDS",
    "FIJI",
    "HERE",
    "MODEL",
    "RESULTS",
    "TinyUNet",
    "analyse",
    "imagej_paths",
    "letterbox",
    "load_model",
    "load_tiff",
    "measure_imagej",
    "normalize",
]

HERE = Path(__file__).resolve().parent
PREPROCESSING = "letterbox_percentile_v1"
THRESHOLD = 0.5  # probability at or above this is organoid
MIN_AREA_PIXELS = 16
MAX_SCALE = 1000  # um/pixel sanity limit
POLYGON_TOLERANCE = 1.0  # outline simplification, in original-image pixels
CSV_COLUMNS = [
    "Image",
    "Area_um2",
    "Perimeter_um",
    "Circularity",
    "AspectRatio",
    "Roundness",
    "Solidity",
    "Feret_um",
    "MinFeret_um",
    "ScaleX_um_per_pixel",
    "ScaleY_um_per_pixel",
    "ReviewStatus",
]

torch.set_num_threads(2)


def load_model(model_path=MODEL):
    """Load the checkpoint and return ``(model, checkpoint)`` ready for inference.

    The checkpoint dict gains two entries: ``_model_sha256`` and ``_model_path``,
    which are written into every result for provenance.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError("Train the prototype model first: run train_prototype.py.")
    checkpoint_bytes = model_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    checkpoint["_model_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint.get("preprocessing") != PREPROCESSING:
        raise RuntimeError("This app requires the aspect-preserving prototype checkpoint.")
    checkpoint["_model_path"] = str(model_path.resolve())
    model = TinyUNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_probability(image, model, size):
    """Run the model on a normalised image; return probabilities at original size."""
    boxed, (top, left, height, width) = letterbox(image, size)
    with torch.no_grad():
        probability = model(torch.from_numpy(boxed[None, None])).sigmoid().numpy()[0, 0]
    return cv2.resize(
        probability[top : top + height, left : left + width],
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def analyse(data, filename, output_dir, model, checkpoint, scale=None):
    """Segment and measure one TIFF, writing all exports into ``output_dir``.

    ``data``        raw bytes of the uploaded TIFF.
    ``filename``    original name, recorded in the outputs.
    ``output_dir``  new folder for this run (must not exist yet).
    ``scale``       operator-confirmed calibration: ``{'x', 'y', 'source'}`` in um/pixel.

    Returns the result dict that is also saved as ``result.json``.
    Raises ``ValueError`` for problems the user can fix (bad scale, empty prediction).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    image, metadata_scale = load_tiff(io.BytesIO(data))
    if scale is None:
        raise ValueError("Confirm the image scale before analysis.")
    sx, sy = float(scale["x"]), float(scale["y"])
    if not all(math.isfinite(v) and 0 < v <= MAX_SCALE for v in (sx, sy)):
        raise ValueError("Scale must be positive, finite and no greater than 1000 um/pixel.")
    (output_dir / "input.tif").write_bytes(data)

    # --- Segmentation -------------------------------------------------------
    gray = normalize(image)
    probability = predict_probability(gray, model, checkpoint["size"])
    raw = np.uint8(probability >= THRESHOLD) * 255
    cv2.imwrite(str(output_dir / "raw_mask.png"), raw)
    np.save(output_dir / "probability.npy", probability)

    # --- Largest region, holes filled --------------------------------------
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Model predicted an empty region; no measurements produced.")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_AREA_PIXELS:
        raise ValueError("Predicted object is too small to measure.")
    mask = np.zeros_like(raw)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    # Hole filling makes the selected region represent the whole outer organoid.
    filled_pixels = int(np.count_nonzero((mask > 0) & (raw == 0)))

    # --- Outline -> polygon -------------------------------------------------
    # Fixed pixel-scale raster-to-polygon conversion, not a fitted circle/ellipse.
    # Old checkpoints retain their original measurement representation for comparison.
    roi_input = None
    representation = checkpoint.get("roi_representation", "threshold_mask")
    if representation == "polygon_1px":
        contour = cv2.approxPolyDP(contour, POLYGON_TOLERANCE, True)
        if len(contour) < 3:
            raise ValueError("Predicted polygon has fewer than three vertices.")
        roi_input = output_dir / "roi_vertices.csv"
        with roi_input.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            # +0.5 places each vertex at its pixel centre.
            writer.writerows((float(p[0]) + 0.5, float(p[1]) + 0.5) for p in contour[:, 0])
        mask[:] = 0
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    cv2.imwrite(str(output_dir / "mask.png"), mask)

    # --- Previews -----------------------------------------------------------
    preview = np.uint8(gray * 255)
    overlay = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), max(2, round(max(image.shape) / 700)))
    Image.fromarray(overlay).save(output_dir / "overlay.png")
    Image.fromarray(preview).save(output_dir / "input_preview.png")

    # --- Measurement --------------------------------------------------------
    values = measure_imagej(output_dir / "mask.png", output_dir, sx, sy, roi_input)

    # --- Warnings shown to the operator --------------------------------------
    warnings = [
        f"{len(checkpoint['names'])}-image demonstration model: inspect the outline before "
        "using these measurements.",
        "Mask restored to original coordinates; boundary detail is limited by model input "
        "resolution.",
    ]
    if len(contours) > 1:
        warnings.append(
            f"{len(contours) - 1} smaller external regions discarded; only the largest object "
            "is measured."
        )
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        warnings.append("Predicted object touches the image edge; it may be incomplete.")
    if filled_pixels:
        warnings.append(
            f"{filled_pixels} internal background pixels filled inside the selected outer boundary."
        )

    # --- Result + provenance -------------------------------------------------
    if representation == "polygon_1px":
        postprocessing = (
            "largest external region; filled interior; polygon simplification tolerance "
            "1 original pixel; pixel-centre coordinates"
        )
    else:
        postprocessing = "largest external region, filled interior, no contour smoothing"
    result = {
        "filename": filename,
        "measurements": values,
        "scale_um_per_pixel": {"x": sx, "y": sy},
        "scale_source": scale["source"],
        "detected_metadata_scale": metadata_scale,
        "width": image.shape[1],
        "height": image.shape[0],
        "warnings": warnings,
        "model": str(Path(checkpoint.get("_model_path", MODEL))),
        "training_images": checkpoint["names"],
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "model_sha256": checkpoint["_model_sha256"],
        "measurement_engine": "ImageJ 1.54p; calibrated original image coordinates",
        "roi_representation": representation,
        "postprocessing": postprocessing,
        "model_input_size": checkpoint["size"],
        "review_status": "unreviewed",
        "units": UNITS,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    with (output_dir / "measurements.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerow([filename] + [values[k] for k in FIELDS] + [sx, sy, "unreviewed"])
    with zipfile.ZipFile(output_dir / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.iterdir():
            if path.is_file() and path.name != "results.zip":
                archive.write(path, path.name)
    return result
