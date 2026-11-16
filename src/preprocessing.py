"""Image loading and preprocessing shared by training, the app and the inspector.

The steps here define the ``letterbox_percentile_v1`` preprocessing recorded in
the checkpoint. Training and inference must use exactly the same steps, so
change them only together with a new preprocessing name and a retrained model.
"""

import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

MAX_PIXELS = 30_000_000
# Conversion factors from an OME unit to micrometres.
MICRONS_PER_UNIT = {"µm": 1, "μm": 1, "um": 1, "nm": 0.001, "mm": 1000, "m": 1e6}
# Pixel sizes above this (um/pixel) are implausible for these microscope images.
SUSPICIOUS_SCALE = 100


def load_tiff(source):
    """Read one single-plane TIFF as a float32 grayscale array.

    ``source`` is a path or a binary file object.

    Returns ``(image, scale)`` where ``scale`` is ``None`` or a dict with the
    pixel size in um/pixel read from OME metadata (``x``, ``y``, ``source`` and,
    if the value looks wrong, ``requires_manual_entry`` plus a ``warning``).
    The scale is only a suggestion: the operator must always confirm it.

    Raises ``ValueError`` with a user-readable message for unsupported input
    (not a TIFF, a multi-plane stack, too large, blank or non-finite).
    """
    with Image.open(source) as im:
        if im.format != "TIFF":
            raise ValueError("Please upload a TIFF image (.tif or .tiff).")
        if getattr(im, "n_frames", 1) != 1:
            raise ValueError(
                "This prototype accepts single-plane TIFFs. "
                "Export one agreed plane from a stack first."
            )
        if im.width * im.height > MAX_PIXELS:
            raise ValueError("Image is larger than the prototype limit (30 megapixels).")
        image = np.asarray(im).copy()
        description = str(im.tag_v2.get(270, ""))  # TIFF tag 270 = ImageDescription

    if image.ndim == 3 and image.shape[2] in (3, 4):
        image = image[..., :3].mean(axis=-1)  # RGB(A) -> grayscale
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite grayscale or RGB image.")
    image = image.astype(np.float32)
    if np.ptp(image) == 0:
        raise ValueError("Image has no intensity variation; cannot predict a useful boundary.")
    return image, _read_ome_scale(description)


def _read_ome_scale(description: str):
    """Extract um/pixel from an OME-XML image description, or return ``None``."""
    try:
        tree = ET.fromstring(description)
        pixels = next(p for p in tree.iter() if p.tag.split("}")[-1] == "Pixels")

        def microns(axis):
            value = float(pixels.attrib["PhysicalSize" + axis])
            unit = pixels.attrib.get("PhysicalSize" + axis + "Unit", "µm")
            return value * MICRONS_PER_UNIT[unit]

        sx, sy = microns("X"), microns("Y")
    except (ValueError, KeyError, StopIteration, ET.ParseError):
        return None
    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        return None
    scale = {"x": sx, "y": sy, "source": "OME PhysicalSizeX/Y; requires operator confirmation"}
    if max(sx, sy) > SUSPICIOUS_SCALE:
        scale["requires_manual_entry"] = True
        scale["warning"] = (
            "OME physical pixel sizes are unusually large for these microscope images and may "
            "represent field dimensions. Enter a verified pixel calibration; values were not "
            "converted automatically."
        )
    return scale


def normalize(image: np.ndarray) -> np.ndarray:
    """Stretch intensities to 0..1 using the 1st and 99th percentiles (robust to outliers)."""
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)


def letterbox(image: np.ndarray, size: int, mask: bool = False):
    """Resize to fit a ``size`` x ``size`` square without distortion, padding with zeros.

    Use ``mask=True`` for label masks so resizing does not invent in-between values.

    Returns ``(square, (top, left, height, width))``. The second value locates
    the resized image inside the square and is needed to map a prediction back
    to original image coordinates.
    """
    height, width = image.shape
    ratio = size / max(height, width)
    new_height, new_width = max(1, round(height * ratio)), max(1, round(width * ratio))
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    square = np.zeros((size, size), np.float32)
    square[top : top + new_height, left : left + new_width] = resized
    return square, (top, left, new_height, new_width)
