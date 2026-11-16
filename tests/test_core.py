"""Fast unit tests that need no lab images and no Fiji.

Run from the repository root:  python -m pytest

They protect the parts that must not change by accident: the model layout the
checkpoint depends on, the preprocessing maths and input validation. They say
nothing about segmentation accuracy.
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from model import TinyUNet
from organoid_core import load_model, predict_probability
from preprocessing import letterbox, load_tiff, normalize


def tiff_bytes(array, **save_options):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="TIFF", **save_options)
    return io.BytesIO(buffer.getvalue())


def gradient(height=32, width=48):
    return (np.arange(height * width) % 251).astype(np.uint8).reshape(height, width)


# --- model -----------------------------------------------------------------


def test_model_has_expected_parameter_count():
    assert sum(p.numel() for p in TinyUNet().parameters()) == 29321


def test_model_output_matches_input_size():
    output = TinyUNet()(torch.zeros(2, 1, 64, 32))
    assert output.shape == (2, 1, 64, 32)


def test_selected_checkpoint_loads():
    model, checkpoint = load_model()
    assert checkpoint["preprocessing"] == "letterbox_percentile_v1"
    assert checkpoint["size"] % 4 == 0
    assert len(checkpoint["_model_sha256"]) == 64
    assert not model.training


def test_prediction_is_returned_at_original_size():
    model, checkpoint = load_model()
    image = normalize(gradient(60, 100).astype(np.float32))
    probability = predict_probability(image, model, checkpoint["size"])
    assert probability.shape == (60, 100)
    assert 0 <= probability.min() and probability.max() <= 1


# --- preprocessing -----------------------------------------------------------


def test_normalize_gives_unit_range():
    result = normalize(gradient().astype(np.float32) * 300)
    assert result.dtype == np.float32
    assert result.min() == 0 and result.max() == 1


def test_letterbox_keeps_aspect_ratio_and_centres():
    square, (top, left, height, width) = letterbox(np.ones((100, 200), np.float32), 64)
    assert square.shape == (64, 64)
    assert (height, width) == (32, 64)
    assert (top, left) == (16, 0)
    assert square[top : top + height].all() and not square[:top].any()


def test_letterbox_mask_stays_binary():
    mask = np.zeros((90, 130), np.float32)
    mask[20:70, 30:100] = 1
    square, _ = letterbox(mask, 64, mask=True)
    assert set(np.unique(square)) == {0, 1}


# --- TIFF validation -----------------------------------------------------------


def test_load_tiff_grayscale():
    image, scale = load_tiff(tiff_bytes(gradient()))
    assert image.shape == (32, 48) and image.dtype == np.float32
    assert scale is None


def test_load_tiff_converts_rgb_to_grayscale():
    rgb = np.stack([gradient()] * 3, axis=-1)
    image, _ = load_tiff(tiff_bytes(rgb))
    assert image.shape == (32, 48)


def test_load_tiff_rejects_blank_image():
    with pytest.raises(ValueError, match="no intensity variation"):
        load_tiff(tiff_bytes(np.zeros((32, 32), np.uint8)))


def test_load_tiff_rejects_stack():
    page = Image.fromarray(gradient())
    with pytest.raises(ValueError, match="single-plane"):
        load_tiff(tiff_bytes(gradient(), save_all=True, append_images=[page]))


def test_load_tiff_rejects_png():
    buffer = io.BytesIO()
    Image.fromarray(gradient()).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="TIFF"):
        load_tiff(io.BytesIO(buffer.getvalue()))


OME = (
    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image><Pixels '
    'PhysicalSizeX="{x}" PhysicalSizeY="{y}" PhysicalSizeXUnit="{unit}" '
    'PhysicalSizeYUnit="{unit}"/></Image></OME>'
)


def test_load_tiff_reads_ome_scale_in_microns():
    source = tiff_bytes(gradient(), description=OME.format(x=619.2, y=619.2, unit="nm"))
    _, scale = load_tiff(source)
    assert scale["x"] == pytest.approx(0.6192)
    assert "requires_manual_entry" not in scale


def test_load_tiff_flags_implausible_ome_scale():
    source = tiff_bytes(gradient(), description=OME.format(x=1.268, y=0.951, unit="mm"))
    _, scale = load_tiff(source)
    assert scale["requires_manual_entry"] is True
