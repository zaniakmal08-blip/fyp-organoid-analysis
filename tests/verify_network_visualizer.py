"""Verify displayed arithmetic against actual PyTorch tensors for all operations.

Needs the pilot images and the selected checkpoint (no Fiji).
Run from the repository root:  python tests/verify_network_visualizer.py
Writes results/local/network_visualizer_checks/arithmetic_verification.json
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import network_visualizer as v
from organoid_core import load_model
from project_paths import RESULTS
from train_three import NAMES, ROOT

v.MODEL_INSTANCE, v.CHECKPOINT = load_model()
info = v.summary()
assert info["parameters"] == 29321 and info["learnable_layers"] == 13
record = v.record((ROOT / "images" / (NAMES[0] + ".tif")).read_bytes(), NAMES[0] + ".tif")
assert len(record["operations"]) == 25
results = []
for op in record["operations"]:
    c, h, w = op["output_shape"]
    for channel, y, x in [(0, 0, 0), (c - 1, h // 2, w // 2), (0, h - 1, w - 1)]:
        r = v.calculate(record["run"], op["id"], channel, y, x, op["input_shape"][0] - 1)
        assert np.isclose(r["actual_output"], r["recalculated"], atol=2e-5, rtol=2e-5), (
            op["id"],
            r,
        )
        results.append(
            {
                "op": op["id"],
                "channel": channel,
                "y": y,
                "x": x,
                "absolute_difference": r["difference"],
            }
        )
weights = {n: p.detach().numpy() for n, p in v.MODEL_INSTANCE.named_parameters()}
buffer = io.BytesIO()
np.savez_compressed(buffer, **weights)
buffer.seek(0)
with np.load(buffer) as loaded:
    assert sum(loaded[k].size for k in loaded.files) == info["parameters"]
report = {
    "model": info,
    "checks": results,
    "calculation_checks": len(results),
    "max_absolute_difference": max(r["absolute_difference"] for r in results),
    "weights_export_value_count": info["parameters"],
    "result": "passed",
}
out = RESULTS / "network_visualizer_checks"
out.mkdir(exist_ok=True)
(out / "arithmetic_verification.json").write_text(json.dumps(report, indent=2))
print(
    "PASS:",
    len(results),
    "pixel calculations across",
    len(record["operations"]),
    "operations; maximum absolute difference:",
    report["max_absolute_difference"],
)
