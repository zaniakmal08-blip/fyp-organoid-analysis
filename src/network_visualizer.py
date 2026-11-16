"""CNN inspector: see every layer's output and re-derive any pixel by hand.

Run ``python src/network_visualizer.py`` and open http://127.0.0.1:8770.
The page is ``network_visualizer.html``; this file is its HTTP server.

It runs one real forward pass of the selected checkpoint, records the input and
output of every layer with forward hooks, and can recompute any single output
value from the stored inputs and weights so the arithmetic can be checked
against PyTorch. It is read-only: nothing is trained or modified.

Endpoints
---------
GET  /                 the user interface
GET  /api/model        parameter counts and layer list          (``summary``)
GET  /api/example      run a forward pass on a pilot image      (``record``)
POST /api/upload       run a forward pass on an uploaded TIFF   (``record``)
GET  /api/calculate    step-by-step arithmetic for one pixel    (``calculate``)
GET  /api/output       logit -> sigmoid -> mask for one pixel
GET  /api/code         source of a pipeline file, for display
GET  /weights.npz      all weights as a NumPy archive
"""

import argparse
import base64
import io
import json
import threading
import traceback
import uuid
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from organoid_core import HERE, letterbox, load_model, load_tiff, normalize
from project_paths import DATA as ROOT

# Files the page may display in its "See the actual Python code" panel.
SOURCE_FILES = [
    "model.py",
    "preprocessing.py",
    "organoid_core.py",
    "imagej_bridge.py",
    "train_prototype.py",
    "network_visualizer.py",
    "train_three.py",
]
LOCK = threading.RLock()
RUN = None
MODEL_INSTANCE = None
CHECKPOINT = None


def png(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    gray = np.uint8(np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1) * 255)
    b = io.BytesIO()
    Image.fromarray(gray).save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def summary():
    modules = list(MODEL_INSTANCE.modules())
    weighted = [
        (n, m)
        for n, m in MODEL_INSTANCE.named_modules()
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    ]
    return {
        "parameters": sum(p.numel() for p in MODEL_INSTANCE.parameters()),
        "conv_layers": sum(isinstance(m, nn.Conv2d) for m in modules),
        "upsampling_layers": sum(isinstance(m, nn.ConvTranspose2d) for m in modules),
        "learnable_layers": len(weighted),
        "relu_layers": sum(isinstance(m, nn.ReLU) for m in modules),
        "size": CHECKPOINT["size"],
        "model_sha256": CHECKPOINT["_model_sha256"],
        "training_images": CHECKPOINT["names"],
        "examples": [n + ".tif" for n in CHECKPOINT["names"]],
        "layers": [
            {
                "name": n,
                "type": type(m).__name__,
                "weight_shape": list(m.weight.shape),
                "parameters": sum(p.numel() for p in m.parameters()),
            }
            for n, m in weighted
        ],
    }


def record(data, name):
    global RUN
    a, _ = load_tiff(io.BytesIO(data))
    original_shape = list(a.shape)
    box, transform = letterbox(normalize(a), CHECKPOINT["size"])
    operations = []
    handles = []
    counts = Counter()

    def hook(name, module):
        def capture(_module, args, output):
            counts[name] += 1
            key = name if name != "pool" else "pool@" + str(counts[name])
            # Arrays retain this exact inference tensor's storage. No training or mutation.
            inp = args[0].detach().numpy()[0]
            out = output.detach().numpy()[0]
            operations.append(
                {
                    "id": key,
                    "name": name,
                    "module": module,
                    "input": inp,
                    "output": out,
                    "type": type(module).__name__,
                }
            )

        return capture

    for n, m in MODEL_INSTANCE.named_modules():
        if n and not list(m.children()):
            handles.append(m.register_forward_hook(hook(n, m)))
    try:
        with torch.no_grad():
            output_tensor = MODEL_INSTANCE(torch.from_numpy(box[None, None]))
            probability = output_tensor.sigmoid().numpy()[0, 0]
    finally:
        for h in handles:
            h.remove()
    top, left, nh, nw = transform
    restored = cv2.resize(
        probability[top : top + nh, left : left + nw], (original_shape[1], original_shape[0])
    )
    preview = np.uint8(normalize(a) * 255)
    overlay = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
    contours = cv2.findContours(
        np.uint8(restored >= 0.5), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), max(2, round(max(a.shape) / 700)))
    im = Image.fromarray(overlay)
    im.thumbnail((800, 600))
    b = io.BytesIO()
    im.save(b, format="PNG")
    run_id = uuid.uuid4().hex
    RUN = {
        "id": run_id,
        "name": name,
        "operations": operations,
        "boxed_input": box,
        "probability": probability,
    }
    return {
        "run": run_id,
        "name": name,
        "original_shape": original_shape,
        "model_shape": list(box.shape),
        "transform": {
            "top_padding": top,
            "left_padding": left,
            "resized_height": nh,
            "resized_width": nw,
        },
        "input_preview": png(box),
        "probability_preview": png(probability),
        "overlay": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode(),
        "operations": [
            {
                "id": o["id"],
                "name": o["name"],
                "type": o["type"],
                "input_shape": list(o["input"].shape),
                "output_shape": list(o["output"].shape),
                "parameters": sum(p.numel() for p in o["module"].parameters()),
                "kernel": list(o["module"].kernel_size)
                if isinstance(o["module"], (nn.Conv2d, nn.ConvTranspose2d))
                else None,
            }
            for o in operations
        ],
    }


def get_op(run_id, op_id):
    if RUN is None or RUN["id"] != run_id:
        raise ValueError(
            "This pass has expired. Upload/load the image again; only the latest pass is kept in memory."
        )
    return next(o for o in RUN["operations"] if o["id"] == op_id)


def calculate(run_id, op_id, channel, y, x, input_channel=0):
    op = get_op(run_id, op_id)
    m = op["module"]
    inp = op["input"]
    out = op["output"]
    if not (0 <= channel < out.shape[0] and 0 <= y < out.shape[1] and 0 <= x < out.shape[2]):
        raise ValueError("Output pixel is outside this feature map.")
    observed = float(out[channel, y, x])
    result = {
        "operation": op_id,
        "type": op["type"],
        "channel": channel,
        "y": y,
        "x": x,
        "actual_output": observed,
        "feature": png(out[channel]),
        "feature_min": float(out[channel].min()),
        "feature_max": float(out[channel].max()),
        "feature_mean": float(out[channel].mean()),
        "input_channels": int(inp.shape[0]),
        "output_channels": int(out.shape[0]),
        "output_height": int(out.shape[1]),
        "output_width": int(out.shape[2]),
    }
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        weights = m.weight.detach().numpy()
        kh, kw = m.kernel_size
        sh, sw = m.stride
        ph, pw = m.padding
        dh, dw = m.dilation
        patches = np.zeros((inp.shape[0], kh, kw), np.float64)
        terms = np.zeros_like(patches)
        valid = np.zeros((kh, kw), bool)
        for ky in range(kh):
            for kx in range(kw):
                if isinstance(m, nn.ConvTranspose2d):
                    iy = y + ph - ky * dh
                    ix = x + pw - kx * dw
                    if iy % sh or ix % sw:
                        continue
                    iy //= sh
                    ix //= sw
                else:
                    iy = y * sh - ph + ky * dh
                    ix = x * sw - pw + kx * dw
                if 0 <= iy < inp.shape[1] and 0 <= ix < inp.shape[2]:
                    valid[ky, kx] = True
                    patches[:, ky, kx] = inp[:, iy, ix]
        selected_weights = (
            weights[:, channel] if isinstance(m, nn.ConvTranspose2d) else weights[channel]
        )
        terms = patches * selected_weights.astype(np.float64)
        sums = terms.sum(axis=(1, 2))
        bias = float(m.bias[channel].detach()) if m.bias is not None else 0.0
        computed = float(sums.sum() + bias)
        if not 0 <= input_channel < inp.shape[0]:
            raise ValueError("Input channel is outside this layer.")
        result.update(
            {
                "formula": "z = sum over ALL input channels and valid kernel positions (input × weight) + bias",
                "input_channel": input_channel,
                "patch": patches[input_channel].tolist(),
                "kernel_weights": selected_weights[input_channel].tolist(),
                "products": terms[input_channel].tolist(),
                "valid_positions": valid.tolist(),
                "channel_sums": sums.tolist(),
                "bias": bias,
                "recalculated": computed,
                "difference": abs(computed - observed),
                "weight_min": float(weights.min()),
                "weight_max": float(weights.max()),
                "weight_layout": "[input, output, kernel_y, kernel_x]"
                if isinstance(m, nn.ConvTranspose2d)
                else "[output, input, kernel_y, kernel_x]",
                "note": "This output is before ReLU. Select the following ReLU operation to inspect max(0,z). Float64 re-summing can differ slightly from the actual float32 convolution.",
            }
        )
        if op["name"] in ["d2.0", "d1.0"]:
            result["skip_connection"] = (
                "The first half of these input channels comes from upsampling; the second half comes from the encoder skip connection (torch.cat)."
            )
        if op["name"] == "out":
            result["note"] = (
                "This is the final logit. There is no following ReLU: sigmoid converts it to the organoid probability shown below. Float64 re-summing can differ slightly from actual float32 convolution."
            )
        if isinstance(m, nn.ConvTranspose2d):
            result["note"] += (
                " With stride-two transposed convolution, only stride-aligned kernel positions contribute to this output pixel; other products are zero."
            )
    elif isinstance(m, nn.ReLU):
        before = float(inp[channel, y, x])
        result.update(
            {
                "formula": "output = max(0, input)",
                "input_value": before,
                "recalculated": max(0, before),
                "difference": abs(max(0, before) - observed),
                "note": "ReLU has no trainable weights.",
            }
        )
    elif isinstance(m, nn.MaxPool2d):
        k = m.kernel_size
        s = m.stride or k
        patch = inp[channel, y * s : y * s + k, x * s : x * s + k]
        result.update(
            {
                "formula": "output = maximum of the 2 × 2 input patch",
                "patch": patch.tolist(),
                "recalculated": float(patch.max()),
                "difference": abs(float(patch.max()) - observed),
                "note": "Max pooling has no trainable weights. The same pool module is called twice in this U-Net.",
            }
        )
    return result


class Handler(BaseHTTPRequestHandler):
    def send(self, value, kind="application/json", status=200):
        if isinstance(value, dict):
            value = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(value)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(value)

    def do_GET(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        try:
            with LOCK:
                if parsed.path == "/":
                    return self.send(
                        (HERE / "network_visualizer.html").read_bytes(), "text/html; charset=utf-8"
                    )
                if parsed.path == "/api/model":
                    return self.send(summary())
                if parsed.path == "/api/calculate":
                    return self.send(
                        calculate(
                            q["run"][0],
                            q["op"][0],
                            int(q.get("channel", ["0"])[0]),
                            int(q["y"][0]),
                            int(q["x"][0]),
                            int(q.get("input", ["0"])[0]),
                        )
                    )
                if parsed.path == "/api/example":
                    name = q["name"][0]
                    if name not in [n + ".tif" for n in CHECKPOINT["names"]]:
                        raise ValueError("Unknown example")
                    return self.send(record((ROOT / "images" / name).read_bytes(), name))
                if parsed.path == "/api/code":
                    name = q["name"][0]
                    if name not in SOURCE_FILES:
                        raise ValueError("Unknown source file")
                    return self.send(
                        {"name": name, "code": (HERE / name).read_text(encoding="utf-8")}
                    )
                if parsed.path == "/weights.npz":
                    b = io.BytesIO()
                    np.savez_compressed(
                        b, **{n: p.detach().numpy() for n, p in MODEL_INSTANCE.named_parameters()}
                    )
                    return self.send(b.getvalue(), "application/octet-stream")
                if parsed.path == "/api/output":
                    if RUN is None or RUN["id"] != q["run"][0]:
                        raise ValueError("Pass expired")
                    y = int(q["y"][0])
                    x = int(q["x"][0])
                    logits = RUN["operations"][-1]["output"][0]
                    if not (0 <= y < logits.shape[0] and 0 <= x < logits.shape[1]):
                        raise ValueError("Pixel outside output")
                    z = float(logits[y, x])
                    p = float(RUN["probability"][y, x])
                    return self.send(
                        {
                            "logit": z,
                            "probability": p,
                            "mask_value": 255 if p >= 0.5 else 0,
                            "formula": "p = 1 / (1 + exp(-z)); foreground if p >= 0.5",
                        }
                    )
            self.send({"error": "Not found"}, status=404)
        except (ValueError, KeyError, StopIteration, IndexError) as e:
            self.send({"error": str(e)}, status=400)

    def do_POST(self):
        if urlparse(self.path).path != "/api/upload":
            return self.send({"error": "Not found"}, status=404)
        if self.headers.get("Origin") not in (None, "http://" + self.headers.get("Host", "")):
            return self.send({"error": "Cross-origin upload rejected"}, status=403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 50_000_000:
                raise ValueError("Upload a TIFF smaller than 50 MB.")
            data = self.rfile.read(length)
            name = parse_qs(urlparse(self.path).query).get("name", ["uploaded.tif"])[0]
            with LOCK:
                result = record(data, name)
            self.send(result)
        except (ValueError, OSError) as e:
            self.send({"error": str(e)}, status=400)
        except Exception:
            traceback.print_exc()
            self.send({"error": "Inference failed; see the terminal."}, status=500)


def main():
    global MODEL_INSTANCE, CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    MODEL_INSTANCE, CHECKPOINT = load_model()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print("TinyUNet calculation inspector:", url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
