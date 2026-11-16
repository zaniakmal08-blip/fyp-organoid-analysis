"""Local browser app: upload a TIFF, confirm the scale, get measurements.

Run ``python src/prototype_app.py`` and open http://127.0.0.1:8765.
The page itself is ``prototype_ui.html``; this file is its small HTTP server.

Endpoints
---------
GET  /                  the user interface
GET  /api/examples      names of the example (training) images
GET  /example/<name>    one example TIFF from the pilot data folder
GET  /files/<run>/<f>   an exported file from an earlier analysis
POST /api/inspect       preview an uploaded TIFF and read its metadata scale
POST /api/analyse       run the full pipeline (``organoid_core.analyse``)

The server listens on localhost only and handles one analysis at a time.
"""

import argparse
import base64
import io
import json
import mimetypes
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from organoid_core import HERE, analyse, load_model, load_tiff, normalize
from project_paths import DATA as ROOT
from project_paths import RESULTS

OUTPUT = RESULTS / "predictions"
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def send(self, data, content_type="application/json", status=200):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self.send((HERE / "prototype_ui.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/examples":
            return self.send(
                {
                    "examples": [n + ".tif" for n in self.server.checkpoint["names"]],
                    "training_count": len(self.server.checkpoint["names"]),
                    "model_sha256": self.server.checkpoint["_model_sha256"],
                }
            )
        if path.startswith("/example/"):
            name = path[len("/example/") :]
            if name not in [n + ".tif" for n in self.server.checkpoint["names"]]:
                return self.send({"error": "Unknown example"}, status=404)
            return self.send((ROOT / "images" / name).read_bytes(), "image/tiff")
        if path.startswith("/files/"):
            target = (OUTPUT / path[len("/files/") :]).resolve()
            if target.is_relative_to(OUTPUT.resolve()) and target.is_file():
                return self.send(
                    target.read_bytes(),
                    mimetypes.guess_type(target.name)[0] or "application/octet-stream",
                )
        self.send({"error": "Not found"}, status=404)

    def do_POST(self):
        # This server is local-only. Reject cross-origin browser requests.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.send({"error": "Cross-origin request rejected"}, status=403)
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path not in ("/api/inspect", "/api/analyse"):
            return self.send({"error": "Not found"}, status=404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 50_000_000:
                raise ValueError("Upload one TIFF smaller than 50 MB.")
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("Incomplete upload.")
            filename = Path(query.get("name", ["uploaded.tif"])[0].replace("\\", "/")).name
            if parsed.path == "/api/inspect":
                a, scale = load_tiff(io.BytesIO(data))
                preview = Image.fromarray(np.uint8(normalize(a) * 255))
                preview.thumbnail((700, 520))
                buffer = io.BytesIO()
                preview.save(buffer, format="PNG")
                return self.send(
                    {
                        "filename": filename,
                        "width": a.shape[1],
                        "height": a.shape[0],
                        "scale": scale,
                        "preview": "data:image/png;base64,"
                        + base64.b64encode(buffer.getvalue()).decode(),
                    }
                )
            sx = float(query["sx"][0])
            sy = float(query["sy"][0])
            scale = {"x": sx, "y": sy, "source": query.get("source", ["operator confirmed"])[0]}
            run = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            with LOCK:
                result = analyse(
                    data, filename, OUTPUT / run, self.server.model, self.server.checkpoint, scale
                )
            result["run"] = run
            result["saved_directory"] = str(OUTPUT / run)
            self.send(result)
        except (ValueError, KeyError, RuntimeError, OSError) as e:
            self.send({"error": str(e)}, status=400)
        except Exception:
            traceback.print_exc()
            self.send({"error": "Analysis failed. See the terminal for details."}, status=500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    model, checkpoint = load_model()
    OUTPUT.mkdir(exist_ok=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        raise SystemExit(f"Cannot start server: {e}. Try --port 8766.") from e
    server.model = model
    server.checkpoint = checkpoint
    url = f"http://127.0.0.1:{args.port}"
    print(f"Open {url}\nResults: {OUTPUT}\nPress Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
