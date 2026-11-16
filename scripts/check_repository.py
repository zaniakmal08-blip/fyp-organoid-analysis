"""Dependency-free structure, manifest, checkpoint and Python syntax checks."""

import ast
import hashlib
import json
import re
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
for folder in ("src", "scripts", "tests"):
    for path in (repo / folder).rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in repo.rglob("*.md"):
    if ".git" in path.parts:
        continue
    for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" not in link and not link.startswith("#"):
            assert (path.parent / link.split("#")[0]).exists(), (path, link)
manifest = json.loads((repo / "data/metadata/pilot_manifest.json").read_text())
assert len({item["path"] for item in manifest}) == len(manifest)
for item in manifest:
    assert len(item["sha256"]) == 64 and item["bytes"] > 0
    local = repo / "data/pilot" / item["path"]
    if local.exists():
        assert hashlib.sha256(local.read_bytes()).hexdigest() == item["sha256"], local
model = repo / "models/tiny_unet.pt"
assert hashlib.sha256(model.read_bytes()).hexdigest() in (repo / "models/README.md").read_text()
print("PASS: Python syntax, documentation links, manifest and selected model checksum.")
