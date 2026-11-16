"""Restore only manifest-listed pilot files from an original FYP folder."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = args.source.resolve() / "annotation_pilot_20"
    target = repo / "data" / "pilot"
    manifest = json.loads((repo / "data/metadata/pilot_manifest.json").read_text())
    for item in manifest:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        src = source / relative
        dst = target / relative
        if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source missing or changed: {src}")
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Refusing to replace different destination: {dst}")
    for item in manifest:
        dst = target / item["path"]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / item["path"], dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == item["sha256"]
    print(f"Verified {len(manifest)} pilot files. No existing differing files overwritten.")


if __name__ == "__main__":
    main()
