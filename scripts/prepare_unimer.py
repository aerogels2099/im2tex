"""Unpack UniMER-Test into the common split layout: <split>/refs.jsonl ({"id", "ref"}) + <split>/images/.

UniMER names each image by the line number of its label in <split>.txt.
    python scripts/prepare_unimer.py
"""
import json
import os
import zipfile
from pathlib import Path

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data")) / "unimer"

if not (DATA / "UniMER-Test").exists():
    with zipfile.ZipFile(DATA / "UniMER-Test.zip") as z:
        z.extractall(DATA)

for split in ("spe", "cpe", "sce", "hwe"):
    src = DATA / "UniMER-Test" / split
    labels = (DATA / "UniMER-Test" / f"{split}.txt").read_text(encoding="utf-8").split("\n")
    out = DATA / "test" / split
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "images").exists():
        (out / "images").symlink_to(src)
    with (out / "refs.jsonl").open("w", encoding="utf-8") as f:
        for png in sorted(src.glob("*.png")):
            f.write(json.dumps({"id": png.stem, "ref": labels[int(png.stem)]}) + "\n")
    print(f"{split}: {len(list(src.glob('*.png')))} -> {out}")
