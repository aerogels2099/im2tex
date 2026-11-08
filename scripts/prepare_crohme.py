"""Write the common split layout for CROHME (the offline version from the CoMER repo).

Splits: train, and the standard test sets 2014 / 2016 / 2019. Each gets refs.jsonl with
{"id", "image", "ref", "norm"} and images/ as PNGs. The source BMPs are white strokes on black;
they're inverted so every dataset is dark ink on white.
    python scripts/prepare_crohme.py
"""
import json
import os
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from im2tex.normalize import normalize_or_none

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data")) / "crohme"

if not (DATA / "data").exists():
    with zipfile.ZipFile(DATA / "data.zip") as z:
        z.extractall(DATA)

for split in ("train", "2014", "2016", "2019"):
    out = DATA / split
    if (out / "images").is_symlink():  # left by an earlier version of this script
        (out / "images").unlink()
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for line in (DATA / "data" / split / "caption.txt").read_text(encoding="utf-8").splitlines():
        name, ref = line.split("\t", 1)
        png = out / "images" / f"{name}.png"
        if not png.exists():
            ImageOps.invert(Image.open(DATA / "data" / split / "img" / f"{name}.bmp").convert("L")).save(png)
        rows.append({"id": name, "image": f"images/{name}.png", "ref": ref, "norm": normalize_or_none(ref)})
    with (out / "refs.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    rejected = [r["ref"] for r in rows if r["norm"] is None]
    print(f"{split}: {len(rows)} formulas, {len(rejected)} labels rejected -> {out}")
    for ref in rejected[:3]:
        print(f"    rejected: {ref}")
