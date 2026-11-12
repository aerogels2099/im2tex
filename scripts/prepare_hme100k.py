"""Write the common split layout for HME100K (photos of handwritten school math), from the parquet files
that scripts/download_data.sh fetches.

Train only: 3,000 of UniMER-Test's 6,332 handwritten images are HME100K test images, so the test split is never
used, and the few train images that also appear there are dropped (same size and nearly the same 16x16
thumbnail). Labels with non-ASCII characters (Chinese punctuation, about 0.5%) are rejected: English only.
Images are written as the parquet files store them (PNG). Needs UniMER-Test prepared first (prepare_unimer.py).
    python scripts/prepare_hme100k.py
"""
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from im2tex.normalize import normalize_or_none

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))
OUT = DATA / "hme100k" / "train"


def thumb(data: bytes):
    im = Image.open(io.BytesIO(data)).convert("L")
    return im.size, np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.int16).ravel()


hwe = DATA / "unimer/test/hwe"
test_thumbs = {}
for line in (hwe / "refs.jsonl").open(encoding="utf-8"):
    size, t = thumb((hwe / json.loads(line)["image"]).read_bytes())
    test_thumbs.setdefault(size, []).append(t)

(OUT / "images").mkdir(parents=True, exist_ok=True)
rows, dropped = [], 0
for f in sorted((DATA / "hme100k").glob("train-*.parquet")):
    t = pq.read_table(f)
    for name, ref, image in zip(t.column("image_path").to_pylist(), t.column("label").to_pylist(),
                                (x["bytes"] for x in t.column("image").to_pylist())):
        size, th = thumb(image)
        if any(np.abs(other - th).mean() < 3 for other in test_thumbs.get(size, [])):
            dropped += 1
            continue
        stem = name.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]  # ./HME100K\train_images\train_0.jpg
        path = OUT / "images" / f"{stem}.png"
        if not path.exists():
            path.write_bytes(image)
        norm = None if any(ord(c) > 127 for c in ref) else normalize_or_none(ref)
        rows.append({"id": stem, "image": f"images/{stem}.png", "ref": ref, "norm": norm})
with (OUT / "refs.jsonl").open("w", encoding="utf-8") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
rejected = [r["ref"] for r in rows if r["norm"] is None]
print(f"train: {len(rows)} formulas ({dropped} dropped as UniMER-Test images), {len(rejected)} labels rejected "
      f"-> {OUT}")
for ref in rejected[:5]:
    print(f"    rejected: {ref}")
