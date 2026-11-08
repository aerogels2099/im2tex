"""Export an im2latex-100k split to images/ + refs.jsonl ({"id", "image", "ref", "norm"}).

Baselines run in other venvs, so they only need PIL to read this.

    python scripts/export_im2latex.py --split test --n 1000
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from im2tex.normalize import normalize_or_none

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, help="first n examples only")
args = ap.parse_args()

ds = load_dataset("yuntian-deng/im2latex-100k", split=args.split)
if args.n:
    ds = ds.select(range(min(args.n, len(ds))))

out = DATA / "im2latex100k" / args.split
(out / "images").mkdir(parents=True, exist_ok=True)
with (out / "refs.jsonl").open("w", encoding="utf-8") as refs:
    for i, ex in enumerate(ds):
        ex["image"].save(out / "images" / f"{i:06d}.png")
        refs.write(json.dumps({"id": f"{i:06d}", "image": f"images/{i:06d}.png", "ref": ex["formula"],
                               "norm": normalize_or_none(ex["formula"])}) + "\n")
print(f"wrote {len(ds)} examples to {out}")
