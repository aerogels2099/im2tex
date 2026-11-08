"""Render MathWriting inks to PNGs and write the common split layout.

Each split gets refs.jsonl with {"id", "image", "ink", "ref", "norm"}. "ref" is MathWriting's own
normalizedLabel (it already drops distinctions handwriting can't show, like \\left( vs (), which then
goes through our normalizer. Training can re-render from "ink" with augmentation.
    python scripts/prepare_mathwriting.py [--root ~/im2tex-data/mathwriting/mathwriting-2024]
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from im2tex.ink import read_inkml, render_ink
from im2tex.normalize import normalize_or_none

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))


def convert(job):
    ink, png = job
    strokes, ann = read_inkml(ink)
    if not png.exists():
        Image.fromarray(render_ink(strokes)).save(png)
    ref = ann.get("normalizedLabel") or ann.get("label", "")
    return {"id": ink.stem, "image": f"images/{ink.stem}.png", "ink": str(ink), "ref": ref,
            "norm": normalize_or_none(ref)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA / "mathwriting" / "mathwriting-2024")
    args = ap.parse_args()
    for split in ("train", "valid", "test", "symbols", "synthetic"):
        inks = sorted((args.root / split).glob("*.inkml"))
        if not inks:
            continue
        out = args.root.parent / "prepared" / split
        (out / "images").mkdir(parents=True, exist_ok=True)
        with Pool() as pool:
            rows = pool.map(convert, [(i, out / "images" / f"{i.stem}.png") for i in inks], chunksize=64)
        with (out / "refs.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        rejected = sum(r["norm"] is None for r in rows)
        print(f"{split}: {len(rows)} inks, {rejected} labels rejected ({rejected / len(rows):.2%}) -> {out}")
