"""Unpack UniMER-1M and UniMER-Test into the common split layout.

Each split directory gets refs.jsonl with {"id", "image", "ref", "norm"}: the image path relative to the
split directory, the original label, and the normalized label (null if the normalizer rejects it).
UniMER names each image by the line number of its label in the split's .txt file.
    python scripts/prepare_unimer.py
"""
import json
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

from im2tex.normalize import normalize_or_none

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data")) / "unimer"
SPLITS = {  # output split -> (zip, image dir inside it, label file inside it)
    "test/spe": ("UniMER-Test", "spe", "spe.txt"),
    "test/cpe": ("UniMER-Test", "cpe", "cpe.txt"),
    "test/sce": ("UniMER-Test", "sce", "sce.txt"),
    "test/hwe": ("UniMER-Test", "hwe", "hwe.txt"),
    "train": ("UniMER-1M", "images", "train.txt"),
}

if __name__ == "__main__":
    for name, (archive, img_dir, label_file) in SPLITS.items():
        if not (DATA / archive).exists():
            print(f"extracting {archive}.zip ...")
            with zipfile.ZipFile(DATA / f"{archive}.zip") as z:
                z.extractall(DATA)
        src = DATA / archive / img_dir
        labels = (DATA / archive / label_file).read_text(encoding="utf-8").split("\n")
        ids = sorted(p.stem for p in src.glob("*.png"))
        refs = [labels[int(i)] for i in ids]
        with Pool() as pool:
            norms = pool.map(normalize_or_none, refs, chunksize=500)

        out = DATA / name
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "images").exists():
            (out / "images").symlink_to(src)
        with (out / "refs.jsonl").open("w", encoding="utf-8") as f:
            for i, ref, norm in zip(ids, refs, norms):
                f.write(json.dumps({"id": i, "image": f"images/{i}.png", "ref": ref, "norm": norm}) + "\n")
        rejected = sum(n is None for n in norms)
        print(f"{name}: {len(ids)} images, {rejected} labels rejected ({rejected / len(ids):.2%}) -> {out}")
