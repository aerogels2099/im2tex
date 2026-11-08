"""Precompute each shard row's ink crop box and token count, saved as index.npy in the shard directory.

The batch sampler plans batches from these sizes without decoding every image on every run.
Re-run after retraining the tokenizer.
    python scripts/index_shards.py
"""
import argparse
import io
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from PIL import Image

from im2tex.data import ink_box
from im2tex.tokenizer import load

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))
_rows = None


def _open(path):
    global _rows
    _rows = load_from_disk(path).select_columns(["image"])


def _boxes(span):
    return [ink_box(Image.open(io.BytesIO(_rows[i]["image"])).convert("L")) for i in range(*span)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    tok = load(DATA / "tokenizer.json")
    for split in ("val", "train"):
        path = DATA / "shards" / split
        ds = load_from_disk(str(path))
        spans = [(i, min(i + 5000, len(ds))) for i in range(0, len(ds), 5000)]
        with Pool(args.workers, _open, (str(path),)) as pool:
            boxes = [b for chunk in pool.imap(_boxes, spans) for b in chunk]
        ntok = [len(e.ids) for e in tok.encode_batch(ds["label"])]
        index = np.column_stack([np.asarray(boxes, dtype=np.int32), np.asarray(ntok, dtype=np.int32)])
        np.save(path / "index.npy", index)
        w, h = index[:, 2] - index[:, 0], index[:, 3] - index[:, 1]
        print(f"{split}: {len(index)} rows; ink box median {int(np.median(w))}x{int(np.median(h))}, "
              f"p95 {int(np.percentile(w, 95))}x{int(np.percentile(h, 95))}; tokens median "
              f"{int(np.median(index[:, 4]))}, max {index[:, 4].max()}")
