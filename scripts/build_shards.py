"""Pack prepared training splits into Arrow shards: a few hundred large files instead of ~1.7M small PNGs.

Large files upload to a rented GPU in minutes and read faster than millions of small ones; Arrow also
gives random access, which the size-bucketing sampler needs. Validation rows are held out by a hash of
their id, so the split never changes between builds. Test sets stay as refs.jsonl folders.
    python scripts/build_shards.py [--out ~/im2tex-data/shards]
"""
import argparse
import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path

from datasets import Dataset, Features, Value
from PIL import Image

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))
# source -> (split directory, fraction held out for validation; None = the whole split is validation)
SOURCES = {
    "unimer": (DATA / "unimer/train", 0.005),
    "synthetic_printed": (DATA / "synthetic/printed", 0.02),
    "mathwriting": (DATA / "mathwriting/prepared/train", 0.0),
    "mathwriting_valid": (DATA / "mathwriting/prepared/valid", None),
    "mathwriting_synthetic": (DATA / "mathwriting/prepared/synthetic", 0.0),
    "mathwriting_symbols": (DATA / "mathwriting/prepared/symbols", 0.0),
    "crohme": (DATA / "crohme/train", 0.0),
}
FEATURES = Features({"id": Value("string"), "source": Value("string"), "label": Value("string"),
                     "image": Value("binary"), "width": Value("int32"), "height": Value("int32")})
CHUNK = 50_000


def held_out(row_id: str, fraction: float) -> bool:
    return int(hashlib.md5(row_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def rows(jobs):
    """jobs: (split, source, start line, stop line) tuples. Yields the rows of `split` in that range."""
    for split, source, start, stop in jobs:
        split_dir, fraction = SOURCES[source]
        with (split_dir / "refs.jsonl").open(encoding="utf-8") as f:
            for line in itertools.islice(f, start, stop):
                r = json.loads(line)
                in_val = True if fraction is None else held_out(f"{source}/{r['id']}", fraction)
                if not r["norm"] or in_val != (split == "val"):
                    continue
                path = split_dir / r["image"]
                with Image.open(path) as im:
                    width, height = im.size
                yield {"id": r["id"], "source": source, "label": r["norm"], "image": path.read_bytes(),
                       "width": width, "height": height}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA / "shards")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    sizes = {s: sum(1 for _ in (d / "refs.jsonl").open()) for s, (d, _) in SOURCES.items()
             if (d / "refs.jsonl").exists()}
    for split in ("val", "train"):
        jobs = [(split, s, i, i + CHUNK) for s, n in sizes.items() for i in range(0, n, CHUNK)]
        ds = Dataset.from_generator(rows, features=FEATURES, gen_kwargs={"jobs": jobs},
                                    num_proc=min(args.workers, len(jobs)))
        ds.save_to_disk(str(args.out / split), max_shard_size="500MB")
        print(f"{split}: {len(ds)} rows {dict(Counter(ds['source']))} -> {args.out / split}")
