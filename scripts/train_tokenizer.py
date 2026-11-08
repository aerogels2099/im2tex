"""Train the model tokenizer on normalized training labels and report sequence lengths per dataset.

    python scripts/train_tokenizer.py <train refs.jsonl ...> [--report <test refs.jsonl ...>] [--vocab 4000]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from im2tex.tokenizer import train

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))

ap = argparse.ArgumentParser()
ap.add_argument("refs", type=Path, nargs="+", help="refs.jsonl files of training splits")
ap.add_argument("--report", type=Path, nargs="*", default=[], help="extra splits for length stats only")
ap.add_argument("--vocab", type=int, default=4000)
ap.add_argument("--out", type=Path, default=DATA / "tokenizer.json")
args = ap.parse_args()


def norms(path):
    return [r["norm"] for r in map(json.loads, path.open(encoding="utf-8")) if r["norm"]]


labels = {p: norms(p) for p in args.refs}
tok = train((l for ls in labels.values() for l in ls), vocab_size=args.vocab)
tok.save(str(args.out))
print(f"vocab {tok.get_vocab_size()} -> {args.out}")

labels |= {p: norms(p) for p in args.report}
print(f"{'dataset':60s} {'labels':>8s} {'median':>6s} {'p95':>5s} {'p99':>5s} {'max':>5s} {'>512':>6s}")
for path, ls in labels.items():
    n = np.array([len(e.ids) for e in tok.encode_batch(ls)])
    print(f"{str(path)[-60:]:60s} {len(ls):8d} {int(np.median(n)):6d} {int(np.percentile(n, 95)):5d} "
          f"{int(np.percentile(n, 99)):5d} {n.max():5d} {(n > 512).mean():6.2%}")
