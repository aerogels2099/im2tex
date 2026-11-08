"""Score a predictions file (JSONL rows with "pred" and "ref") on string and rendered-image metrics.

String metrics compare normalized prediction to normalized reference, so spellings that render the same
(\\mathbf{C} vs {\\bf C}) aren't counted as errors; a prediction the normalizer rejects is compared as-is.
Image metrics render the raw strings, so they don't depend on the normalizer at all.
    python scripts/evaluate.py runs/pix2tex_im2latex100k_test.jsonl
"""
import argparse
import json
from pathlib import Path

from im2tex.metrics import image_metrics, string_metrics
from im2tex.normalize import normalize_or_none
from im2tex.render.latex_batch import render

ap = argparse.ArgumentParser()
ap.add_argument("preds", type=Path)
ap.add_argument("--dpi", type=int, default=200)
args = ap.parse_args()

rows = [json.loads(line) for line in args.preds.open(encoding="utf-8")]
preds, refs = [r["pred"] for r in rows], [r["ref"] for r in rows]

metrics = string_metrics([normalize_or_none(p) or p for p in preds], [normalize_or_none(r) or r for r in refs])
imgs = render(preds + refs, dpi=args.dpi)  # one call so pred and ref batches render in parallel
metrics |= image_metrics(imgs[: len(rows)], imgs[len(rows) :])

print(json.dumps(metrics, indent=2))
args.preds.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
