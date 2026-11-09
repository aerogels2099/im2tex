"""Score whole-page predictions (JSONL rows with "pred" and "ref" page transcriptions, see im2tex/page.py).

    python scripts/evaluate_pages.py runs/got_ocr/notes.jsonl     # writes runs/got_ocr/notes.metrics.json
"""
import argparse
import json
from pathlib import Path

from im2tex.page import page_metrics

ap = argparse.ArgumentParser()
ap.add_argument("preds", type=Path)
args = ap.parse_args()

rows = [json.loads(line) for line in args.preds.open(encoding="utf-8")]
metrics = page_metrics([r["pred"] for r in rows], [r["ref"] for r in rows])
if "seconds" in rows[0]:
    metrics["sec_per_image"] = sum(r["seconds"] for r in rows) / len(rows)

print(json.dumps(metrics, indent=2))
args.preds.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
