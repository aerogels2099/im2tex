"""Read handwritten PDF pages whose ink is vector strokes (OneNote's PDF export) into text with $...$ math: each
line of writing is found from the strokes, drawn like the training data and read; a line that is all math
becomes a display formula ($$...$$). Typed text and pasted pictures on a page are not read.

    python scripts/read_pdf.py notes.pdf --pages 3 5-7 [--run run6] [--cpu] [--jsonl pages.jsonl]
"""
import argparse
import json
from pathlib import Path

import pypdfium2 as pdfium
import torch
from PIL import Image

from im2tex.gpu import cap_memory
from im2tex.ink import page_lines, render_ink
from im2tex.inference import load_predictor
from im2tex.page import PageError, split_page


def page_numbers(specs, count):
    """["3", "5-7"] -> [3, 5, 6, 7]; none given -> every page."""
    if not specs:
        return list(range(1, count + 1))
    out = []
    for s in specs:
        a, _, b = s.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def as_display(text):
    """A line read as math only -> $$...$$; otherwise as read."""
    try:
        parts = split_page(text)
    except PageError:
        return text
    if parts and all(k != "text" for k, _ in parts):
        return "$$" + " ".join(c.strip() for _, c in parts) + "$$"
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", nargs="*", default=[], help="page numbers or ranges like 5-7 (default: all)")
    ap.add_argument("--run", default="run6", help="training run whose best checkpoint to use")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU, e.g. while the GPU is training")
    ap.add_argument("--jsonl", type=Path, help="also write one {\"page\", \"text\"} row per page here")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda":
        cap_memory()
    predict = load_predictor(args.run, device=device, lines=True)
    pdf = pdfium.PdfDocument(args.pdf)
    rows = []
    for n in page_numbers(args.pages, len(pdf)):
        lines = page_lines(pdf[n - 1])
        images = [Image.fromarray(render_ink([o for s in line for o in s.outlines], filled=True)).convert("RGB")
                  for line in lines]
        text = "\n".join(as_display(predict([im])[0]) for im in images)  # one at a time: padding changes readings
        print(f"=== page {n}" + ("" if lines else " (no handwriting)"))
        if text:
            print(text)
        rows.append({"page": n, "text": text})
    if args.jsonl:
        args.jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
