"""Read handwritten PDF pages whose ink is vector strokes (OneNote's PDF export) into text with $...$ math: each
line of writing is found from the strokes, drawn like the training data and read; a line that is all math
becomes a display formula ($$...$$). Page by page, typed text and pasted pictures are not read. With --markdown,
each note becomes a Markdown file: its typed title and date on top, its pages joined (lines cut by a page break
read whole), pictures such as printouts saved beside it where they sit.

    python scripts/read_pdf.py notes.pdf --pages 3 5-7 [--run run6] [--cpu] [--jsonl pages.jsonl]
    python scripts/read_pdf.py notes.pdf --markdown notes/ [--pages 3-6]
"""
import argparse
import json
import re
from pathlib import Path

import pypdfium2 as pdfium
import torch

from im2tex.gpu import cap_memory
from im2tex.inference import load_predictor
from im2tex.pdf_notes import group_notes, read_note, read_page


def page_numbers(specs, count):
    """["3", "5-7"] -> [3, 5, 6, 7]; none given -> every page."""
    if not specs:
        return list(range(1, count + 1))
    out = []
    for s in specs:
        a, _, b = s.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", nargs="*", default=[], help="page numbers or ranges like 5-7 (default: all)")
    ap.add_argument("--run", default="run6", help="training run whose best checkpoint to use")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU, e.g. while the GPU is training")
    ap.add_argument("--jsonl", type=Path, help="also write one {\"page\", \"text\"} row per page here")
    ap.add_argument("--markdown", type=Path, help="write one .md per note into this folder instead")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda":
        cap_memory()
    predict = load_predictor(args.run, device=device, lines=True)
    pdf = pdfium.PdfDocument(args.pdf)
    pages = page_numbers(args.pages, len(pdf))
    if args.markdown:
        args.markdown.mkdir(parents=True, exist_ok=True)
        written = set()
        for note in group_notes(pdf, pages):
            link = f"pictures/p{note[0]}"
            title, text = read_note(pdf, note, predict, args.markdown / link, link)
            stem = re.sub(r'[\\/:*?"<>|]+', "-", title).strip() if title else f"page {note[0]}"
            if stem in written:
                stem += f" (p{note[0]})"
            written.add(stem)
            (args.markdown / f"{stem}.md").write_text(text, encoding="utf-8")
            print(f"{args.markdown / stem}.md  (pages {note[0]}-{note[-1]})")
    else:
        rows = []
        for n in pages:
            text = "\n".join(line.text for line in read_page(pdf[n - 1], predict))
            print(f"=== page {n}" + ("" if text else " (no handwriting)"))
            if text:
                print(text)
            rows.append({"page": n, "text": text})
        if args.jsonl:
            args.jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
