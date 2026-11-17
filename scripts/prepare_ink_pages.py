"""Turn handwritten PDF pages with vector ink (OneNote's PDF export) and their page transcriptions into line
splits for training, in the common layout (train / validation: images/ + refs.jsonl).

Each page's lines are found from its strokes (im2tex.ink.page_lines), read with a trained run, and matched in
order to the transcription's lines by dynamic programming on edit distance; groups are allowed, so a line found
in two pieces doesn't shift the rest. A found line matched to exactly one written line becomes a row, labeled in
the line format (words, and $ latex $ with the math normalized). Skipped: [figure] / [table] lines and table
rows (| a | b |), and pairs whose reading is too far from the label to be the same line. 10% of the pages (by a
hash of the page number) are the validation split, whole pages.
    python scripts/prepare_ink_pages.py notes.pdf <transcriptions dir> <out dir> [--run run5]
Transcriptions: one file per page in the page format (im2tex/page.py), named ...p<page number>.txt.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from rapidfuzz.distance import Levenshtein

from im2tex.ink import Table, page_lines, render_ink
from im2tex.inference import load_predictor
from im2tex.normalize import normalize
from im2tex.page import normalize_page, split_page

# Characters each extra piece or line in a group costs. Low: a line found in two pieces must group with its written
# line rather than pair one piece with all of it (a wrong training row); groups only cost rows, never add bad ones.
GROUP = 2
MAX_CER = 0.6  # a pair read further than this from its label is taken to be mismatched


def line_label(line: str) -> str:
    """A page-format line -> the line training format: words as written, math as $ normalized $."""
    out = []
    for kind, content in split_page(line):
        out.append(" ".join(content.split()) if kind == "text" else f"$ {normalize(content)} $")
    return " ".join(p for p in out if p)


def comparable(text: str) -> str:
    """A reading or a label as one line with math normalized, display math written inline."""
    return normalize_page(text, strict=False).replace("$$", "$")


def match(reads: list[str], labels: list[str]) -> list[tuple[int, int]]:
    """Order-preserving alignment of found lines (their readings) to written lines -> 1:1 pairs (i, j)."""
    a, b = [comparable(r) for r in reads], [comparable(t) for t in labels]
    inf = float("inf")
    cost = [[inf] * (len(b) + 1) for _ in range(len(a) + 1)]
    back = [[None] * (len(b) + 1) for _ in range(len(a) + 1)]
    cost[0][0] = 0
    for i in range(len(a) + 1):
        for j in range(len(b) + 1):
            if cost[i][j] == inf:
                continue
            for di in range(4):
                for dj in range(3):
                    if (di, dj) == (0, 0) or i + di > len(a) or j + dj > len(b):
                        continue
                    p, q = " ".join(a[i:i + di]), " ".join(b[j:j + dj])
                    c = Levenshtein.distance(p, q) + GROUP * (di + dj - 2) if di and dj else len(p) + len(q)
                    if cost[i][j] + c < cost[i + di][j + dj]:
                        cost[i + di][j + dj], back[i + di][j + dj] = cost[i][j] + c, (di, dj)
    pairs, i, j = [], len(a), len(b)
    while i or j:
        di, dj = back[i][j]
        i, j = i - di, j - dj
        if (di, dj) == (1, 1):
            pairs.append((i, j))
    return pairs[::-1]


def held_out(page: int) -> bool:
    return int(hashlib.md5(f"p{page}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("transcriptions", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--run", default="run5", help="trained run that reads the lines for matching")
    args = ap.parse_args()

    predict = load_predictor(args.run, lines=True)
    pdf = pdfium.PdfDocument(args.pdf)
    rows = {"train": [], "validation": []}
    for t in sorted(args.transcriptions.glob("*.txt")):
        n = int(re.search(r"p(\d+)$", t.stem).group(1))
        split = "validation" if held_out(n) else "train"
        labels = [l for l in t.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines = [line for line in page_lines(pdf[n - 1]) if not isinstance(line, Table)]  # tables aren't lines
        images =[Image.fromarray(render_ink([o for s in line for o in s.outlines], filled=True)) for line in lines]
        reads = [predict([im.convert("RGB")])[0] for im in images]
        kept = 0
        for i, j in match(reads, labels):
            if labels[j].strip() in ("[figure]", "[table]") or labels[j].lstrip().startswith("|"):
                continue  # a figure, or a table (written out as Markdown rows): not a line of writing
            label = line_label(labels[j])
            if Levenshtein.normalized_distance(comparable(reads[i]), comparable(labels[j])) > MAX_CER:
                continue
            row_id = f"p{n:03d}_{i:02d}"
            (args.out / split / "images").mkdir(parents=True, exist_ok=True)
            images[i].save(args.out / split / "images" / f"{row_id}.png")
            rows[split].append({"id": row_id, "image": f"images/{row_id}.png", "ref": label, "norm": label})
            kept += 1
        print(f"page {n} ({split}): {len(lines)} lines found, {len(labels)} written, {kept} kept")
    for split, r in rows.items():
        (args.out / split).mkdir(parents=True, exist_ok=True)
        (args.out / split / "refs.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in r),
                                                    encoding="utf-8")
        print(f"{split}: {len(r)} lines -> {args.out / split}")
