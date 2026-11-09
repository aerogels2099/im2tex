"""Write test splits of OmniDocBench's display equations, cut from real document pages by their annotated boxes.

OmniDocBench (opendatalab/OmniDocBench; research use only, not commercial) has 1,651 annotated pages. Each
display equation ("equation_isolated") is cropped with a small margin. Labels the normalizer rejects or pdflatex
can't render (Chinese text, or broken LaTeX such as a matrix with more columns than declared) are dropped.
Splits, by the page's source:
    slides_notes   PPT2PDF, note (handwritten notes)
    textbooks      book, colorful_textbook
    papers_exams   academic_literature, exam_paper
Inline math has no boxes in the annotations, so it isn't used. Test only: never train on these.
    python scripts/prepare_omnidocbench.py      (downloads ~1.4 GB on first run)
"""
import json
import os
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

from im2tex.normalize import normalize_or_none, strip_math_delimiters
from im2tex.render.latex_batch import render

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data")) / "omnidocbench"
SPLITS = {"PPT2PDF": "slides_notes", "note": "slides_notes", "book": "textbooks", "colorful_textbook": "textbooks",
          "academic_literature": "papers_exams", "exam_paper": "papers_exams"}
MARGIN = 4  # pixels around the annotated box

raw = Path(snapshot_download("opendatalab/OmniDocBench", repo_type="dataset", local_dir=DATA / "raw",
                             allow_patterns=["OmniDocBench.json", "images/*"]))
found = []  # (split, id, page info, box polygon, latex)
for page in json.loads((raw / "OmniDocBench.json").read_text(encoding="utf-8")):
    info = page["page_info"]
    split = SPLITS.get(info["page_attribute"]["data_source"])
    for det in page["layout_dets"]:
        if split and det["category_type"] == "equation_isolated" and not det.get("ignore") and det.get("latex"):
            found.append((split, f"{Path(info['image_path']).stem}_{det['anno_id']}", info, det["poly"],
                          strip_math_delimiters(det["latex"].strip()).strip()))
norms = [normalize_or_none(f[4]) for f in found]
renders = render([f[4] for f in found], workers=4)

rows, dropped = {s: [] for s in set(SPLITS.values())}, Counter()
for (split, image_id, info, poly, ref), norm, img in zip(found, norms, renders):
    if norm is None or img is None:
        dropped[split] += 1
        continue
    with Image.open(raw / "images" / info["image_path"]) as page:
        sx, sy = page.width / info["width"], page.height / info["height"]
        xs, ys = [x * sx for x in poly[0::2]], [y * sy for y in poly[1::2]]
        box = (max(int(min(xs)) - MARGIN, 0), max(int(min(ys)) - MARGIN, 0),
               min(int(max(xs)) + MARGIN + 1, page.width), min(int(max(ys)) + MARGIN + 1, page.height))
        (DATA / split / "images").mkdir(parents=True, exist_ok=True)
        page.crop(box).convert("RGB").save(DATA / split / "images" / f"{image_id}.png")
    rows[split].append({"id": image_id, "image": f"images/{image_id}.png", "ref": ref, "norm": norm})

for split, split_rows in sorted(rows.items()):
    with (DATA / split / "refs.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in split_rows)
    print(f"{split}: {len(split_rows)} formulas, {dropped[split]} labels dropped -> {DATA / split}")
