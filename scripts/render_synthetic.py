"""Render normalized formulas in many math fonts, so the model sees more than Computer Modern.

Real inputs come from textbooks (Times/Palatino-like fonts) and slides (PowerPoint equations use
Cambria Math). Rendered once at high DPI; training downscales at random.
    python scripts/render_synthetic.py ~/im2tex-data/unimer/train/refs.jsonl --n 50000
"""
import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image

from im2tex.render.latex_batch import PDFLATEX_PREAMBLE, render

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))
XELATEX_PREAMBLE = r"""\documentclass{article}
\usepackage{amsmath,mathtools,cancel}
\usepackage{unicode-math}
\setmainfont{%s}
\setmathfont{%s}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{1pt}
\pagestyle{empty}
"""
FONTS = {  # name -> (engine, preamble)
    "cm": ("pdflatex", PDFLATEX_PREAMBLE),
    **{name: ("xelatex", XELATEX_PREAMBLE % (text, math)) for name, (math, text) in {
        "latinmodern": ("latinmodern-math.otf", "lmroman10-regular.otf"),
        "stix2": ("STIXTwoMath-Regular.otf", "STIXTwoText-Regular.otf"),
        "xits": ("XITSMath-Regular.otf", "STIXTwoText-Regular.otf"),
        "termes": ("texgyretermes-math.otf", "texgyretermes-regular.otf"),
        "pagella": ("texgyrepagella-math.otf", "texgyrepagella-regular.otf"),
        "schola": ("texgyreschola-math.otf", "texgyreschola-regular.otf"),
        "bonum": ("texgyrebonum-math.otf", "texgyrebonum-regular.otf"),
        "cambria": ("Cambria Math", "Cambria"),
    }.items()},
}

ap = argparse.ArgumentParser()
ap.add_argument("source", type=Path, help="refs.jsonl with normalized labels")
ap.add_argument("--n", type=int, default=50000)
ap.add_argument("--dpi", type=int, default=240)
ap.add_argument("--inline", type=float, default=0.3, help="fraction rendered in text (inline) style")
ap.add_argument("--out", type=Path, default=DATA / "synthetic" / "printed")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = random.Random(args.seed)
labels = [r["norm"] for r in map(json.loads, args.source.open(encoding="utf-8")) if r["norm"]]
labels = rng.sample(labels, min(args.n, len(labels)))
(args.out / "images").mkdir(parents=True, exist_ok=True)

rows, names = [], list(FONTS)
jobs = {}  # (font, style) -> labels
for label in labels:
    jobs.setdefault((rng.choice(names), "" if rng.random() < args.inline else r"\displaystyle"), []).append(label)
for (font, style), group in sorted(jobs.items()):
    engine, preamble = FONTS[font]
    imgs = render(group, dpi=args.dpi, preamble=preamble, engine=engine, style=style)
    ok = 0
    for label, img in zip(group, imgs):
        if img is None:
            continue
        rid = f"{len(rows):07d}"
        Image.fromarray(img).save(args.out / "images" / f"{rid}.png")
        rows.append({"id": rid, "image": f"images/{rid}.png", "ref": label, "norm": label, "font": font,
                     "style": "display" if style else "inline"})
        ok += 1
    print(f"{font:12s} {'display' if style else 'inline':8s} {ok}/{len(group)} rendered", flush=True)

with (args.out / "refs.jsonl").open("w", encoding="utf-8") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print(f"{len(rows)} images -> {args.out}")
