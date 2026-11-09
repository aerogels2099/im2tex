"""Write the CROHME 2023 test split: handwriting scanned from paper, by new writers in Nantes, Luleå and Tokyo.

Images are IMG/test/CROHME2023_test; labels are the expression-level LaTeX of the matching InkML files (the 54
images without InkML have no LaTeX and are skipped). The labels use MathJax's \\lt and \\gt, which LaTeX lacks;
they become < and >. Labels the normalizer rejects or pdflatex can't render are dropped. Test only: the
archive's train/val folders repeat older CROHME data we train on.
    python scripts/prepare_crohme2023.py      (after scripts/download_data.sh)
"""
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from im2tex.normalize import normalize_or_none, strip_math_delimiters
from im2tex.render.latex_batch import render

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data")) / "crohme2023"
IMG, INK = "TC11_CROHME23/IMG/test/CROHME2023_test/", "TC11_CROHME23/INKML/test/CROHME2023_test/"
NS = "{http://www.w3.org/2003/InkML}"
MATHJAX = {"lt": "<", "gt": ">"}

with zipfile.ZipFile(DATA / "CROHME23.zip") as z:
    stems = sorted(Path(n).stem for n in z.namelist() if n.startswith(INK) and n.endswith(".inkml"))
    refs = []
    for stem in stems:
        # find() looks at the root's own annotations only: nested trace groups carry per-symbol "truth" labels
        truth = ET.fromstring(z.read(f"{INK}{stem}.inkml")).find(f"{NS}annotation[@type='truth']")
        ref = strip_math_delimiters((truth.text or "").strip()).strip()
        refs.append(re.sub(r"\\(lt|gt)(?![a-zA-Z])", lambda m: MATHJAX[m[1]], ref))
    norms = [normalize_or_none(r) for r in refs]
    renders = render(refs, workers=4)

    out = DATA / "test"
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, dropped = [], []
    for stem, ref, norm, img in zip(stems, refs, norms, renders):
        if norm is None or img is None:
            dropped.append(ref)
            continue
        (out / "images" / f"{stem}.png").write_bytes(z.read(f"{IMG}{stem}.png"))
        rows.append({"id": stem, "image": f"images/{stem}.png", "ref": ref, "norm": norm})

with (out / "refs.jsonl").open("w", encoding="utf-8") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print(f"test: {len(rows)} formulas, {len(dropped)} labels dropped -> {out}")
for ref in dropped[:5]:
    print(f"    dropped: {ref}")
