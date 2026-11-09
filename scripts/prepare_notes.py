"""Test only: 100 photographed pages of handwritten English linear-algebra notes (HumynLabs on Hugging Face).

Each PDF holds one page photo with a watermark and printed bands drawn over it; the photo is saved as is,
without them. The dataset says CC BY 4.0, but the overlays credit pkalika.in, so keep these pages and any
labels of them private. They come without transcriptions: label them as pages.
    python scripts/prepare_notes.py
    python scripts/label.py ~/im2tex-data/notes/pages ~/im2tex-data/notes/linalg --pages
"""
import os
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from huggingface_hub import snapshot_download

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data")) / "notes"

raw = Path(snapshot_download("HumynLabs/English-Handwritten-Math-Notes-Dataset", repo_type="dataset",
                             local_dir=DATA / "raw", allow_patterns=["*.pdf"]))
pages = DATA / "pages"
pages.mkdir(parents=True, exist_ok=True)
for pdf in sorted(raw.glob("*.pdf")):
    out = pages / f"{pdf.stem}.png"
    if not out.exists():
        (photo,) = pdfium.PdfDocument(pdf)[0].get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])
        photo.get_bitmap(render=False).to_pil().convert("RGB").save(out)
print(f"{len(list(pages.glob('*.png')))} pages in {pages}")
