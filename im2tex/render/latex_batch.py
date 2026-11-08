"""Render many formulas with one TeX run: each formula becomes its own tightly cropped PDF page.

One run per formula would spend most of its time on TeX startup. Batching amortizes that, so failures
have to be isolated: a run counts only if the log has no errors AND it produced one page per formula.
Otherwise the formulas blamed by the log are retried alone and the rest are retried as a batch;
if that still fails, the batch is split in half.
"""
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pymupdf

PDFLATEX_PREAMBLE = r"""\documentclass{article}
\usepackage{amsmath,amssymb,mathrsfs,bm,mathtools,cancel,upgreek,stmaryrd,esint}
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{1pt}
\pagestyle{empty}
"""

_ERROR_LINE = re.compile(r"^.*?doc\.tex:(\d+): ", re.M)


def render(formulas: list[str], dpi: int = 200, preamble: str = PDFLATEX_PREAMBLE,
           engine: str = "pdflatex", style: str = r"\displaystyle", batch_size: int = 200,
           workers: int = 8) -> list:
    """Return one grayscale uint8 array per formula, or None where the formula fails to compile.
    style is prepended inside $...$: \\displaystyle, or "" for inline (text) style."""
    results = [None] * len(formulas)
    chunks = [list(range(i, min(i + batch_size, len(formulas)))) for i in range(0, len(formulas), batch_size)]
    tex = dict(dpi=dpi, preamble=preamble, engine=engine, style=style)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda idx: _render_into(idx, formulas, results, tex), chunks))
    return results


def _render_into(idx, formulas, results, tex, use_blame=True):
    if not idx:
        return
    pages, blamed = _run_tex([formulas[i] for i in idx], **tex)
    if pages is not None:
        for i, img in zip(idx, pages):
            results[i] = img
        return
    if len(idx) == 1:
        return  # this formula is broken
    if use_blame and 0 < len(blamed) < len(idx):
        for j in blamed:
            _render_into([idx[j]], formulas, results, tex)
        rest = [i for j, i in enumerate(idx) if j not in blamed]
        _render_into(rest, formulas, results, tex, use_blame=False)
    else:
        mid = len(idx) // 2
        _render_into(idx[:mid], formulas, results, tex)
        _render_into(idx[mid:], formulas, results, tex)


def _run_tex(batch, dpi, preamble, engine, style):
    """Compile one batch. Returns (pages, set()) on a clean run, else (None, blamed local indices)."""
    lines = preamble.rstrip("\n").split("\n") + [r"\begin{document}"]
    starts = []
    for f in batch:
        starts.append(len(lines) + 1)  # 1-based line number where this item begins
        # The closing $ is on its own line so a stray % in the formula can't comment it out.
        lines += [r"\begin{preview}$" + style, " ".join(f.split()), r"$\end{preview}", ""]
    lines.append(r"\end{document}")

    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp, "doc.tex")
        tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = [engine, "-interaction=nonstopmode", "-file-line-error", "-no-shell-escape", tex.name]
        try:
            subprocess.run(cmd, cwd=tmp, capture_output=True, timeout=20 + 0.2 * len(batch))
        except subprocess.TimeoutExpired:
            return None, set()
        log = Path(tmp, "doc.log").read_text(encoding="utf-8", errors="replace")
        err_lines = [int(m) for m in _ERROR_LINE.findall(log)]
        pdf = Path(tmp, "doc.pdf")
        if not err_lines and "\n! " not in log and pdf.exists():
            with pymupdf.open(pdf) as doc:
                if doc.page_count == len(batch):
                    return [_rasterize(page, dpi) for page in doc], set()
    blamed = {max(j for j, s in enumerate(starts) if s <= n) for n in err_lines if n >= starts[0]} if starts else set()
    return None, blamed


def _rasterize(page, dpi):
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, : pix.width].copy()
