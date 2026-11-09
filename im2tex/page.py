"""Whole-page transcriptions: plain text with $inline$ and $$display$$ LaTeX math, one written line per line.

    Every symmetric matrix is hermitian. For example $A = A^T$ when
    $$ A = \\begin{pmatrix} 1 & 2 \\\\ 2 & 3 \\end{pmatrix} $$
    [figure]

Headings, bullets and numbering are written as they appear, a drawing is a line "[figure]", crossed-out
writing is left out and a literal dollar sign is \\$. Scoring collapses all whitespace, so how the lines
are broken doesn't matter.
"""
import re
from collections import Counter

import numpy as np
from rapidfuzz.distance import Levenshtein

from im2tex.normalize import normalize, normalize_or_none

_DELIM = re.compile(r"(?<!\\)\$\$?")


class PageError(ValueError):
    pass


def split_page(text: str) -> list[tuple[str, str]]:
    """-> [(kind, content)] in order, kind "text", "inline" or "display". Raises PageError on an unclosed
    or mismatched $ / $$."""
    parts, pos, delim, start = [], 0, None, 0
    for m in _DELIM.finditer(text):
        if delim is None:
            parts.append(("text", text[pos:m.start()]))
            delim, start = m.group(), m.start()
        elif m.group() == delim:
            parts.append(("display" if delim == "$$" else "inline", text[pos:m.start()]))
            delim = None
        else:
            raise PageError(f"{m.group()} inside {delim}...{delim} at character {m.start()}")
        pos = m.end()
    if delim is not None:
        raise PageError(f"{delim} at character {start} is never closed")
    parts.append(("text", text[pos:]))
    return [(k, c) for k, c in parts if k != "text" or c.strip()]


def _parts(text: str, strict: bool) -> list[tuple[str, str]]:
    """Parts with whitespace collapsed and math normalized. Not strict (for predictions): a page that doesn't
    split is all text and math the normalizer rejects is kept as written."""
    try:
        parts = split_page(text)
    except PageError:
        if strict:
            raise
        parts = [("text", text)]
    out = []
    for kind, content in parts:
        if kind != "text":
            content = normalize(content) if strict else normalize_or_none(content) or content
        out.append((kind, " ".join(content.split())))
    return out


def _join(parts) -> str:
    return " ".join(c if k == "text" else f"$${c}$$" if k == "display" else f"${c}$" for k, c in parts)


def normalize_page(text: str, strict: bool = True) -> str:
    """One canonical line: whitespace collapsed, every formula through im2tex.normalize. Strict raises
    PageError / NormalizeError on a label that can't be read."""
    return _join(_parts(text, strict))


def page_metrics(preds: list[str], refs: list[str]) -> dict:
    """Mean normalized character edit distance of whole pages and the character error rate of their text
    (math left out), plus formula recall and precision over all pages: a formula counts as found when the
    same normalized formula appears on the predicted page."""
    page_edit, text_cer, found, n_ref, n_pred = [], [], 0, 0, 0
    for pred, ref in zip(preds, refs, strict=True):
        p, r = _parts(pred, strict=False), _parts(ref, strict=False)
        pj, rj = _join(p), _join(r)
        page_edit.append(Levenshtein.distance(pj, rj) / max(len(pj), len(rj), 1))
        pt, rt = (" ".join(c for k, c in x if k == "text") for x in (p, r))
        text_cer.append(Levenshtein.distance(pt, rt) / max(len(rt), 1))
        pf, rf = (Counter(c for k, c in x if k != "text") for x in (p, r))
        found += sum((pf & rf).values())
        n_ref, n_pred = n_ref + sum(rf.values()), n_pred + sum(pf.values())
    return {"n": len(refs), "page_edit": float(np.mean(page_edit)), "text_cer": float(np.mean(text_cer)),
            "formulas": n_ref, "formula_recall": found / max(n_ref, 1), "formula_precision": found / max(n_pred, 1)}
