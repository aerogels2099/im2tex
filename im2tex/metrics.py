"""String metrics (on tokens) and the rendered-image match metric."""
import numpy as np
import sacrebleu
from rapidfuzz.distance import Levenshtein

from im2tex.tokenize_latex import tokenize


def string_metrics(preds: list[str], refs: list[str]) -> dict:
    """BLEU-4, normalized token edit distance, and CROHME-style ExpRate (exact / <=1 / <=2 token errors)."""
    assert len(preds) == len(refs)
    p_tok = [tokenize(p) for p in preds]
    r_tok = [tokenize(r) for r in refs]
    dists = [Levenshtein.distance(p, r) for p, r in zip(p_tok, r_tok)]
    n = len(refs)
    bleu = sacrebleu.corpus_bleu(
        [" ".join(p) for p in p_tok], [[" ".join(r) for r in r_tok]], tokenize="none", force=True
    )
    return {
        "n": n,
        "bleu4": bleu.score / 100,
        "edit_dist": float(np.mean([d / max(len(p), len(r), 1) for d, p, r in zip(dists, p_tok, r_tok)])),
        "exprate": sum(d == 0 for d in dists) / n,
        "exprate_le1": sum(d <= 1 for d in dists) / n,
        "exprate_le2": sum(d <= 2 for d in dists) / n,
    }


def _ink(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Binarize (dark = ink) and crop to the ink bounding box."""
    ink = np.asarray(img) < threshold
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    if rows.size == 0:
        return ink[:0, :0]
    return ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def image_match(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two renders are pixel-identical after binarizing and trimming whitespace."""
    a, b = _ink(a), _ink(b)
    return a.shape == b.shape and np.array_equal(a, b)


def image_metrics(pred_imgs: list, ref_imgs: list) -> dict:
    """pred/ref renders as arrays; None = failed to render. Samples whose reference failed are excluded."""
    pairs = [(p, r) for p, r in zip(pred_imgs, ref_imgs) if r is not None]
    n = len(pairs)
    return {
        "n_image": n,
        "render_rate": sum(p is not None for p, _ in pairs) / n,
        "image_match": sum(p is not None and image_match(p, r) for p, r in pairs) / n,
    }
