"""Synthetic handwritten lines that mix words and math, for the line recognizer: IAM text lines cut at the gaps
between words, with handwritten formulas (MathWriting, HME100K, CROHME) glued in at the text's writing size.
One row in ten is a formula alone, written as a line. Half the glued formulas are short inline math with a letter
(x, T, u_1, \\hat{x}, x = 5), as in notes ("the matrix $A$ is"). Bare numbers are left out: in a line they are text.

Labels are page-format lines (im2tex/page.py) with a space inside each $ ... $, so a formula is tokenized as in
formula mode: "the map $ f : V \\to W $ is linear". Writes <data>/mixed_lines/<split>/{refs.jsonl, images/}.
    python scripts/make_mixed_lines.py --split train --n 150000
    python scripts/make_mixed_lines.py --split val --n 1000
"""
import argparse
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

from im2tex.data import ink_box

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))
# split -> (IAM split, [(formula split, share of formulas)]); validation lines and formulas are never trained on
SPLITS = {"train": ("train", [("mathwriting/prepared/train", 0.6), ("hme100k/train", 0.25), ("crohme/train", 0.15)]),
          "val": ("validation", [("mathwriting/prepared/valid", 1.0)])}
MAX_W = 2600  # px at IAM's 128 px line height; IAM's own lines are 1,800 wide at the median, 2,600 at p90
MAX_FORMULA_TOKENS = 40  # inline math is short
SHORT_TOKENS, SHORT_SHARE = 5, 0.5  # glued formulas this short, with a letter, drawn this often
LETTER = re.compile(r"[A-Za-z]|\\(var)?(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho"
                    r"|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)")
# Letters as tall as an x, and what may sit beside them on a formula's main line without changing its height.
X_HEIGHT = set("aceimnorsuvwxz") | {"\\" + g for g in ("alpha", "epsilon", "varepsilon", "iota", "kappa", "nu", "omega",
                                                       "pi", "sigma", "tau", "upsilon")}
FLAT = {"=", "+", "-", ",", "\\hat", "\\tilde", "\\bar", "\\vec", "\\dot", "\\overline"}
_lines = _formulas = _short = _seed = None


def components(a):
    """-> ink mask (specks under 20 px dropped) and the (height, width, area) of each ink component of 30+ px.
    Ink is darker than halfway between the darkest ink and the paper."""
    b = a < (np.percentile(a, 1) + np.median(a)) / 2
    labels, _ = ndimage.label(b, structure=np.ones((3, 3)))
    area = np.bincount(labels.ravel())
    area[0] = 0
    hw = np.array([(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in ndimage.find_objects(labels)])
    hwa = np.column_stack([hw.reshape(-1, 2), area[1:]])
    return (area >= 20)[labels], hwa[hwa[:, 2] >= 30]


def whiten(a, mask):
    """Paper to pure white: everything lighter than the darker paper (10th percentile of the pixels away from
    ink) becomes 255 and the ink is stretched to match, so pieces from different scans and photos meet without
    a seam."""
    paper = a[~ndimage.binary_dilation(mask, iterations=2)]
    white = np.percentile(paper, 10) if paper.size else 255
    return (a.astype(np.float32) * 255 / max(white, 1)).clip(0, 255).astype(np.uint8)


def symbol_px(hwa):
    """Median height of ink components weighted by their ink, so the specks of a noisy photo barely count."""
    order = np.argsort(hwa[:, 0])
    c = np.cumsum(hwa[order, 2])
    return float(hwa[order[np.searchsorted(c, c[-1] / 2)], 0])


def x_height(mask):
    """The writing's x-height (symbol_px is the height of ascenders when joined-up words are single components).
    The median of three measures of the ink in each row, which misjudge different writers: the band of rows holding
    at least half the ink of the fullest row (too narrow under heavy letter tops), at least a third of it, and the
    rows holding the middle 70% of the ink (too wide with big loops)."""
    rows = mask.sum(1)
    band = lambda frac: np.ptp(np.flatnonzero(rows >= frac * rows.max())) + 1  # noqa: E731
    c = np.cumsum(rows) / rows.sum()
    return float(np.median([band(1 / 2), band(1 / 3), np.searchsorted(c, 0.85) - np.searchsorted(c, 0.15) + 1]))


def x_height_only(norm):
    """Whether a formula's main line (outside sub- and superscripts) is letters as tall as an x, with nothing taller:
    x, u = a, \\hat { c }, x _ { 1 }, but not T, f, 2 x or \\sqrt { x }."""
    tokens, main, scripts, depth = norm.split(), [], [], 0  # scripts: brace depths where a sub/superscript opened
    for prev, t in zip([None] + tokens, tokens):
        if t == "{":
            depth += 1
            if prev in ("_", "^"):
                scripts.append(depth)
        elif t == "}":
            if scripts and scripts[-1] == depth:
                scripts.pop()
            depth -= 1
        elif not scripts and t not in ("_", "^"):
            main.append(t)
    return any(t in X_HEIGHT for t in main) and all(t in X_HEIGHT or t in FLAT for t in main)


def word_cuts(mask, words):
    """x positions between the words of a line and the widths of those gaps, or None when the blank-column gaps
    can't be matched to the label's spaces with confidence: the k widest gaps (k spaces) must stand clearly above
    the rest, and each word's width must fit its letter count. A label with a double quote is skipped: a lone
    quote is often written apart, a gap that isn't a space."""
    k = len(words) - 1
    cols = mask.any(0)
    xs = np.flatnonzero(cols)
    if k == 0 or xs.size == 0 or any('"' in w for w in words):
        return None
    edges = np.diff(cols[xs[0]: xs[-1] + 1].astype(np.int8))
    starts, ends = np.flatnonzero(edges == -1) + 1 + xs[0], np.flatnonzero(edges == 1) + 1 + xs[0]
    widths = ends - starts
    if len(widths) < k:
        return None
    ranked = np.sort(widths)[::-1]
    kth, rest = ranked[k - 1], ranked[k] if len(ranked) > k else 0
    if kth < 6 or kth < max(1.4 * rest, rest + 4):
        return None
    pick = widths >= kth
    cuts = (starts[pick] + ends[pick]) // 2
    per_char = np.diff([xs[0], *cuts, xs[-1] + 1]) / np.array([len(w) + 1 for w in words])
    if np.any(per_char < 0.5 * np.median(per_char)) or np.any(per_char > 2 * np.median(per_char)):
        return None
    return cuts.tolist(), widths[pick].tolist()


def _cuttable(row):
    """(image path, words, cuts, gap widths) or None."""
    path, text = row
    mask, _ = components(np.asarray(Image.open(path).convert("L")))
    words = text.split(" ")
    found = word_cuts(mask, words)
    return (path, words, *found) if found else None


def short(norm):
    """Short inline math: a few tokens, one of them a Latin or Greek letter."""
    tokens = norm.split()
    return len(tokens) <= SHORT_TOKENS and any(LETTER.fullmatch(t) for t in tokens)


def bare_number(norm):
    """1 5, 2 . 5, 1 , 0 0 0: text in a line, not math."""
    return re.fullmatch(r"[0-9.,]+", norm.replace(" ", "")) is not None


def _init(lines, formulas, seed):
    global _lines, _formulas, _short, _seed
    _lines, _formulas, _seed = lines, formulas, seed
    _short = [([r for r in rows if short(r[1])], share) for rows, share in formulas]
    _short = [(rows, share) for rows, share in _short if rows]


def _formula(rng, short_math=False):
    """-> (grayscale array cropped to its ink with the paper made white, normalized LaTeX). short_math: draw from
    the short inline math."""
    pools = _short if short_math else _formulas
    shares = np.array([s for _, s in pools])
    rows = pools[rng.choice(len(pools), p=shares / shares.sum())][0]
    path, norm = rows[rng.integers(len(rows))]
    img = Image.open(path).convert("L")
    a = np.asarray(img.crop(ink_box(img)))
    return whiten(a, components(a)[0]), norm


def _formula_px(a):
    """A formula's symbol height, leaving out flat symbols (minus signs, fraction bars)."""
    _, hwa = components(a)
    upright = hwa[hwa[:, 0] >= 0.4 * hwa[:, 1]]
    return symbol_px(upright) if len(upright) else symbol_px(hwa) if len(hwa) else a.shape[0]


def glue(rng, line):
    """Cut a run of words out of an IAM line and glue one or two formulas in before, between or after them,
    scaled to the line's writing size and centered on its writing. -> (grayscale array, label)."""
    path, words, cuts, gaps = line
    a = np.asarray(Image.open(path).convert("L"))
    mask, hwa = components(a)
    text_px, x_px, center = symbol_px(hwa), x_height(mask), np.median(np.nonzero(mask)[0])
    a = whiten(a, mask)
    xs = np.flatnonzero(mask.any(0))
    bounds = [xs[0], *cuts, xs[-1] + 1]  # word j spans bounds[j]:bounds[j + 1]
    ratio = rng.uniform(0.8, 1.2)  # formula symbols against the text's ink components
    formulas = []
    while len(formulas) < (1 if rng.random() < 0.75 else 2):
        f, norm = _formula(rng, rng.random() < SHORT_SHARE)
        s = (x_px if x_height_only(norm) else text_px) * ratio / _formula_px(f)
        size = max(1, round(f.shape[1] * s)), max(1, round(f.shape[0] * s))
        # Outside 1/3 to 3x the symbols were likely mismeasured (a faint photo breaks into specks): draw another.
        if 1 / 3 <= s <= 3 and size[0] <= 0.6 * MAX_W:
            formulas.append((np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS)), norm))
    # The longest run of words from a random start that fits beside the formulas (at least one word).
    budget = rng.uniform(0.4, 1.0) * MAX_W - sum(f.shape[1] for f, _ in formulas)
    first = int(rng.integers(len(words)))
    end = first + 1
    while end < len(words) and bounds[end + 1] - bounds[first] <= budget:
        end += 1
    slots = set(rng.choice(end - first + 1, len(formulas), replace=False).tolist())  # formula before word first+s
    pieces, run = [], []

    def close_run():
        if run:
            cols = np.flatnonzero(mask[:, bounds[run[0]]: bounds[run[-1] + 1]].any(0)) + bounds[run[0]]
            pieces.append((a[:, cols[0]: cols[-1] + 1], 0, " ".join(words[j] for j in run)))
            run.clear()

    for s in range(end - first + 1):
        if s in slots:
            close_run()
            f, norm = formulas.pop()
            top = round(center + rng.normal(0, 0.1) * text_px - f.shape[0] / 2)
            pieces.append((f, top, f"$ {norm} $"))
        if first + s < end:
            run.append(first + s)
    close_run()

    lo = min(0, *(top for _, top, _ in pieces))
    hi = max(a.shape[0], *(top + p.shape[0] for p, top, _ in pieces))
    spaces = [round(np.median(gaps) * rng.uniform(0.7, 1.3)) for _ in pieces[1:]]
    canvas = np.full((hi - lo, sum(p.shape[1] for p, _, _ in pieces) + sum(spaces)), 255, np.uint8)
    x = 0
    for k, (p, top, _) in enumerate(pieces):
        region = canvas[top - lo: top - lo + p.shape[0], x: x + p.shape[1]]
        region[:] = np.minimum(region, p)
        x += p.shape[1] + (spaces[k] if k < len(spaces) else 0)
    return canvas, " ".join(label for _, _, label in pieces)


def make(i):
    """Row i of the split, the same on every run."""
    rng = np.random.default_rng([_seed, i])
    if rng.random() < 0.1:
        f, norm = _formula(rng)
        return f, f"$ {norm} $"
    return glue(rng, _lines[rng.integers(len(_lines))])


def _write(job):
    i, out = job
    a, label = make(i)
    row_id = f"{out.name}-{i:06d}"
    Image.fromarray(a).save(out / "images" / f"{row_id}.jpg", quality=90)
    return {"id": row_id, "image": f"images/{row_id}.jpg", "ref": label, "norm": label}


def load_sources(split, workers):
    """-> (cuttable IAM lines, [(formulas, share)]) for a split. IAM lines with a crossed-out word are left out:
    the label drops the word but the image still shows it, so the gaps no longer match. So are bare numbers."""
    iam_split, sources = SPLITS[split]
    raw = pq.read_table(DATA / "iam" / f"{iam_split}.parquet", columns=["text"]).column("text").to_pylist()
    struck = {f"{iam_split}-{i:05d}" for i, t in enumerate(raw) if "#" in t}
    d = DATA / "iam" / iam_split
    rows = [(str(d / r["image"]), r["norm"]) for r in map(json.loads, open(d / "refs.jsonl", encoding="utf-8"))
            if r["norm"] and r["id"] not in struck]
    with Pool(workers) as pool:
        lines = [x for x in pool.map(_cuttable, rows, chunksize=64) if x]
    formulas = []
    for rel, share in sources:
        refs = map(json.loads, open(DATA / rel / "refs.jsonl", encoding="utf-8"))
        formulas.append(([(str(DATA / rel / r["image"]), r["norm"]) for r in refs
                          if r["norm"] and len(r["norm"].split()) <= MAX_FORMULA_TOKENS
                          and not bare_number(r["norm"])], share))
    print(f"{split}: {len(lines)} of {len(rows)} IAM lines cut at word gaps; formulas "
          + ", ".join(f"{rel} {len(f)}" for (f, _), (rel, _) in zip(formulas, sources)))
    return lines, formulas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    lines, formulas = load_sources(args.split, args.workers)
    out = DATA / "mixed_lines" / args.split
    (out / "images").mkdir(parents=True, exist_ok=True)
    seed = list(SPLITS).index(args.split)
    with Pool(args.workers, _init, (lines, formulas, seed)) as pool:
        rows = pool.map(_write, [(i, out) for i in range(args.n)], chunksize=256)
    with (out / "refs.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{args.split}: {len(rows)} lines -> {out}")
