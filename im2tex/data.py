"""Training data: shard rows -> cropped, resized image tensors + token ids, batched by size.

Images are cropped to their ink, resized to fit (max_h, max_w) keeping the aspect ratio, and stored as
ink = 1 / background = 0 so zero padding is blank paper. Batches group similar sizes under a pixel and a
token budget, so batch size adapts: many small formulas or a few large ones.
"""
import io

import numpy as np
import torch
from PIL import Image

from im2tex.augment import augment

INK = 200  # pixels darker than this count as ink when cropping
MARGIN = 4  # pixels kept around the ink
SPECK = 0.01  # most of the ink a cluster at the edge can hold and still be dropped as dust
PAD_ID = 0  # tokenizer's <pad>


def _clusters(counts, gap):
    """Runs of ink along one axis, split where more than gap lines are empty: starts, ends, share of the ink."""
    at = np.flatnonzero(counts)
    cut = np.flatnonzero(np.diff(at) > gap + 1)
    starts, ends = at[np.r_[0, cut + 1]], at[np.r_[cut, at.size - 1]] + 1
    total = np.r_[0, np.cumsum(counts)]
    return starts, ends, (total[ends] - total[starts]) / total[-1]


def _trim(starts, ends, dust):
    """[start, end) of the clusters left after dropping clusters from both ends while dust(cluster) holds."""
    lo, hi = 0, len(starts) - 1
    while lo < hi and dust(lo):
        lo += 1
    while hi > lo and dust(hi):
        hi -= 1
    return starts[lo], ends[hi]


def ink_box(img: Image.Image):
    """(x0, y0, x1, y1) of the ink plus a small margin; the whole image if it has no ink.

    Specks of dust on a scan are left out: clusters at the edge with under 1% of the ink, more than the
    formula's height (the middle 90% of its ink rows) away from the rest and off the formula's line. A full
    stop a \\quad after a formula stays: it sits on the line."""
    ink = np.asarray(img) < INK
    if not ink.any():
        return 0, 0, ink.shape[1], ink.shape[0]
    x0, y0, x1, y1 = 0, 0, ink.shape[1], ink.shape[0]
    while True:
        win = ink[y0:y1, x0:x1]
        rows, cols = win.sum(1), win.sum(0)
        cum = np.cumsum(rows)
        line_top, line_bottom = np.searchsorted(cum, 0.05 * cum[-1]), np.searchsorted(cum, 0.95 * cum[-1])
        height = line_bottom - line_top + 1
        s, e, share = _clusters(rows, height)
        top, bottom = _trim(s, e, lambda i: share[i] < SPECK)  # far above or below: off the line by definition
        s, e, share = _clusters(cols, height)

        def off_line(i):  # beside the formula, punctuation sits on its line; dust can be anywhere
            on = np.flatnonzero(win[:, s[i]:e[i]].any(1))
            return share[i] < SPECK and (on[-1] < line_top or on[0] > line_bottom)
        left, right = _trim(s, e, off_line)
        box = x0 + left, y0 + top, x0 + right, y0 + bottom
        if box == (x0, y0, x1, y1):
            break
        x0, y0, x1, y1 = box
    return (max(x0 - MARGIN, 0), max(y0 - MARGIN, 0),
            min(x1 + MARGIN, ink.shape[1]), min(y1 + MARGIN, ink.shape[0]))


def fit(w: int, h: int, max_h: int, max_w: int, max_up: float = 2.0):
    """Size that fits (max_h, max_w) keeping the aspect ratio, enlarging at most max_up times."""
    s = min(max_h / h, max_w / w, max_up)
    return max(1, round(w * s)), max(1, round(h * s))


def to_tensor(img: Image.Image) -> torch.Tensor:
    """(1, H, W) float: ink 1, paper 0."""
    return torch.from_numpy(255 - np.asarray(img, dtype=np.uint8)).float().div_(255).unsqueeze(0)


def preprocess(img: Image.Image, max_h: int, max_w: int, box=None) -> torch.Tensor:
    """Inference path: grayscale, crop to ink, fit. Training adds augmentation (see FormulaDataset)."""
    img = img.convert("L")
    img = img.crop(box or ink_box(img))
    return to_tensor(img.resize(fit(*img.size, max_h, max_w), Image.BILINEAR))


class FormulaDataset(torch.utils.data.Dataset):
    """Shard rows -> (image tensor, token ids). index: (N, 5) int array of crop box x0, y0, x1, y1 and
    token count, from scripts/index_shards.py."""

    def __init__(self, rows, index, tokenizer, max_h, max_w, train):
        self.rows, self.index, self.tokenizer = rows, index, tokenizer
        self.max_h, self.max_w, self.train = max_h, max_w, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[int(i)]
        img = Image.open(io.BytesIO(r["image"])).convert("L").crop(tuple(int(v) for v in self.index[i, :4]))
        size = fit(*img.size, self.max_h, self.max_w)
        if self.train:
            img = augment(img, size, np.random.default_rng())
        else:
            img = img.resize(size, Image.BILINEAR)
        return to_tensor(img), torch.tensor(self.tokenizer.encode(r["label"]).ids)


def collate(batch):
    """Pad images to the batch's largest size, rounded up to a multiple of 16, and token ids with PAD_ID."""
    h = -(-max(b[0].shape[1] for b in batch) // 16) * 16
    w = -(-max(b[0].shape[2] for b in batch) // 16) * 16
    images = torch.zeros(len(batch), 1, h, w)
    for k, (img, _) in enumerate(batch):
        images[k, :, : img.shape[1], : img.shape[2]] = img
    tokens = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=PAD_ID)
    return images, tokens


class SizeBatchSampler:
    """Endless random batches of similar-sized samples, each under a pixel and a token budget.

    sizes: (N, 3) int array of fitted height, width and token count. Samples are shuffled, then sorted by
    size within large chunks so batches pad little while their order stays random."""

    def __init__(self, sizes, indices, max_pixels, max_tokens, max_batch, seed=0, chunk=50_000):
        self.sizes, self.indices = sizes, np.asarray(indices)
        self.max_pixels, self.max_tokens, self.max_batch = max_pixels, max_tokens, max_batch
        self.seed, self.chunk = seed, chunk

    def batches(self, order):
        h, w, t = self.sizes[:, 0], self.sizes[:, 1], self.sizes[:, 2]
        order = order[np.lexsort((w[order], h[order] // 32))]  # height bands, then width
        batch, mh, mw, mt = [], 0, 0, 0
        for i in order:
            nh, nw, nt = max(mh, h[i]), max(mw, w[i]), max(mt, t[i])
            n = len(batch) + 1
            if batch and (n * nh * nw > self.max_pixels or n * nt > self.max_tokens or n > self.max_batch):
                yield batch
                batch, nh, nw, nt = [], h[i], w[i], t[i]
            batch.append(int(i))
            mh, mw, mt = nh, nw, nt
        if batch:
            yield batch

    def __iter__(self):
        epoch = 0
        while True:
            rng = np.random.default_rng(self.seed + epoch)
            order = rng.permutation(self.indices)
            for start in range(0, len(order), self.chunk):
                group = list(self.batches(order[start:start + self.chunk]))
                for k in rng.permutation(len(group)):
                    yield group[k]
            epoch += 1
