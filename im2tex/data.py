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
from im2tex.tokenizer import LINE

INK = 200  # pixels darker than this count as ink when cropping
MARGIN = 4  # pixels kept around the ink
PAD_ID = 0  # tokenizer's <pad>
# Sources whose rows are lines of words and math, read from <line> instead of <bos>.
LINE_SOURCES = frozenset({"iam", "iam_valid", "mixed_lines", "mixed_lines_valid", "gnhk", "gnhk_valid"})


def ink_box(img: Image.Image):
    """(x0, y0, x1, y1) of the ink plus a small margin; the whole image if it has no ink."""
    a = np.asarray(img)
    ink = a < INK
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    if rows.size == 0:
        return 0, 0, a.shape[1], a.shape[0]
    return (max(cols[0] - MARGIN, 0), max(rows[0] - MARGIN, 0),
            min(cols[-1] + MARGIN + 1, a.shape[1]), min(rows[-1] + MARGIN + 1, a.shape[0]))


def fit(w: int, h: int, max_h: int, max_w: int, max_up: float = 2.0):
    """Size that fits (max_h, max_w) keeping the aspect ratio, enlarging at most max_up times."""
    s = min(max_h / h, max_w / w, max_up)
    return max(1, round(w * s)), max(1, round(h * s))


def train_scale(w, h, max_h: int, max_w: int, u):
    """Training scale: the inference one (fit, enlarging up to 2x) times u, but never past (max_h, max_w).
    Takes numpy arrays too, for planning batches."""
    limit = np.minimum(max_h / h, max_w / w)
    return np.minimum(np.minimum(limit, 2.0) * u, limit)


def planned_sizes(index, max_h: int, max_w: int, scales=None):
    """(N, 3) for the batch sampler: height and width at train_scale, rounded up to multiples of 16, and token
    count. index: (N, 5) crop boxes and token counts; scales: per-row u (augment.draw_scales), None = 1."""
    w, h = index[:, 2] - index[:, 0], index[:, 3] - index[:, 1]
    s = train_scale(w, h, max_h, max_w, 1.0 if scales is None else scales)
    ceil16 = lambda x: (np.ceil(np.maximum(np.round(x), 1) / 16) * 16).astype(np.int64)  # noqa: E731
    return np.column_stack([ceil16(h * s), ceil16(w * s), index[:, 4]])


def to_tensor(img: Image.Image) -> torch.Tensor:
    """(1, H, W) float: ink 1, paper 0."""
    return torch.from_numpy(255 - np.asarray(img, dtype=np.uint8)).float().div_(255).unsqueeze(0)


def preprocess(img: Image.Image, max_h: int, max_w: int, box=None, max_up: float = 2.0) -> torch.Tensor:
    """Inference path: grayscale, crop to ink, fit. Training adds augmentation (see FormulaDataset)."""
    img = img.convert("L")
    img = img.crop(box or ink_box(img))
    return to_tensor(img.resize(fit(*img.size, max_h, max_w, max_up), Image.BILINEAR))


class FormulaDataset(torch.utils.data.Dataset):
    """Shard rows -> (image tensor, token ids). index: (N, 5) int array of crop box x0, y0, x1, y1 and
    token count, from scripts/index_shards.py. scales: per-row size factor when training (augment.draw_scales,
    the same array the batch sampler planned with); None = the inference size."""

    def __init__(self, rows, index, tokenizer, max_h, max_w, train, scales=None):
        self.rows, self.index, self.tokenizer = rows, index, tokenizer
        self.max_h, self.max_w, self.train, self.scales = max_h, max_w, train, scales

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[int(i)]
        img = Image.open(io.BytesIO(r["image"])).convert("L").crop(tuple(int(v) for v in self.index[i, :4]))
        if self.train:
            u = 1.0 if self.scales is None else self.scales[i]
            s = train_scale(img.width, img.height, self.max_h, self.max_w, u)
            img = augment(img, (max(1, round(img.width * s)), max(1, round(img.height * s))), np.random.default_rng())
        else:
            img = img.resize(fit(*img.size, self.max_h, self.max_w), Image.BILINEAR)
        ids = self.tokenizer.encode(r["label"]).ids
        if r.get("source") in LINE_SOURCES:
            ids[0] = self.tokenizer.token_to_id(LINE)
        return to_tensor(img), torch.tensor(ids)


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
