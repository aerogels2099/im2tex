"""Training-time image augmentation: make clean renders look like screenshots, scans, slides and phone photos.

Each row's size is drawn before batches are planned (draw_scales), so the batch memory budget holds. Geometry,
stroke and margin changes happen at the source resolution and the result is fitted inside the planned size.
Then the resolution-level damage (binarization, contrast, paper, lines, lighting, specks, JPEG, noise) is added,
often at a lower resolution and enlarged back: what a small scan looks like after test-time enlarging.
"""
import io

import numpy as np
from PIL import Image, ImageFilter

# Median symbol height (connected ink components) of each source in training inputs at u = 1: the inference size
# (fitted, enlarged up to 2x) shrunk by augment's margins, measured on 400 rows per source. Sources not listed are
# handwritten.
PRINTED_SYMBOL_PX = {"unimer": 15, "synthetic_printed": 21}
HANDWRITTEN_SYMBOL_PX = 41
# Handwritten sources that come out smaller. HME100K's photos: about 35 px at the inference size (threshold halfway
# between ink and paper, as the pencil is faint), about 30 px after the margins. IAM's text lines and the mixed
# lines made from them are long, so fitting them 1024 px wide shrinks them: 27.5 and 33.5 px at the inference
# size (the margins barely matter on a wide line).
SMALLER_HANDWRITTEN_SYMBOL_PX = {"hme100k": 30, "iam": 27, "mixed_lines": 32}


def draw_scales(sources, rng: np.random.Generator) -> np.ndarray:
    """Per training row, a factor on its inference size. 60%: 0.5 to 1.0, as always. 40%: log-uniform, so the
    source's median symbol lands anywhere from 8 px (printed) or 12 px (handwritten) up to 60 px. Without that,
    size alone tells printed from handwritten, and printed slides (30-36 px symbols) are bigger than any
    printed training image."""
    names, inverse = np.unique(np.asarray(sources), return_inverse=True)
    printed = np.isin(names, list(PRINTED_SYMBOL_PX))[inverse]
    px = np.array([PRINTED_SYMBOL_PX.get(s, SMALLER_HANDWRITTEN_SYMBOL_PX.get(s, HANDWRITTEN_SYMBOL_PX))
                   for s in names], dtype=np.float64)[inverse]
    lo, hi = np.where(printed, 8.0, 12.0) / px, 60.0 / px
    wide = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    return np.where(rng.random(len(px)) < 0.4, wide, rng.uniform(0.5, 1.0, len(px)))


def augment(img: Image.Image, size, rng: np.random.Generator) -> Image.Image:
    """img: grayscale, cropped to its ink. size: (w, h) planned for it; the result fits inside it."""
    # Thicker strokes (the min filter spreads dark). Never thinner: training strokes are 3 px or less at the
    # source resolution, so a max filter would wipe out fraction bars and minus signs and leave the label wrong.
    if rng.random() < 0.1:
        img = img.filter(ImageFilter.MinFilter(3))
    if rng.random() < 0.3:  # slight rotation and shear, as in photos and scans; nothing leaves the frame
        img = img.rotate(rng.uniform(-2, 2), Image.BILINEAR, expand=True, fillcolor=255)
        sh = rng.uniform(-0.1, 0.1)
        shift = sh * img.height if sh > 0 else 0
        img = img.transform((img.width + int(abs(sh) * img.height) + 1, img.height), Image.AFFINE,
                            (1, sh, -shift, 0, 1, 0), Image.BILINEAR, fillcolor=255)
    img = _loose_crop(img, rng)
    w, h = _fit_inside(img.size, size)
    img = img.resize((w, h), Image.BILINEAR)
    # A low-resolution image enlarged to this size: the damage happens small, then bilinear back up. Never below
    # 16 px tall, so the symbols stay legible.
    low = rng.random() < 0.4 and h > 20
    if low:
        f = rng.uniform(max(0.4, 16 / h), 0.85)
        img = img.resize((max(4, round(w * f)), round(h * f)), Image.BILINEAR)
    img = _damage(img, rng)
    return img.resize((w, h), Image.BILINEAR) if low else img


def _loose_crop(img, rng):
    """Margins, since real crops are rarely tight; sometimes the cut-off edge of a neighboring line shows above or
    below (a slice of the formula itself, in the margin)."""
    pad = rng.integers(0, 12, size=4)  # left, top, right, bottom
    near = rng.random() < 0.15
    if near:
        fh = min(img.height, int(rng.integers(max(2, img.height // 10), max(3, img.height // 3) + 1)))
        above = rng.random() < 0.5
        pad[1 if above else 3] = max(pad[1 if above else 3], fh + 2)
    canvas = Image.new("L", (img.width + pad[0] + pad[2], img.height + pad[1] + pad[3]), 255)
    canvas.paste(img, (int(pad[0]), int(pad[1])))
    if near:
        a, src = np.array(canvas), np.asarray(img)
        piece = src[-fh:] if above else src[:fh]  # the line above shows its bottom, the line below its top
        x = int(rng.integers(0, a.shape[1]))
        piece = piece[:, : a.shape[1] - x]
        rows = slice(0, fh) if above else slice(a.shape[0] - fh, a.shape[0])
        a[rows, x : x + piece.shape[1]] = np.minimum(a[rows, x : x + piece.shape[1]], piece)
        canvas = Image.fromarray(a)
    return canvas


def _damage(img, rng):
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape
    if rng.random() < 0.15:  # hard black/white strokes, like CROHME; split halfway to the darkest ink, so faint
        a = np.where(a < (a.min() + 255) / 2, 0.0, 255.0)  # (small, blurred) strokes don't vanish
    if rng.random() < 0.3:  # gray paper or slide background, gray ink or low-contrast font
        paper, ink = rng.uniform(170, 255), rng.uniform(0, 90)
        a = ink + (paper - ink) * a / 255
    if rng.random() < 0.12:  # ruled or grid lines under the ink, as on notepaper
        a = np.minimum(a, _lines(h, w, rng))
    if rng.random() < 0.05:  # faint bleed-through from the other side of the page
        a = np.minimum(a, 255 - (255 - a[:, ::-1]) * rng.uniform(0.05, 0.2))
    if rng.random() < 0.2:  # paper texture: smooth darker blotches
        small = rng.normal(0, 1, (max(2, h // 16), max(2, w // 16))).astype(np.float32)
        a -= np.abs(np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR))) * rng.uniform(5, 25)
    if rng.random() < 0.2:  # uneven lighting or a slide's gradient: darker towards one side
        t = rng.uniform(0, 2 * np.pi)
        ramp = np.cos(t) * np.arange(w)[None, :] / w + np.sin(t) * np.arange(h)[:, None] / h
        ramp = (ramp - ramp.min()) / max(ramp.max() - ramp.min(), 1e-6)
        a *= 1 - rng.uniform(0.1, 0.4) * ramp
    if rng.random() < 0.1:  # specks, as on scans
        n = int(rng.integers(1, max(2, h * w // 2000) + 1))
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        a[ys, xs] = np.minimum(a[ys, xs], rng.uniform(0, 150, n))
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.3:  # JPEG artifacts, common in screenshots
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(rng.integers(30, 90)))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    if rng.random() < 0.3:
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, rng.uniform(3, 15), (h, w))
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    return img


def _lines(h, w, rng):
    """Light ruled lines, sometimes a grid, at a random spacing and phase (white elsewhere)."""
    out = np.full((h, w), 255.0, np.float32)
    # Spaced like lines of writing: 0 to 2 lines cross a one-line crop.
    gap, level, width = max(6.0, rng.uniform(0.6, 1.5) * h), rng.uniform(150, 225), int(rng.integers(1, 3))
    for y in np.arange(rng.uniform(0, gap), h, gap):
        out[int(y) : int(y) + width] = level
    if rng.random() < 0.3:
        for x in np.arange(rng.uniform(0, gap), w, gap):
            out[:, int(x) : int(x) + width] = level
    return out


def _fit_inside(src, box):
    """Largest size with src's aspect ratio inside box (w, h); at least 8 px per side."""
    s = min(box[0] / src[0], box[1] / src[1])
    return max(8, min(round(src[0] * s), int(box[0]))), max(8, min(round(src[1] * s), int(box[1])))
