"""Training-time image augmentation: make clean renders look like screenshots, scans and phone photos.

Each row's size is drawn before batches are planned (draw_scales), so the batch memory budget holds. Geometry,
stroke and margin changes happen at the source resolution and the result is fitted inside the planned size;
then resolution-level damage (binarization, contrast, blur, JPEG, noise) is added.
"""
import io

import numpy as np
from PIL import Image, ImageFilter

# Median symbol height (connected ink components) of each source in training inputs at u = 1: the inference size
# (fitted, enlarged up to 2x) shrunk by augment's margins, measured on 400 rows per source. Sources not listed are
# handwritten.
PRINTED_SYMBOL_PX = {"unimer": 15, "synthetic_printed": 21}
HANDWRITTEN_SYMBOL_PX = 41


def draw_scales(sources, rng: np.random.Generator) -> np.ndarray:
    """Per training row, a factor on its inference size. 60%: 0.5 to 1.0, as always. 40%: log-uniform, so the
    source's median symbol lands anywhere from 8 px (printed) or 12 px (handwritten) up to 60 px. Without that,
    size alone tells printed from handwritten, and printed slides (30-36 px symbols) are bigger than any
    printed training image."""
    names, inverse = np.unique(np.asarray(sources), return_inverse=True)
    printed = np.isin(names, list(PRINTED_SYMBOL_PX))[inverse]
    px = np.array([PRINTED_SYMBOL_PX.get(s, HANDWRITTEN_SYMBOL_PX) for s in names], dtype=np.float64)[inverse]
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
    pad = rng.integers(0, 12, size=4)  # margins, since real crops are rarely tight
    canvas = Image.new("L", (img.width + pad[0] + pad[2], img.height + pad[1] + pad[3]), 255)
    canvas.paste(img, (int(pad[0]), int(pad[1])))
    w, h = _fit_inside(canvas.size, size)
    img = canvas.resize((w, h), Image.BILINEAR)

    a = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.15:  # hard black/white strokes, like CROHME
        a = np.where(a < 128, 0.0, 255.0)
    if rng.random() < 0.3:  # gray paper, gray ink
        paper, ink = rng.uniform(170, 255), rng.uniform(0, 90)
        a = ink + (paper - ink) * a / 255
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if rng.random() < 0.25:  # soft focus: down and back up
        f = rng.uniform(0.5, 0.8)
        img = img.resize((max(4, round(w * f)), max(4, round(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if rng.random() < 0.3:  # JPEG artifacts, common in screenshots
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(rng.integers(30, 90)))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    if rng.random() < 0.3:
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, rng.uniform(3, 15), (h, w))
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    return img


def _fit_inside(src, box):
    """Largest size with src's aspect ratio inside box (w, h); at least 8 px per side."""
    s = min(box[0] / src[0], box[1] / src[1])
    return max(8, min(round(src[0] * s), int(box[0]))), max(8, min(round(src[1] * s), int(box[1])))
