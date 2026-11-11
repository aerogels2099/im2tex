"""Training-time image augmentation: make clean renders look like screenshots, scans and phone photos.

Geometry and stroke changes happen at the source resolution; the result is resized to fit inside the size
the batch sampler planned for (randomly smaller), so augmentation never breaks the batch memory budget;
then resolution-level damage (binarization, contrast, blur, JPEG, noise) is added.
"""
import io

import numpy as np
from PIL import Image, ImageFilter


def augment(img: Image.Image, size, rng: np.random.Generator) -> Image.Image:
    """img: grayscale, cropped to its ink. size: (w, h) the sampler planned; the result fits inside it."""
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
    u = rng.uniform(0.5, 1.0)  # often smaller than planned, so the model sees many text sizes
    w, h = _fit_inside(canvas.size, (size[0] * u, size[1] * u))
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
