"""Write the common split layout for GNHK's handwritten English text lines (train / validation / test), cut from
the camera-captured pages that scripts/download_data.sh fetches.

GNHK labels words: a four-point outline (in the photo's displayed orientation), the text, a line number and H
(handwritten) or P (printed). A line is cut straightened: its angle is the line through its word centers (the
page's median for one-word lines), the crop is rotated to that angle and padded, and what lies outside the
words' boxes is painted the paper's color so neighboring lines don't show. Labels are the words left to right, with stand-alone punctuation
attached and $ written \\$ (the page format). Skipped: sideways pages, lines with printed words, unreadable
(%NA%) or untranscribed math (%math%) words, math-like
words (=, +, ->, ...; page transcriptions would put them in $...$), lines tilted more than 30 degrees from
their page, and curved or misread lines (taller than 2.5 word heights once straightened). Crossed-out words
(%SC%) stay in the image and are left out of the label, as page transcriptions do. GNHK has no validation
split: 10% of its train pages (by a hash of the file name) are held out, whole pages, so no writer is in both.
    python scripts/prepare_gnhk.py
"""
import hashlib
import io
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFilter, ImageOps

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data")) / "gnhk"
FILES = {"train": ["train-00000-of-00002", "train-00001-of-00002"], "test": ["test-00000-of-00001"]}
MAX_H = 192  # stored line height cap: the pages are 4128 px wide and lines are ~150 px high
MATH_LIKE = re.compile(r"^(=|\+|\*|\^|_|<|>|->|<-|=>|\{|\})$|=")


def label(words: list[str]) -> str | None:
    """Words of one line, left to right -> its label, or None when the line is skipped."""
    if any(w in ("%NA%", "%math%") or MATH_LIKE.search(w) for w in words):
        return None
    words = [w.replace("$", r"\$") for w in words if w != "%SC%"]
    text = re.sub(r" ([.,;:?!)])", r"\1", " ".join(words))
    return re.sub(r"\( ", "(", text) or None


def corners(word) -> np.ndarray:
    p = word["polygon"]
    return np.array([[p[f"x{k}"], p[f"y{k}"]] for k in range(4)], dtype=np.float64)


def direction(words) -> float | None:
    """Angle (radians, image coordinates) of the line through the words' centers, pointing right: the pages are
    stored upright, but GNHK sometimes lists a line's words in reverse. None for one word."""
    if len(words) < 2:
        return None
    centers = np.array([corners(w).mean(0) for w in words])
    dx, dy = np.linalg.svd(centers - centers.mean(0))[2][0]
    return math.atan2(dy, dx) if dx >= 0 else math.atan2(-dy, -dx)


def along(words, angle: float):
    """-> function mapping image points to line coordinates (u along the line, v across), centered on the words."""
    center = np.concatenate([corners(w) for w in words]).mean(0)
    c, s = math.cos(angle), math.sin(angle)
    return lambda q: np.stack([c * (q[:, 0] - center[0]) + s * (q[:, 1] - center[1]),
                               -s * (q[:, 0] - center[0]) + c * (q[:, 1] - center[1])], 1)


def left_to_right(words, angle: float) -> list:
    to_line = along(words, angle)
    return sorted(words, key=lambda w: to_line(corners(w)).mean(0)[0])


def cut(image: Image.Image, words, angle: float, pad: float = 0.15, keep: float = 0.25) -> Image.Image | None:
    """Crop the words' line rotated by -angle, padded by `pad` word heights. Outside the words' boxes grown by
    `keep` word heights it is painted the paper's color (soft edged), so the lines above and below don't show.
    None when the straightened line is more than 2.5 word heights tall (curved, or a misread angle)."""
    pts = np.concatenate([corners(w) for w in words])
    center = pts.mean(0)
    c, s = math.cos(angle), math.sin(angle)
    to_line = along(words, angle)
    local = to_line(pts)
    word_h = float(np.median([np.ptp(to_line(corners(w))[:, 1]) for w in words]))
    u0, v0 = local.min(0) - pad * word_h
    u1, v1 = local.max(0) + pad * word_h
    if v1 - v0 > (2.5 + 2 * pad) * word_h:
        return None
    w, h = int(round(u1 - u0)), int(round(v1 - v0))
    # output (u, v) -> input x = cx + c (u0 + u) - s (v0 + v), y = cy + s (u0 + u) + c (v0 + v)
    data = (c, -s, center[0] + c * u0 - s * v0, s, c, center[1] + s * u0 + c * v0)
    crop = image.transform((w, h), Image.AFFINE, data, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    for word in words:
        q = to_line(corners(word)) - (u0, v0)
        (a, b), (c_, d) = q.min(0) - keep * word_h, q.max(0) + keep * word_h
        draw.rectangle((a, b, c_, d), fill=255)
    pixels = np.asarray(crop).reshape(-1, 3)
    bright = pixels.mean(1) >= np.percentile(pixels.mean(1), 75)  # paper, even on a line that is mostly ink
    paper = Image.new("RGB", crop.size, tuple(int(x) for x in np.median(pixels[bright], 0)))
    crop = Image.composite(crop, paper, mask.filter(ImageFilter.GaussianBlur(keep * word_h / 3)))
    if crop.height > MAX_H:
        crop = crop.resize((max(1, round(crop.width * MAX_H / crop.height)), MAX_H), Image.LANCZOS)
    return crop


def page_lines(annotations: str):
    """-> {line index: words}, keeping the annotation order (reading order)."""
    lines = defaultdict(list)
    for w in json.loads(annotations):
        lines[w["line_idx"]].append(w)
    return lines


def held_out(filename: str) -> bool:
    return int(hashlib.md5(filename.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.1


if __name__ == "__main__":
    rows = defaultdict(list)
    skipped = defaultdict(int)
    for source, files in FILES.items():
        for f in files:
            for r in pq.read_table(DATA / f"{f}.parquet").to_pylist():
                split = "test" if source == "test" else "validation" if held_out(r["filename"]) else "train"
                lines = page_lines(r["annotations"])
                angles = [a for a in map(direction, lines.values()) if a is not None]
                page_angle = float(np.median(angles)) if angles else 0.0
                if abs(math.degrees(page_angle)) > 30:
                    skipped["sideways page"] += 1
                    continue
                image = None
                stem = Path(r["filename"]).stem
                for idx, words in sorted(lines.items()):
                    if any(w["type"] != "H" for w in words):
                        skipped["printed words"] += 1
                        continue
                    angle = direction(words)
                    if angle is None:
                        angle = page_angle
                    elif abs(math.degrees(angle - page_angle)) > 30:
                        skipped["line tilted"] += 1
                        continue
                    ref = label([w["text"] for w in left_to_right(words, angle)])
                    if ref is None:
                        skipped["unreadable, math or math-like"] += 1
                        continue
                    if image is None:
                        # the outlines are in the photo's displayed orientation (EXIF rotation applied)
                        image = ImageOps.exif_transpose(Image.open(io.BytesIO(r["image"]["bytes"]))).convert("RGB")
                    crop = cut(image, words, angle)
                    if crop is None:
                        skipped["curved"] += 1
                        continue
                    row_id = f"{stem}-{idx:03d}"
                    out = DATA / split / "images" / f"{row_id}.jpg"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(out, quality=90)
                    rows[split].append({"id": row_id, "image": f"images/{row_id}.jpg", "ref": ref, "norm": ref})
    for split, rs in rows.items():
        with (DATA / split / "refs.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
        print(f"{split}: {len(rs)} lines -> {DATA / split}")
    print("skipped:", dict(skipped))
