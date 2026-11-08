"""Read MathWriting / CROHME InkML (pen strokes) and render it to a grayscale image."""
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

_NS = "{http://www.w3.org/2003/InkML}"


def read_inkml(path):
    """Return (strokes, annotations): strokes as (N, 2) float arrays of x, y; annotations as a dict."""
    root = ET.parse(path).getroot()
    ann = {a.get("type"): (a.text or "") for a in root.iter(f"{_NS}annotation")}
    strokes = []
    for trace in root.iter(f"{_NS}trace"):
        pts = [p.split() for p in (trace.text or "").strip().split(",") if p.strip()]
        if pts:
            strokes.append(np.array([[float(p[0]), float(p[1])] for p in pts]))
    return strokes, ann


def render_ink(strokes, char_px: float = 28.0, width_px: float = 2.5, pad_px: int = 8,
               supersample: int = 4, max_px: int = 2048) -> np.ndarray:
    """Draw strokes black on white. Scale so the median stroke is `char_px` across (its larger side, so
    a flat '-' still counts), which evens out how large different writers write; draw at `supersample`x
    and downsample for anti-aliasing. The longer image side is capped at `max_px`."""
    extents = [np.ptp(s, axis=0).max() for s in strokes if len(s) > 1]
    pts = np.concatenate(strokes)
    origin = pts.min(0)
    span = max(float(np.ptp(pts, axis=0).max()), 1e-6)
    scale = char_px / max(float(np.median(extents)) if extents else span, 1e-6)
    scale = min(scale, (max_px - 2 * pad_px) / span)
    size = np.ceil((pts.max(0) - origin) * scale) + 2 * pad_px
    ss = supersample
    img = Image.new("L", (int(size[0]) * ss, int(size[1]) * ss), 255)
    draw = ImageDraw.Draw(img)
    w = max(1, round(width_px * ss))
    for s in strokes:
        xy = ((s - origin) * scale + pad_px) * ss
        if len(xy) == 1:  # a dot
            x, y = xy[0]
            draw.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=0)
        else:
            draw.line([tuple(p) for p in xy], fill=0, width=w, joint="curve")
    return np.asarray(img.resize((img.width // ss, img.height // ss), Image.LANCZOS))
