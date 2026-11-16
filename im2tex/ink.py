"""Read pen strokes (MathWriting / CROHME InkML, or the ink in a PDF such as OneNote's export) and render them to
a grayscale image; find the lines of writing on a PDF page from its strokes."""
import ctypes
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pypdfium2.raw as pdfium_c
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
               supersample: int = 4, max_px: int = 2048, filled: bool = False) -> np.ndarray:
    """Draw strokes black on white. Scale so the median stroke is `char_px` across (its larger side, so
    a flat '-' still counts), which evens out how large different writers write; draw at `supersample`x
    and downsample for anti-aliasing. The longer image side is capped at `max_px`. With `filled`, each
    stroke is a closed outline to fill (PDF ink, which already has its width) instead of a pen path."""
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
        if filled:
            if len(xy) > 2:
                draw.polygon([tuple(p) for p in xy], fill=0)
        elif len(xy) == 1:  # a dot
            x, y = xy[0]
            draw.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=0)
        else:
            draw.line([tuple(p) for p in xy], fill=0, width=w, joint="curve")
    return np.asarray(img.resize((img.width // ss, img.height // ss), Image.LANCZOS))


# PDF ink. OneNote's PDF export keeps every pen stroke as a filled outline, in the order it was written, which
# says which strokes belong together: a fraction's numerator, bar and denominator are written one after another.
# Distances below are in units of h, the median stroke height on the page (about a letter's height).
JOIN_X, BACK_X, JOIN_Y = 6.0, 3.0, 0.8  # the next stroke continues the piece within these to the right/left/up-down
MERGE_X = 3.0  # pieces written at different times merge when they share a row with at most this gap
FRAGMENT_STROKES, FRAGMENT_W, FRAGMENT_H, FRAGMENT_GAP = 6, 2.5, 4.0, 1.5  # limits (a fraction too), exponents added later
BAR_W, BAR_FLAT, BAR_GAP = 1.5, 0.35, 1.2  # a fraction bar written apart: this wide, flat, its terms this close
SHARE_X = 0.5  # a line written later goes before an earlier one below it when they share this much width


@dataclass
class PdfStroke:
    outlines: list  # closed polygons, (N, 2) arrays of x, y in pt with y down
    box: tuple  # x0, y0, x1, y1
    clipped: bool  # reaches past its clip rectangle: OneNote cuts ink at its page breaks this way


def _clipped(obj) -> bool:
    clip = pdfium_c.FPDFPageObj_GetClipPath(obj)
    if not clip:
        return False
    ys = []
    for p in range(pdfium_c.FPDFClipPath_CountPaths(clip)):
        for s in range(pdfium_c.FPDFClipPath_CountPathSegments(clip, p)):
            x, y = ctypes.c_float(), ctypes.c_float()
            pdfium_c.FPDFPathSegment_GetPoint(pdfium_c.FPDFClipPath_GetPathSegment(clip, p, s),
                                              ctypes.byref(x), ctypes.byref(y))
            ys.append(y.value)
    left, bottom, right, top = (ctypes.c_float() for _ in range(4))
    pdfium_c.FPDFPageObj_GetBounds(obj, *map(ctypes.byref, (left, bottom, right, top)))
    return bool(ys) and (bottom.value < min(ys) - 0.5 or top.value > max(ys) + 0.5)


def pdf_strokes(page) -> list[PdfStroke]:
    """The ink of a pypdfium2 page in drawing order: filled paths, dark enough to be writing (a highlighter's
    light fill is left out). Typed text and pictures are not strokes."""
    _, height = page.get_size()
    out = []
    for o in page.get_objects(max_depth=5):
        if o.type != pdfium_c.FPDF_PAGEOBJ_PATH:
            continue
        r, g, b, a = (ctypes.c_uint() for _ in range(4))
        pdfium_c.FPDFPageObj_GetFillColor(o.raw, *map(ctypes.byref, (r, g, b, a)))
        if 0.299 * r.value + 0.587 * g.value + 0.114 * b.value > 160 or a.value < 128:
            continue
        m, outlines, cur = o.get_matrix(), [], []
        for k in range(pdfium_c.FPDFPath_CountSegments(o.raw)):
            seg = pdfium_c.FPDFPath_GetPathSegment(o.raw, k)
            x, y = ctypes.c_float(), ctypes.c_float()
            pdfium_c.FPDFPathSegment_GetPoint(seg, ctypes.byref(x), ctypes.byref(y))
            if pdfium_c.FPDFPathSegment_GetType(seg) == pdfium_c.FPDF_SEGMENT_MOVETO and cur:
                outlines.append(np.array(cur))
                cur = []
            cur.append((m.a * x.value + m.c * y.value + m.e, height - (m.b * x.value + m.d * y.value + m.f)))
        if cur:
            outlines.append(np.array(cur))
        if outlines:
            pts = np.concatenate(outlines)
            out.append(PdfStroke(outlines, (*pts.min(0), *pts.max(0)), _clipped(o.raw)))
    return out


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _merge_rows(pieces, h):
    """Merge pieces that share a row with a small gap, or where one sits inside the other."""
    merged = True
    while merged:
        merged = False
        for a in range(len(pieces)):
            for b in range(a + 1, len(pieces)):
                A, B = pieces[a][1], pieces[b][1]
                overlap = min(A[3], B[3]) - max(A[1], B[1])
                gap = max(A[0], B[0]) - min(A[2], B[2])
                inside = any(X[0] <= (Y[0] + Y[2]) / 2 <= X[2] and X[1] <= (Y[1] + Y[3]) / 2 <= X[3]
                             for X, Y in ((A, B), (B, A)))
                if inside or (overlap > 0.5 * min(A[3] - A[1], B[3] - B[1]) and gap < MERGE_X * h):
                    pieces[a] = [pieces[a][0] + pieces[b][0], _union(A, B)]
                    del pieces[b]
                    merged = True
                    break
            if merged:
                break


def _join_bars(pieces, h) -> bool:
    """A piece that is a lone flat stroke, a fraction bar written apart from its terms, takes the pieces right
    above and below it whose middle is over it. Returns whether any bar did."""
    for k, (idx, (x0, y0, x1, y1)) in enumerate(pieces):
        if len(idx) != 1 or x1 - x0 < BAR_W * h or y1 - y0 > BAR_FLAT * (x1 - x0):
            continue
        terms = [m for m, (_, b) in enumerate(pieces) if m != k
                 and x0 - 0.3 * h <= (b[0] + b[2]) / 2 <= x1 + 0.3 * h
                 and (0 <= y0 - b[3] <= BAR_GAP * h or 0 <= b[1] - y1 <= BAR_GAP * h)]
        if any(pieces[m][1][3] <= y0 for m in terms) and any(pieces[m][1][1] >= y1 for m in terms):
            for m in sorted(terms, reverse=True):
                pieces[k] = [pieces[k][0] + pieces[m][0], _union(pieces[k][1], pieces[m][1])]
            for m in sorted(terms, reverse=True):
                del pieces[m]
            return True
    return False


def _reading_order(lines) -> list[list[int]]:
    """Lines ([stroke indices, box]) in the order they were started, except that a line written later into space
    left above earlier ones goes before the first of them that is below it and shares its width."""
    placed = []
    for idx, (x0, y0, x1, y1) in sorted(lines, key=lambda p: min(p[0])):
        at = next((k for k, (_, b) in enumerate(placed)
                   if b[1] >= y1 and min(x1, b[2]) - max(x0, b[0]) > SHARE_X * min(x1 - x0, b[2] - b[0])),
                  len(placed))
        placed.insert(at, (sorted(idx), (x0, y0, x1, y1)))
    return [idx for idx, _ in placed]


def find_lines(boxes) -> list[list[int]]:
    """Group strokes, given as boxes in writing order, into lines of writing -> stroke indices per line, in
    reading order: the order the lines were started (notes are written top to bottom, a column at a time), but
    a line squeezed in later above earlier ones comes before them.

    A stroke written right after the previous ones, close to them, continues their piece. Pieces written at
    different times merge when they share a row with a small gap, or one sits inside the other; a fraction bar
    written on its own joins its terms; a few narrow strokes just above or below a line (limits, exponents, a
    numerator added later) join it. Pieces far apart on one row stay separate lines (a side note like u = x^2
    next to an integral)."""
    if not boxes:
        return []
    hs = np.array([b[3] - b[1] for b in boxes])
    h = float(np.median(hs[hs > 1.0])) if (hs > 1.0).any() else 5.0
    pieces = []  # [stroke indices, box]
    for i, b in enumerate(boxes):
        if pieces:
            box = pieces[-1][1]
            if (b[1] < box[3] + JOIN_Y * h and b[3] > box[1] - JOIN_Y * h
                    and b[0] < box[2] + JOIN_X * h and b[2] > box[0] - BACK_X * h):
                pieces[-1] = [pieces[-1][0] + [i], _union(box, b)]
                continue
        pieces.append([[i], b])
    _merge_rows(pieces, h)
    while _join_bars(pieces, h):
        pass
    _merge_rows(pieces, h)

    def fragment(p):
        x0, y0, x1, y1 = p[1]
        return len(p[0]) <= FRAGMENT_STROKES and x1 - x0 <= FRAGMENT_W * h and y1 - y0 <= FRAGMENT_H * h

    def gaps(f, p):
        (fx0, fy0, fx1, fy1), (x0, y0, x1, y1) = f[1], p[1]
        return max(x0 - fx1, fx0 - x1, 0.0), max(y0 - fy1, fy0 - y1, 0.0)

    lines = [p for p in pieces if not fragment(p)]
    for f in [p for p in pieces if fragment(p)]:
        near = [p for p in lines if max(gaps(f, p)) <= FRAGMENT_GAP * h]
        if near:
            host = min(near, key=lambda p: sum(gaps(f, p)))
            host[0], host[1] = host[0] + f[0], _union(host[1], f[1])
        else:
            lines.append(f)
    return _reading_order(lines)


def page_lines(page) -> list[list[PdfStroke]]:
    """The lines of handwriting on a PDF page in reading order, each as its strokes. Lines cut by a page break
    are left out: their other half is on the neighbouring page."""
    strokes = pdf_strokes(page)
    lines = [[strokes[i] for i in line] for line in find_lines([s.box for s in strokes])]
    return [line for line in lines if not any(s.clipped for s in line)]
