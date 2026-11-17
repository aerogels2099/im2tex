"""Read pen strokes (MathWriting / CROHME InkML, or the ink in a PDF such as OneNote's export) and render them to
a grayscale image; find the lines of writing on a PDF page from its strokes."""
import bisect
import ctypes
import xml.etree.ElementTree as ET
from collections import Counter
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
FRAC_W, FRAC_GAP = 2.0, 1.0  # a long fraction bar written with its numerator: this wide, its terms' strokes this close
SHARE_X = 0.5  # a line written later goes before an earlier one below it when they share this much width
COLUMN_W = 4.0  # lines this wide beside a line's own column make a column of work of their own (not a label)
# Tables (a truth table): a head, a long flat rule under it, then columns of cells; lines drawn across rows are
# marks, not cells.
RULE_W, RULE_FLAT = 6.0, 0.15  # a rule: this wide at least, drawn in flat pieces (height at most this of width)
HEAD_H, FIRST_ROW, ROW_GAP = 3.5, 2.0, 1.2  # the head's bottom within this above the rule, the first row's top
#                                             within this below it; the table ends at an empty gap this tall
COL_GAP, HEAD_GAP = 0.5, 0.6  # strokes this close across share a column; the head's titles are this far apart
DRAW_W = 4.0  # a drawing: level and upright lines at least this long crossing (writing has none)


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


def _join_fractions(pieces, boxes, h, points=None):
    """The pieces holding a long fraction bar and the strokes right above and right below it (its numerator and
    denominator, within its width) merge: a big fraction written in pieces is read as one. With the strokes'
    outline points, a bar that slopes is followed where it runs; without them, only a level one counts."""
    where = {i: k for k, (idx, _) in enumerate(pieces) for i in idx}
    root = list(range(len(pieces)))

    def find(k):
        while root[k] != k:
            k = root[k]
        return k

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if i not in where or x1 - x0 < FRAC_W * h or y1 - y0 > BAR_FLAT * (x1 - x0) or points is None and y1 - y0 > h:
            continue
        pts = points[i] if points is not None else np.array([[x0, y0], [x0, y1]])
        over = [j for j, b in enumerate(boxes) if j != i and j in where and x0 - 0.3 * h <= (b[0] + b[2]) / 2 <= x1 + 0.3 * h]
        at = {}  # the bar's top and bottom edge under each of them
        for j in over:
            ys = pts[np.argsort(np.abs(pts[:, 0] - (boxes[j][0] + boxes[j][2]) / 2))[:8], 1]
            at[j] = (float(ys.min()), float(ys.max()))
        up = [j for j in over if at[j][0] - FRAC_GAP * h <= boxes[j][3] <= sum(at[j]) / 2]
        down = [j for j in over if sum(at[j]) / 2 <= boxes[j][1] <= at[j][1] + FRAC_GAP * h]
        if up and down:
            for j in up + down:
                root[find(where[j])] = find(where[i])
    groups = {}
    for k, p in enumerate(pieces):
        groups.setdefault(find(k), []).append(p)
    pieces[:] = [[[i for p in ps for i in p[0]], (min(p[1][0] for p in ps), min(p[1][1] for p in ps),
                                                   max(p[1][2] for p in ps), max(p[1][3] for p in ps))]
                 for ps in groups.values()]


def _shares(a, b) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > SHARE_X * min(a[2] - a[0], b[2] - b[0])


def _reading_order(lines) -> list[list[int]]:
    """Lines ([stroke indices, box]) in the order they were started, except that a line written later into space
    left above earlier ones goes before the first of them that is below it and shares its width."""
    placed = []
    for idx, box in sorted(lines, key=lambda p: min(p[0])):
        at = next((k for k, (_, b) in enumerate(placed) if b[1] >= box[3] and _shares(box, b)), len(placed))
        placed.insert(at, (sorted(idx), box))
    return [idx for idx, _ in placed]


def column_order(boxes, h, movable) -> list[int]:
    """Lines' boxes in reading order -> their indices in a new order, where a line in `movable` (one that goes on
    a chain of steps: it starts with = or ≡) moves up to right after the line above it in its column (the nearest
    above that shares its width) when the lines between are all a column beside it (at least COLUMN_W wide,
    clear of its column's width) that had reached its height: the writing went back to an earlier column of
    work written side by side, and each column is read on its own."""
    order = list(range(len(boxes)))
    for _ in range(len(order) ** 2):
        for k, i in enumerate(order):
            box = boxes[i]
            above = [m for m in range(k) if _shares(boxes[order[m]], box)
                     and boxes[order[m]][1] + boxes[order[m]][3] < box[1] + box[3]]
            if i not in movable or not above or (p := max(above, key=lambda m: boxes[order[m]][3])) == k - 1:
                continue
            x0, x1 = min(box[0], boxes[order[p]][0]), max(box[2], boxes[order[p]][2])
            between = [boxes[j] for j in order[p + 1:k]]
            if all(b[2] - b[0] >= COLUMN_W * h and (b[2] <= x0 or b[0] >= x1) for b in between) and any(
                    min(b[3], box[3]) > max(b[1], box[1]) for b in between):
                order.insert(p + 1, order.pop(k))
                break
        else:
            break
    return order


def _letter_height(boxes) -> float:
    hs = np.array([b[3] - b[1] for b in boxes])
    return float(np.median(hs[hs > 1.0])) if (hs > 1.0).any() else 5.0


def _runs(boxes, idx, axis, gap) -> list[list[int]]:
    """Stroke indices -> groups along x (axis 0) or y (axis 1), in order: strokes whose extents overlap or come
    within `gap` of the group's."""
    runs = []
    for i in sorted(idx, key=lambda i: boxes[i][axis]):
        if runs and boxes[i][axis] <= runs[-1][1] + gap:
            runs[-1][0].append(i)
            runs[-1][1] = max(runs[-1][1], boxes[i][axis + 2])
        else:
            runs.append([[i], boxes[i][axis + 2]])
    return [r for r, _ in runs]


def _rules(boxes, h) -> list[list[int]]:
    """Long flat lines, each drawn in one stroke or several end to end -> their stroke indices, top first."""
    groups = []
    for i in sorted(range(len(boxes)), key=lambda i: boxes[i][0]):
        b = boxes[i]
        if b[2] - b[0] < 2 * h or b[3] - b[1] > RULE_FLAT * (b[2] - b[0]):
            continue
        g = next((g for g in groups if -2 * h <= b[0] - boxes[g[-1]][2] <= h
                  and min(b[3], boxes[g[-1]][3]) - max(b[1], boxes[g[-1]][1]) > -0.5 * h), None)
        if g:
            g.append(i)
        else:
            groups.append([i])
    groups = [g for g in groups if max(boxes[i][2] for i in g) - min(boxes[i][0] for i in g) >= RULE_W * h]
    return sorted(groups, key=lambda g: min(boxes[i][1] for i in g))


def _cells(boxes, column, h) -> list[list[int]]:
    """A column's strokes -> its cells, top to bottom: strokes stacked apart, except that a flat bit joins its
    neighbours: a T's bar written apart joins the stem under it, a fraction's bar both its terms (the numerator
    close above, the denominator a little apart below), any other the nearer cell."""
    cells = _runs(boxes, column, 1, 0.0)
    ext = lambda c, a: (min(boxes[i][a] for i in c), max(boxes[i][a + 2] for i in c))
    flat = lambda c: ext(c, 1)[1] - ext(c, 1)[0] < min(0.6 * h, 0.6 * (ext(c, 0)[1] - ext(c, 0)[0]))
    while len(cells) > 1 and (k := next((k for k, c in enumerate(cells) if flat(c)), None)) is not None:
        up = ext(cells[k], 1)[0] - ext(cells[k - 1], 1)[1] if k else np.inf
        down = ext(cells[k + 1], 1)[0] - ext(cells[k], 1)[1] if k + 1 < len(cells) else np.inf
        if k + 1 < len(cells) and ext(cells[k + 1], 0)[1] - ext(cells[k + 1], 0)[0] <= 0.5 * h:
            a, b = k, k + 1  # a T's bar over its stem
        elif up <= 0.5 * h and 0.2 * h <= down <= 0.7 * h:
            a, b = k - 1, k + 1  # a fraction's bar between its terms
        else:
            a, b = (k - 1, k) if up <= down else (k, k + 1)
        cells[a:b + 1] = [[i for c in cells[a:b + 1] for i in c]]
    k = 0
    while k + 1 < len(cells):  # a fraction whose bar touches its numerator: the denominator a little under the bar
        low = max(boxes[i][3] for i in cells[k])
        bars = [boxes[i] for i in cells[k] if boxes[i][3] >= low - 0.3 * h and boxes[i][2] - boxes[i][0] >= 0.8 * h
                and boxes[i][3] - boxes[i][1] < 0.3 * (boxes[i][2] - boxes[i][0])]
        x = (ext(cells[k + 1], 0)[0] + ext(cells[k + 1], 0)[1]) / 2
        if len(cells[k]) > 1 and 0.15 * h <= ext(cells[k + 1], 1)[0] - low <= 0.7 * h and any(
                b[0] - 0.2 * h <= x <= b[2] + 0.2 * h for b in bars):
            cells[k:k + 2] = [cells[k] + cells[k + 1]]
        else:
            k += 1
    return cells


def _upright(b, h) -> bool:
    return b[3] - b[1] >= 2.5 * h and b[2] - b[0] <= 0.3 * (b[3] - b[1])  # a line drawn between columns


def _table_under(strokes, rule, marks, h, taken):
    """The table whose head the rule (stroke indices) underlines -> (its stroke indices, rows of cells), or None.
    Drawn as a grid (lines down across the rule), its columns lie between those lines and its rows end with them.
    Otherwise its columns are those of the first row, and it ends at an empty gap or at writing across the space
    between them (the line under the table)."""
    boxes = [s.box for s in strokes]
    pts = np.concatenate([q for i in rule for q in strokes[i].outlines])
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()

    def rule_y(x):  # the rule's height at x: it may slope or wave
        return float(pts[np.argsort(np.abs(pts[:, 0] - x))[:8], 1].mean())

    def middle(idx, axis):
        return (min(boxes[i][axis] for i in idx) + max(boxes[i][axis + 2] for i in idx)) / 2

    near = [i for i, b in enumerate(boxes) if i not in rule and i not in taken and x0 - h <= (b[0] + b[2]) / 2 <= x1 + h]
    y = {i: rule_y((boxes[i][0] + boxes[i][2]) / 2) for i in near}
    upright = {i for i in near if _upright(boxes[i], h)}
    head = [i for i in near if i not in upright and y[i] - HEAD_H * h <= boxes[i][3] <= y[i] + 0.3 * h]
    below = sorted((i for i in near if i not in upright and boxes[i][3] > y[i] + 0.3 * h and boxes[i][1] >= y[i] - h),
                   key=lambda i: boxes[i][1])
    seps = sorted((i for i in upright if boxes[i][1] < y[i] < boxes[i][3]), key=lambda i: middle([i], 0))
    if len(seps) >= 2:  # a grid
        edges = [middle([i], 0) for i in seps]
        body = [i for i in below if boxes[i][1] < max(boxes[i][3] for i in seps) + 0.5 * h]
        columns = [[] for _ in range(len(edges) + 1)]
        for i in body:
            b = boxes[i]
            on_line = b[2] - b[0] <= 0.35 * h and min(abs(middle([i], 0) - e) for e in edges) <= 0.4 * h
            if i not in marks and not on_line:  # (a line drawn in pieces)
                columns[bisect.bisect(edges, middle([i], 0))].append(i)
    else:
        first = [i for i in below if i not in marks and boxes[i][1] <= y[i] + FIRST_ROW * h]
        cols = [[min(boxes[i][0] for i in c), max(boxes[i][2] for i in c)] for c in _runs(boxes, first, 0, COL_GAP * h)]
        body, columns, end = [], [[] for _ in cols], max(boxes[i][3] for i in rule) + FIRST_ROW * h
        for i in below:
            b = boxes[i]
            if b[1] > end:
                break
            if i not in marks:
                js = [j for j, (c0, c1) in enumerate(cols) if b[0] < c1 + COL_GAP * h and b[2] > c0 - COL_GAP * h]
                if len(js) > 1:  # across the space between columns: the writing under the table, all of its line
                    body = [k for k in body if boxes[k][3] < b[1]]
                    break
                if js:
                    j = js[0]
                    cols[j] = [min(cols[j][0], b[0]), max(cols[j][1], b[2])]
                else:  # a column whose cells start lower down
                    j = sum(c0 < b[0] for c0, _ in cols)
                    cols.insert(j, [b[0], b[2]])
                    columns.insert(j, [])
                columns[j].append(i)
                end = max(end, b[3] + ROW_GAP * h)
            body.append(i)
    kept = set(body)
    columns = [[k for k in col if k in kept] for col in columns if kept & set(col)]
    cols = [(min(boxes[i][0] for i in c), max(boxes[i][2] for i in c)) for c in columns]
    cells = [_cells(boxes, c, h) for c in columns]
    if len(cols) < 3:
        return None
    n = Counter(map(len, cells)).most_common(1)[0][0]  # rows: most columns have one cell in each
    rows = [[[] for _ in cols] for _ in range(n + 1)]
    centers = [middle(c, 1) for c in next(c for c in cells if len(c) == n)]
    for j, column in enumerate(cells):
        for k, c in enumerate(column):  # such a column's k-th cell is row k's; another column's goes by height
            rows[1 + (k if len(column) == n else int(np.argmin([abs(middle(c, 1) - y) for y in centers])))][j] += c
    for title in _runs(boxes, head, 0, HEAD_GAP * h):
        rows[0][int(np.argmin([abs(middle(title, 0) - (c0 + c1) / 2) for c0, c1 in cols]))] += title
    if sum(bool(c) for row in rows[1:] for c in row) < 0.75 * n * len(cols) or n < 2 and (
            len(cols) < 4 or sum(map(bool, rows[0])) < 0.75 * len(cols) or len(seps) < 2):
        return None  # one row (a pmf): a grid, a title over each column
    bands = [(min(boxes[i][1] for c in row for i in c), max(boxes[i][3] for c in row for i in c)) if any(row) else None
             for row in rows]
    top, bottom = min(b[0] for b in bands if b), max(b[1] for b in bands if b)
    used = set(rule + head + body) | {i for i in upright if boxes[i][1] < bottom and boxes[i][3] > top}
    side, left = [[] for _ in rows], cols[0][0]  # the rows' titles, left of the rule ("p(Y = y)")
    for i, b in enumerate(boxes):
        if i in used or i in taken or not left - 12 * h <= b[0] <= b[2] <= left:
            continue
        k = next((k for k, band in enumerate(bands) if band and band[0] <= (b[1] + b[3]) / 2 <= band[1]), None)
        if k is not None and _upright(b, h):
            used.add(i)
        elif k is not None:
            side[k].append(i)
    if sum(map(bool, side[1:])) >= 0.75 * n:
        rows = [[s] + row for s, row in zip(side, rows)]
        used |= {i for s in side for i in s}
    return sorted(used), rows


def find_tables(strokes: list[PdfStroke]) -> list[tuple[list[int], list[list[list[int]]]]]:
    """Tables among strokes in writing order -> for each, its stroke indices and its rows of cells (stroke indices;
    the first row the head). A table is titles over a long flat rule and, under it, columns of cells with empty
    space between them (a truth table); lines drawn across its rows are marks, in no cell. Its rows are read down
    each column: one column's cells need not line up with the next one's."""
    if not strokes:
        return []
    boxes = [s.box for s in strokes]
    h = _letter_height(boxes)
    rules = _rules(boxes, h)
    marks = {i for g in rules for i in g}
    tables, taken = [], set()
    for rule in rules:
        if not taken & set(rule) and (t := _table_under(strokes, rule, marks, h, taken)):
            tables.append(t)
            taken.update(t[0])
    return tables


def find_lines(boxes, fixed=(), points=None) -> list[list[int]]:
    """Group strokes, given as boxes in writing order, into lines of writing -> stroke indices per line, in
    reading order: the order the lines were started (notes are written top to bottom, a column at a time), but
    a line squeezed in later above earlier ones comes before them. Groups of strokes in `fixed` (tables) are
    lines as they are; `points`, each stroke's outline points, if given, let a sloping fraction bar count.

    A stroke written right after the previous ones, close to them, continues their piece. Pieces written at
    different times merge when they share a row with a small gap, or one sits inside the other; a fraction bar
    written on its own joins its terms; a few narrow strokes just above or below a line (limits, exponents, a
    numerator added later) join it; an integral sign left on its own with its limits (its upper limit, written
    above it, broke the piece off) joins the line right of it on its row. Pieces far apart on one row stay
    separate lines (a side note like u = x^2 next to an integral)."""
    if not boxes:
        return []
    h = _letter_height(boxes)
    skip = {i for g in fixed for i in g}

    def fragment(p):
        x0, y0, x1, y1 = p[1]
        return len(p[0]) <= FRAGMENT_STROKES and x1 - x0 <= FRAGMENT_W * h and y1 - y0 <= FRAGMENT_H * h

    pieces = []  # [stroke indices, box]
    for i, b in enumerate(boxes):
        if i in skip:
            continue
        if pieces:
            box = pieces[-1][1]
            if (b[1] < box[3] + JOIN_Y * h and b[3] > box[1] - JOIN_Y * h
                    and b[0] < box[2] + JOIN_X * h and b[2] > box[0] - BACK_X * h):
                pieces[-1] = [pieces[-1][0] + [i], _union(box, b)]
                continue
        pieces.append([[i], b])
    _join_fractions(pieces, boxes, h, points)
    _merge_rows(pieces, h)
    while _join_bars(pieces, h):
        pass
    _merge_rows(pieces, h)

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
    for f in [p for p in lines if len(p[0]) <= 8 and p[1][2] - p[1][0] <= 3 * h and any(
            boxes[i][3] - boxes[i][1] >= 2.5 * h and boxes[i][2] - boxes[i][0] <= 0.35 * (boxes[i][3] - boxes[i][1])
            for i in p[0])]:  # an integral sign and its limits, alone
        right = [p for p in lines if p is not f and -0.5 * h <= p[1][0] - f[1][2] <= JOIN_X * h
                 and min(p[1][3], f[1][3]) - max(p[1][1], f[1][1]) >= 0.5 * (p[1][3] - p[1][1])]
        if right:
            host = min(right, key=lambda p: p[1][0])
            host[0], host[1] = host[0] + f[0], _union(host[1], f[1])
            lines = [p for p in lines if p is not f]
    for g in fixed:
        lines.append([list(g), (min(boxes[i][0] for i in g), min(boxes[i][1] for i in g),
                                max(boxes[i][2] for i in g), max(boxes[i][3] for i in g))])
    return _reading_order(lines)


def find_drawings(strokes: list[PdfStroke], skip=()) -> list[tuple[list[int], tuple]]:
    """Drawings among strokes (a sketch of a solid, boxes drawn over a graph), strokes in `skip` (tables) aside:
    long level lines crossing long upright ones, with every stroke inside the box around them -> for each, its
    stroke indices and that box. Writing has no such crossings: an integral sign, a bracket or a fraction bar
    is long one way only, and they meet at most at their ends."""
    boxes = [s.box for s in strokes]
    if not boxes:
        return []
    h = _letter_height(boxes)
    w, t = (lambda b: b[2] - b[0]), (lambda b: b[3] - b[1])
    level = [i for i, b in enumerate(boxes) if i not in skip and w(b) >= DRAW_W * h and t(b) <= 0.3 * w(b)]
    upright = [i for i, b in enumerate(boxes) if i not in skip and t(b) >= DRAW_W * h and w(b) <= 0.3 * t(b)]
    regions = []
    for a in level:
        for c in upright:
            (ax0, ay0, ax1, ay1), (cx0, cy0, cx1, cy1) = boxes[a], boxes[c]
            if ax0 + 0.5 * h < (cx0 + cx1) / 2 < ax1 - 0.5 * h and cy0 + 0.5 * h < (ay0 + ay1) / 2 < cy1 - 0.5 * h:
                regions.append(_union(boxes[a], boxes[c]))
    merged = True
    while merged:  # crossings whose boxes meet are one drawing
        merged = False
        for k in range(len(regions)):
            for m in range(k + 1, len(regions)):
                a, b = regions[k], regions[m]
                if a[0] <= b[2] + h and b[0] <= a[2] + h and a[1] <= b[3] + h and b[1] <= a[3] + h:
                    regions[k] = _union(a, b)
                    del regions[m]
                    merged = True
                    break
            if merged:
                break
    out, taken = [], set(skip)
    for region in regions:  # grows by the lines that touch it (at least 2h long) and whatever lies inside
        idx, grown = [], True
        while grown:
            x0, y0, x1, y1 = region
            more = [i for i, b in enumerate(boxes) if i not in taken and (
                x0 - 0.5 * h <= (b[0] + b[2]) / 2 <= x1 + 0.5 * h and y0 - 0.5 * h <= (b[1] + b[3]) / 2 <= y1 + 0.5 * h
                or max(w(b), t(b)) >= 2 * h and b[0] <= x1 and x0 <= b[2] and b[1] <= y1 and y0 <= b[3])]
            grown = bool(more)
            for i in more:
                region = _union(region, boxes[i])
            idx += more
            taken.update(more)
        if idx:
            out.append((sorted(idx), region))
    return out


class Table(list):
    """A table's strokes, a line of its own in the reading order, read a cell at a time: `rows` of cells, each the
    cell's strokes (the first row the head; an empty cell [])."""

    def __init__(self, strokes, rows):
        super().__init__(strokes)
        self.rows = rows


def ink_lines(strokes: list[PdfStroke]) -> list[list[PdfStroke]]:
    """Strokes in writing order -> the lines of writing in reading order, each its strokes (find_lines); a table
    is one of them, as a Table (find_tables)."""
    boxes = [s.box for s in strokes]
    tables = {tuple(idx): rows for idx, rows in find_tables(strokes)}
    out = []
    for line in find_lines(boxes, list(tables), [np.concatenate(s.outlines) for s in strokes]):
        rows = tables.get(tuple(line))
        picked = [strokes[i] for i in line]
        out.append(Table(picked, [[[strokes[i] for i in c] for c in row] for row in rows]) if rows else picked)
    return out


def page_lines(page) -> list[list[PdfStroke]]:
    """The lines of handwriting on a PDF page in reading order, each as its strokes (a table as a Table). Lines
    cut by a page break are left out: their other half is on the neighbouring page."""
    lines = ink_lines(pdf_strokes(page))
    return [line for line in lines if not any(s.clipped for s in line)]
