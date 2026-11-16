"""Handwritten notes in a PDF whose ink is vector strokes (OneNote's export) -> text with $...$ math, a page at a
time, or Markdown, a note at a time.

OneNote prints a note's title and date at the top of its first page and a footer on every page. Its pages overlap:
the bottom of one page is drawn again at the top of the next, so a line near a page break is cut on one page or on
both. A note's lines are found a page at a time, then joined at the page breaks."""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypdfium2.raw as pdfium_c
from PIL import Image

from im2tex.ink import PdfStroke, find_lines, page_lines, pdf_strokes, render_ink
from im2tex.page import PageError, split_page

PAGE_SHIFT = 734.0  # pt from a page's top to the next one's in OneNote's export, when no shared strokes tell
DATE = re.compile(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), \w+ \d{1,2}, \d{4}")
FOOTER = 50.0  # pt: typed text this close to the page's bottom edge is the footer
ICON = 50.0  # pt: a picture smaller than this is an attachment's icon, with the file's name typed under it
ITEM = re.compile(r"\(?[a-z]{1,3}[.)]\s")  # b) / (c) / ii. starting a line: an item, not a wrapped sentence
LABEL = re.compile(r"(\d+[.)]|\(?[a-z]{1,3}\))( (\d+[.)]|\(?[a-z]{1,3}\)))*")  # 4. / b) / 3. a) before a formula
RELATIONS = {"=", r"\equiv", r"\leq", r"\geq", "<", ">", r"\neq", r"\approx", r"\Rightarrow", r"\Leftrightarrow",
             r"\iff", r"\implies"}  # a step's relation: rows of a solution align at the first one
BINARY = {"+", "-", r"\pm", r"\times", r"\cdot", r"\rightarrow", r"\leftarrow", r"\to", r"\in", r"\notin",
          r"\subseteq", r"\subset", r"\wedge", r"\vee", r"\oplus", r"\cup", r"\cap"}
WORD = re.compile(r"\\[a-zA-Z]+")


@dataclass
class Line:
    text: str
    box: tuple  # x0, y0, x1, y1 in pt, y down


def _union(boxes):
    b = np.array(boxes, dtype=float)
    return (*b[:, :2].min(0), *b[:, 2:].max(0))


def as_display(text: str) -> str:
    """A line read as math only -> $$...$$; otherwise as read."""
    try:
        parts = split_page(text)
    except PageError:
        return text
    if parts and all(k != "text" for k, _ in parts):
        return "$$" + " ".join(c.strip() for _, c in parts) + "$$"
    return text


def join_side_notes(lines: list[Line]) -> list[Line]:
    """A line read as a note in parentheses, like "(De Morgan's Law)" written to the right of a step, joins the
    line to its left on its row (the one it overlaps most in height); a display line it joins becomes inline."""
    out = list(lines)
    for k in range(len(out) - 1, -1, -1):
        note = out[k]
        if not note.text.lstrip().startswith("("):
            continue
        x0, y0, _, y1 = note.box
        hosts = [(min(y1, l.box[3]) - max(y0, l.box[1]), m) for m, l in enumerate(out)
                 if m != k and l.box[2] <= x0 + 1.0 and not l.text.lstrip().startswith("(")]
        hosts = [(overlap, m) for overlap, m in hosts if overlap > 0.25 * (y1 - y0)]
        if not hosts:
            continue
        m = max(hosts)[1]
        host = out[m].text
        if host.startswith("$$") and host.endswith("$$"):
            host = "$" + host[2:-2] + "$"
        out[m] = Line(host + " " + note.text.strip(), _union([out[m].box, note.box]))
        del out[k]
    return out


def read_lines(lines: list[list[PdfStroke]], predict) -> list[Line]:
    """Lines of strokes -> their readings, one at a time (padding in a batch changes readings), side notes joined."""
    out = []
    for strokes in lines:
        im = Image.fromarray(render_ink([q for s in strokes for q in s.outlines], filled=True)).convert("RGB")
        out.append(Line(as_display(predict([im])[0]), _union([s.box for s in strokes])))
    return join_side_notes(out)


def read_page(page, predict) -> list[Line]:
    """A page's lines of writing, read, in reading order. Lines cut by a page break are left out (read_note reads
    them whole)."""
    return read_lines(page_lines(page), predict)


def typed_lines(page) -> list[Line]:
    """The lines of the page's text layer (typed text: OneNote's title, date, footer, attachment names), y down."""
    tp = page.get_textpage()
    height = page.get_height()
    out, chars, boxes = [], [], []

    def flush():
        text = " ".join("".join(chars).split())
        if text and boxes:
            out.append(Line(text, _union(boxes)))
        chars.clear()
        boxes.clear()

    for i in range(tp.count_chars()):
        c = tp.get_text_range(i, 1)
        if c in ("\r", "\n"):
            flush()
            continue
        chars.append(c)
        left, bottom, right, top = tp.get_charbox(i)
        if right > left:
            boxes.append((left, height - top, right, height - bottom))
    flush()
    return out


def group_notes(pdf, pages: list[int]) -> list[list[int]]:
    """Page numbers -> the notes they belong to: a note starts on a page with a typed date (OneNote's first page
    of a note) or where the given pages skip one."""
    notes = []
    for n in pages:
        starts = DATE.search(pdf[n - 1].get_textpage().get_text_range())
        if not notes or starts or n != notes[-1][-1] + 1:
            notes.append([n])
        else:
            notes[-1].append(n)
    return notes


def _shifted(s: PdfStroke, dy: float) -> PdfStroke:
    return PdfStroke([q + [0.0, dy] for q in s.outlines], (s.box[0], s.box[1] + dy, s.box[2], s.box[3] + dy), False)


def _overlap(a: list[PdfStroke], b: list[PdfStroke]) -> tuple[float, dict]:
    """Strokes of page a drawn again on page b (the same outline, moved up the same distance) -> that distance and
    {index in b: index in a} of the copies; PAGE_SHIFT and none when fewer than 3 agree."""
    by_size = {}
    for j, t in enumerate(b):
        by_size.setdefault(sum(len(q) for q in t.outlines), []).append(j)
    found = {}
    for i, s in enumerate(a):
        pa = np.concatenate(s.outlines)
        for j in by_size.get(len(pa), []):
            d = np.concatenate(b[j].outlines) - pa
            if np.abs(d - d[0]).max() < 0.05 and abs(d[0][0]) < 0.05:
                found.setdefault(round(-float(d[0][1]), 1), {})[j] = i
    shift, copies = max(found.items(), key=lambda kv: len(kv[1]), default=(PAGE_SHIFT, {}))
    return (shift, copies) if len(copies) >= 3 else (PAGE_SHIFT, {})


def _rows_meet(a: list[PdfStroke], b: list[PdfStroke]) -> bool:
    (_, ay0, _, ay1), (_, by0, _, by1) = _union([s.box for s in a]), _union([s.box for s in b])
    return min(ay1, by1) > max(ay0, by0)


def note_lines(pages: list[list[PdfStroke]]) -> tuple[list[list[PdfStroke]], list[float]]:
    """A note's pages of strokes -> its lines of writing in reading order, in one frame (each page moved down below
    the last), and how far down each page went. Lines are found a page at a time (each page's own letter height),
    leaving out a page's copies of strokes from the page above, so a line drawn on both is read once. A line cut
    by a break reaches past it (its strokes are whole, only drawn clipped): the lines on either side that share
    rows with it are found again from their strokes together, which reads it whole."""
    out, offsets, offset, above = [], [], 0.0, []  # above: indices in out of the previous page's lines
    for k, strokes in enumerate(pages):
        copies = {}
        if k:
            shift, copies = _overlap(pages[k - 1], strokes)
            offset += shift
        offsets.append(offset)
        kept = [_shifted(s, offset) for i, s in enumerate(strokes) if i not in copies]
        lines = [[kept[i] for i in line] for line in find_lines([s.box for s in kept])]
        old = [m for m in above if any(_rows_meet(out[m], l) for l in lines)]
        new = [l for l in lines if any(_rows_meet(out[m], l) for m in above)]
        if old:
            pooled = [s for m in old for s in out[m]] + [s for l in new for s in l]
            found = [[pooled[i] for i in line] for line in find_lines([s.box for s in pooled])]
            out = out[:old[0]] + found + [l for m, l in enumerate(out[old[0]:], old[0]) if m not in old]
            lines = [l for l in lines if not any(l is n for n in new)]
        above = list(range(len(out), len(out) + len(lines)))
        out += lines
    return out, offsets


def _escape(line: str) -> str:
    """Keep Markdown from reading a written line as a list item, heading or quote: "40. For" -> "40\\. For"."""
    m = re.match(r"(\d+)([.)])(\s|$)", line)
    if m:
        return m.group(1) + "\\" + line[len(m.group(1)):]
    if re.match(r"([-+*]|#+|>)(\s|$)", line):
        return "\\" + line
    return line


def _continues(prev: str, line: str) -> bool:
    """A sentence wrapped onto the next line: it goes on in lowercase, and neither line is a display formula or
    a picture."""
    return (not prev.startswith(("$$", "![", "#")) and not line.startswith("$$")
            and line[:1].islower() and not ITEM.match(line))


def latex(tokens: list[str]) -> str:
    """The model's LaTeX, one token per space-separated piece, -> source as a person writes it: x^2, \\frac{1}{2},
    \\int_0^1 2^x dx, spaces around relations and binary operators. An evaluation bar (| with limits, not closing
    an absolute value) becomes \\Big|."""
    toks, k, open_abs = [], 0, False
    while k < len(tokens):
        t, nxt = tokens[k], tokens[k + 1] if k + 1 < len(tokens) else ""
        if t == "|":
            if nxt not in ("_", "^"):
                open_abs = not open_abs
            elif open_abs:
                open_abs = False
            else:
                t = r"\Big|"
        if (t in ("^", "_") and nxt == "{" and k + 3 < len(tokens) and tokens[k + 3] == "}"
                and (len(tokens[k + 2]) == 1 and tokens[k + 2] not in "{}" or WORD.fullmatch(tokens[k + 2]))):
            toks.append(t + tokens[k + 2])  # x ^ { 2 } -> x^2
            k += 4
            continue
        toks.append(t)
        k += 1
    s = ""
    for k, t in enumerate(toks):
        prev = toks[k - 1] if k else None
        unary = t in ("-", "+") and (prev is None or prev in RELATIONS | BINARY | {"{", "(", "[", "&", ","})
        if t == "&":
            s = s.rstrip() + " &"
        elif t in RELATIONS and prev == "&":
            s += t + " "
        elif t in RELATIONS | BINARY and not unary:
            s = s.rstrip() + f" {t} "
        elif t == ",":
            s = s.rstrip() + ", "
        elif t == r"\\":
            s = s.rstrip() + r" \\ "
        elif t[:1].isalnum() and (re.search(r"\\[a-zA-Z]+$", s) or prev and prev[:1] in "^_" and len(prev) > 1):
            s += " " + t  # \ln 2, and \int_0^1 2^x rather than \int_0^12^x
        else:
            s += t
    return s.strip()


def _align_at(tokens: list[str]) -> int | None:
    """Where a display row aligns with the rest of its solution: its first relation outside any brackets (a row
    may also start with \\rightarrow, continuing the row above)."""
    depth = 0
    for i, t in enumerate(tokens):
        if t in ("{", "(", "[", r"\{"):
            depth += 1
        elif t in ("}", ")", "]", r"\}"):
            depth -= 1
        elif depth == 0 and (t in RELATIONS or i == 0 and t == r"\rightarrow"):
            return i
    return None


def _display(rows: list[tuple[list[str], str | None]]) -> str:
    """A solution's display rows (tokens, reason or None) -> one centered block: the rows aligned at their first
    relation (a row without one ends at the alignment), reasons such as "(De Morgan's Law)" in a column of their
    own; rows with no relation at all are just centered."""
    if len(rows) == 1 and not rows[0][1]:
        return "$$\n" + latex(rows[0][0]) + "\n$$"
    if not any(_align_at(t) is not None or reason for t, reason in rows):
        return "$$\n\\begin{gathered}\n" + " \\\\\n".join(latex(t) for t, _ in rows) + "\n\\end{gathered}\n$$"
    out = []
    for t, reason in rows:
        i = _align_at(t)
        line = latex(t[:i] + ["&"] + t[i:] if i is not None else t + ["&"])
        if reason:
            line += " && \\text{" + re.sub(r"([{}&%#_$])", r"\\\1", reason) + "}"
        out.append(line)
    return "$$\n\\begin{aligned}\n" + " \\\\\n".join(out) + "\n\\end{aligned}\n$$"


def _inline(text: str) -> str:
    """A written line with words: its math written compactly between $ signs (no space inside them)."""
    try:
        parts = split_page(text)
    except PageError:
        return text
    return "".join(c if k == "text" else f"${latex(c.split())}$" if k == "inline" else f"$${latex(c.split())}$$"
                   for k, c in parts)


def _kind(text: str):
    """A written line -> ("display", (tokens, reason)) for a formula alone or followed by a reason in parentheses,
    ("label", (label, tokens)) for a problem label followed by a formula, or ("text", None)."""
    try:
        parts = [(k, c) for k, c in split_page(text) if k != "text" or c.strip()]
    except PageError:
        return "text", None
    if len(parts) == 1 and parts[0][0] != "text":
        return "display", (parts[0][1].split(), None)
    if len(parts) == 2 and parts[0][0] != "text" and parts[1][0] == "text" and re.fullmatch(r"\(.*\)", parts[1][1].strip()):
        return "display", (parts[0][1].split(), parts[1][1].strip())
    if len(parts) == 2 and parts[0][0] == "text" and parts[1][0] != "text" and LABEL.fullmatch(parts[0][1].strip()):
        return "label", (parts[0][1].strip(), parts[1][1].split())
    return "text", None


def to_markdown(title: str | None, date: str | None, lines: list[str]) -> str:
    """A note -> Markdown laid out like a typeset solution: the title as a heading and the date under it; words
    left-aligned, each written line a paragraph of its own except that a wrapped sentence stays one; formulas
    displayed and centered, consecutive ones (one solution's steps) in one block aligned at their relations; a
    problem label before a formula on a line of its own above it."""
    blocks = [f"# {title or date or 'Untitled'}"] + ([f"*{date}*"] if title and date else [])
    rows, para, last = [], [], None  # pending display rows; pending paragraph lines; the raw last line of para

    def flush():
        if para:
            blocks.append("\n".join(para))
            para.clear()
        if rows:
            blocks.append(_display(rows))
            rows.clear()

    for text in lines:
        kind, parts = _kind(text)
        if kind == "display":
            if para or rows and _align_at(parts[0]) is None and _align_at(rows[-1][0]) == 0:
                flush()  # after words, or a new expression after a chain of = steps: a new solution
            rows.append(parts)
        elif kind == "label":
            flush()
            blocks.append(f"**{parts[0]}**")
            rows.append((parts[1], None))
        elif para and _continues(last, text):
            para.append(_inline(text))
            last = text
        else:
            flush()
            para.append(_escape(_inline(text)))
            last = text
    flush()
    return "\n\n".join(blocks) + "\n"


def read_note(pdf, pages: list[int], predict, assets: Path, link: str) -> tuple[str | None, str]:
    """A note's pages -> (title, Markdown). Printouts and other pictures are saved to `assets` as 1.png, 2.png, ...
    and linked as `link`/1.png where they sit; attachment icons and their names are left out."""
    ink, offsets = note_lines([pdf_strokes(pdf[n - 1]) for n in pages])
    placed, head = [], []  # pictures and typed lines, placed by position; the first page's top typed lines
    for k, (n, dy) in enumerate(zip(pages, offsets)):
        page = pdf[n - 1]
        icons = []
        for obj in page.get_objects(max_depth=5):
            if obj.type != pdfium_c.FPDF_PAGEOBJ_IMAGE:
                continue
            left, bottom, right, top = obj.get_bounds()
            box = (left, page.get_height() - top, right, page.get_height() - bottom)
            if right - left < ICON and top - bottom < ICON:
                icons.append(box)
                continue
            assets.mkdir(parents=True, exist_ok=True)
            name = f"{sum(t.startswith('![') for t, _ in placed) + 1}.png"
            obj.get_bitmap(render=True).to_pil().save(assets / name)
            placed.append((f"![]({link}/{name})", (box[0], box[1] + dy, box[2], box[3] + dy)))
        for t in typed_lines(page):
            x0, y0, x1, _ = t.box
            if y0 > page.get_height() - FOOTER:
                continue
            if any(i[0] - 30 <= x1 and x0 <= i[2] + 30 and 0 <= y0 - i[3] <= 40 for i in icons):
                continue  # an attachment's name
            if k == 0 and (DATE.search(t.text) or y0 < 60):
                head.append(t)
            else:
                placed.append((t.text, (t.box[0], y0 + dy, t.box[2], t.box[3] + dy)))
    date = next((t.text for t in head if DATE.search(t.text)), None)
    title = " ".join(t.text for t in sorted(head, key=lambda t: t.box[1]) if not DATE.search(t.text)) or None

    lines = read_lines(ink, predict)
    for text, box in sorted(placed, key=lambda p: p[1][1], reverse=True):
        at = next((k for k, l in enumerate(lines) if l.box[1] >= box[1]), len(lines))
        lines.insert(at, Line(text, box))
    return title, to_markdown(title, date, [l.text for l in lines])
