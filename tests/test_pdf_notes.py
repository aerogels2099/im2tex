import ctypes

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from im2tex.ink import PdfStroke
from im2tex.pdf_notes import (Line, group_notes, join_side_notes, latex, note_lines, read_note, to_markdown,
                              typed_lines)


def test_a_note_in_parentheses_joins_the_line_on_its_row():
    lines = [Line("$$\\equiv x$$", (10, 10, 50, 20)), Line("$$= y$$", (10, 30, 40, 40)),
             Line("(De Morgan's Law)", (80, 8, 120, 17)), Line("(see above)", (10, 60, 50, 70))]
    assert join_side_notes(lines) == [Line("$\\equiv x$ (De Morgan's Law)", (10, 8, 120, 20)),
                                      Line("$$= y$$", (10, 30, 40, 40)), Line("(see above)", (10, 60, 50, 70))]


def test_markdown_keeps_written_lines_apart_and_numbers_as_written():
    lines = ["12. There exists an irrational number and", "a rational number whose product is rational",
             "15. b) True", "d) True", "$$x = 1$$", "so x is odd", "- 3 apples"]
    assert to_markdown("5.3-4 Homework", "Thursday, September 11, 2025 4:04 PM", lines) == (
        "# 5.3-4 Homework\n\n*Thursday, September 11, 2025 4:04 PM*\n\n"
        "12\\. There exists an irrational number and\na rational number whose product is rational\n\n"
        "15\\. b) True\n\nd) True\n\n$$\nx = 1\n$$\n\nso x is odd\n\n\\- 3 apples\n")
    assert to_markdown(None, "Monday, December 15, 2025 1:31 AM", ["x"]) == "# Monday, December 15, 2025 1:31 AM\n\nx\n"


def test_latex_is_written_compactly():
    assert latex(r"\frac { 2 ^ { x } } { \ln 2 } | _ { 0 } ^ { 1 }".split()) == r"\frac{2^x}{\ln 2}\Big|_0^1"
    assert latex(r"\int _ { 0 } ^ { 1 } 2 ^ { x } d x".split()) == r"\int_0^1 2^x dx"
    assert latex(r"= - 3 x ^ { 2 } + 2 x , x \in \mathbb { R }".split()) == r"= -3x^2 + 2x, x \in \mathbb{R}"
    assert latex(r"| x | ^ { 2 } = x ^ { 1 0 }".split()) == r"|x|^2 = x^{10}"  # an absolute value keeps its bar


def test_a_solution_is_one_centered_block_aligned_at_its_relations():
    lines = [r"4. $ \int _ { 0 } ^ { 1 } x d x $", r"$$= \frac { 1 } { 2 }$$", r"$$u = x ^ { 2 }$$",
             r"$ \equiv p $ (De Morgan's Law)", r"$$\int _ { 0 } ^ { 2 } x d x$$", "$$= 2$$",  # the next problem
             r"So the area is $ \frac { 1 } { 2 } $", "$$x + 1$$", "$$y$$"]
    assert to_markdown("T", None, lines) == (
        "# T\n\n**4.**\n\n"
        "$$\n\\begin{aligned}\n\\int_0^1 xdx & \\\\\n&= \\frac{1}{2} \\\\\nu &= x^2 \\\\\n"
        "&\\equiv p && \\text{(De Morgan's Law)}\n\\end{aligned}\n$$\n\n"
        "$$\n\\begin{aligned}\n\\int_0^2 xdx & \\\\\n&= 2\n\\end{aligned}\n$$\n\n"
        "So the area is $\\frac{1}{2}$\n\n$$\n\\begin{gathered}\nx + 1 \\\\\ny\n\\end{gathered}\n$$\n")


def _stroke(x, y, s):
    return PdfStroke([np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], float)], (x, y, x + s, y + s), False)


def test_a_line_at_a_page_break_is_read_once_and_whole():
    a = [_stroke(10, 100, 5), _stroke(20, 100, 6), _stroke(30, 100, 5.5),  # a line
         _stroke(10, 785, 7), _stroke(22, 786, 6.5), _stroke(34, 785, 7.5)]  # a line at the bottom
    b = [_stroke(10, 51, 7), _stroke(22, 52, 6.5), _stroke(34, 51, 7.5),  # the same strokes, 734 pt up
         _stroke(46, 52, 8),  # the rest of that line, only on this page
         _stroke(10, 300, 6), _stroke(20, 300, 6.2), _stroke(30, 300, 5.8)]
    lines, offsets = note_lines([a, b])
    assert offsets == [0.0, 734.0]
    assert [[round(s.box[1]) for s in line] for line in lines] == [[100] * 3, [785, 786, 785, 786], [1034] * 3]
    lines, _ = note_lines([a, b[:3] + b[4:]])  # drawn whole on both pages: read once
    assert [len(line) for line in lines] == [3, 3, 3]


def _note_pdf(pages):
    """A PDF of 595 x 843 pt pages, each with typed lines (text, x, baseline y down) and ink squares (x, y, size)."""
    pdf = pdfium.PdfDocument.new()
    for typed, squares in pages:
        page = pdf.new_page(595, 843)
        for text, x, y in typed:
            obj = pdfium_c.FPDFPageObj_NewTextObj(pdf.raw, b"Helvetica", 10.0)
            ws = (text + "\0").encode("utf-16-le")
            pdfium_c.FPDFText_SetText(obj, (ctypes.c_ushort * (len(ws) // 2)).from_buffer_copy(ws))
            pdfium_c.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, 843 - y)
            pdfium_c.FPDFPage_InsertObject(page.raw, obj)
        for x, y, s in squares:
            obj = pdfium_c.FPDFPageObj_CreateNewPath(x, 843 - y)
            for px, py in ((x + s, 843 - y), (x + s, 843 - y - s), (x, 843 - y - s)):
                pdfium_c.FPDFPath_LineTo(obj, px, py)
            pdfium_c.FPDFPath_Close(obj)
            pdfium_c.FPDFPageObj_SetFillColor(obj, 0, 0, 0, 255)
            pdfium_c.FPDFPath_SetDrawMode(obj, pdfium_c.FPDF_FILLMODE_WINDING, 0)
            pdfium_c.FPDFPage_InsertObject(page.raw, obj)
        pdfium_c.FPDFPage_GenerateContent(page.raw)
    return pdf


def test_a_note_reads_its_title_and_date_and_leaves_out_footers(tmp_path):
    pdf = _note_pdf([
        ([("5.3-4 Homework", 73, 50), ("Thursday, September 11, 2025 4:04 PM", 72, 75), ("MA-UY 1124 Page 1", 250, 835)],
         [(100, 200, 8), (110, 200, 9)]),
        ([("MA-UY 1124 Page 2", 250, 835)], [(100, 300, 7)]),
        ([("Homework 6", 73, 50), ("Tuesday, October 20, 2026 11:36 PM", 72, 75)], []),
    ])
    assert [t.text for t in typed_lines(pdf[0])] == ["5.3-4 Homework", "Thursday, September 11, 2025 4:04 PM",
                                                     "MA-UY 1124 Page 1"]
    assert group_notes(pdf, [1, 2, 3]) == [[1, 2], [3]] and group_notes(pdf, [2, 3]) == [[2], [3]]
    reads = iter(["One", "Two"])  # the line on page 1, then page 2's
    title, text = read_note(pdf, [1, 2], lambda images: [next(reads)], tmp_path, "pictures")
    assert title == "5.3-4 Homework"
    assert text == "# 5.3-4 Homework\n\n*Thursday, September 11, 2025 4:04 PM*\n\nOne\n\nTwo\n"
