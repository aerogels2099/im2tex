import ctypes

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from im2tex.ink import PdfStroke
from im2tex.pdf_notes import (Line, group_notes, join_side_notes, latex, note_lines, read_note, to_markdown,
                              typed_lines)


def test_a_note_in_parentheses_joins_the_line_on_its_row():
    lines = [Line("$$\\equiv x$$", (10, 10, 50, 20)), Line("$$= y$$", (10, 30, 40, 40)),
             Line("(De Morgan's Law)", (80, 8, 120, 17)), Line("(see above)", (10, 60, 50, 70))]
    assert join_side_notes(lines) == [Line("$\\equiv x$ (De Morgan's Law)", (10, 8, 120, 20)),
                                      Line("$$= y$$", (10, 30, 40, 40)), Line("(see above)", (10, 60, 50, 70))]


def test_markdown_keeps_written_lines_apart_and_problem_numbers_in_bold():
    lines = ["12. There exists an irrational number and", "a rational number whose product is rational",
             "15. b) True", "d) True", "$$x = 1$$", "so x is odd", "- 3 apples", "3 125", "2. $ x $ is odd"]
    assert to_markdown("5.3-4 Homework", "Thursday, September 11, 2025 4:04 PM", lines) == (
        "# 5.3-4 Homework\n\n*Thursday, September 11, 2025 4:04 PM*\n\n"
        "**12.** There exists an irrational number and\na rational number whose product is rational\n\n"
        "**15. b)** True\n\n**d)** True\n\n$$\nx = 1\n$$\n\nso x is odd\n\n\\- 3 apples\n\n3 125\n\n**2.** $x$ is odd\n")
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
        "$$\n\\begin{aligned}\n\\int_0^1 xdx &= \\frac{1}{2} \\\\\nu &= x^2 \\\\\n"
        "&\\equiv p && \\text{(De Morgan's Law)}\n\\end{aligned}\n$$\n\n"
        "$$\n\\int_0^2 xdx = 2\n$$\n\n"
        "So the area is $\\frac{1}{2}$\n\n$$\n\\begin{gathered}\nx + 1 \\\\\ny\n\\end{gathered}\n$$\n")


def test_a_first_row_without_a_relation_goes_on_the_row_of_its_first_step():
    lines = ["$$( 0 . 2 ) ( 0 . 3 ) + ( 0 . 1 ) ( 0 . 8 )$$", "$$= 0 . 0 6 + 0 . 0 8$$", "$$= 0 . 1 4$$"]
    assert to_markdown("T", None, lines) == (
        "# T\n\n$$\n\\begin{aligned}\n(0.2)(0.3) + (0.1)(0.8) &= 0.06 + 0.08 \\\\\n&= 0.14\n\\end{aligned}\n$$\n")
    lines = ["$$0 1 2 3$$", r"$$\ell$$", "$$E ( Y )$$", "$$= 2$$", "$$= 2 . 8$$"]  # rows nothing goes on from
    assert to_markdown("T", None, lines) == (
        "# T\n\n$$\n0123\n$$\n\n$$\n\\ell\n$$\n\n$$\n\\begin{aligned}\nE(Y) &= 2 \\\\\n&= 2.8\n\\end{aligned}\n$$\n")


def test_a_substitution_noted_between_steps_goes_after_them():
    lines = ["$$\\int f d x$$", "$$= \\int g d x$$", "$$d u = 2 x$$", "$$= \\int u d u$$"]
    assert to_markdown("T", None, lines) == (
        "# T\n\n$$\n\\begin{aligned}\n\\int fdx &= \\int gdx \\\\\n&= \\int udu\n\\end{aligned}\n$$\n\n$$\ndu = 2x\n$$\n")


def test_a_problem_label_read_into_the_math_is_taken_out():
    lines = ["4. $ a ) : Q ( - 2 , 1 ) $", "$$1 4 . 2 5 x ^ { 2 } = c ^ { 2 }$$", "$ c ) $ It was good",
             "$$1 . 8 1 = x$$", "$$1 2 )$$", "13, $ y = 4 9 $"]  # 14. then 25x^2; 1.81 a decimal; 13, for 13.
    assert to_markdown("T", None, lines) == (
        "# T\n\n**4. a)**\n\n$$\nQ(-2, 1)\n$$\n\n**14.**\n\n$$\n25x^2 = c^2\n$$\n\n**c)** It was good\n\n"
        "$$\n1.81 = x\n$$\n\n**12)**\n\n**13.**\n\n$$\ny = 49\n$$\n")


def test_a_proofs_numbered_steps_are_one_table():
    lines = [r"44) 1. $ r \vee s $ (premise b)", r"2. $ \sim q \vee s $ (premise d)", r"$ \therefore s $ (step 1)",
             "$$x = 1$$"]
    assert to_markdown("T", None, lines) == (
        "# T\n\n**44)**\n\n$$\n\\begin{aligned}\n\\text{1.}\\quad & r \\vee s && \\text{(premise b)} \\\\\n"
        "\\text{2.}\\quad & \\sim q \\vee s && \\text{(premise d)} \\\\\n& \\therefore s && \\text{(step 1)}\n"
        "\\end{aligned}\n$$\n\n$$\nx = 1\n$$\n")


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
    """A PDF of 595 x 843 pt pages, each with typed lines (text, x, baseline y down), ink squares (x, y, size) and
    optionally pictures (x, y, size)."""
    pdf = pdfium.PdfDocument.new()
    for typed, squares, *pictures in pages:
        page = pdf.new_page(595, 843)
        for x, y, s in pictures[0] if pictures else []:
            img = pdfium.PdfImage.new(pdf)
            img.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (8, 8), "gray")))
            img.set_matrix(pdfium.PdfMatrix().scale(s, s).translate(x, 843 - y - s))
            page.insert_obj(img)
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
        ([("5.3-4 Homework", 73, 50), ("Thursday, September 11, 2025 4:04 PM", 72, 75), ("MA-UY 1124 Page 1", 250, 835),
          ("PS", 300, 432), ("1", 300, 444)],  # an attached file's name, wrapped under its icon
         [(100, 200, 8), (110, 200, 9)], [(300, 400, 24)]),
        ([("MA-UY 1124 Page 2", 250, 835)], [(100, 300, 7)]),
        ([("Homework 6", 73, 50), ("Tuesday, October 20, 2026 11:36 PM", 72, 75)], []),
    ])
    assert [t.text for t in typed_lines(pdf[0])] == ["5.3-4 Homework", "Thursday, September 11, 2025 4:04 PM",
                                                     "MA-UY 1124 Page 1", "PS", "1"]
    assert group_notes(pdf, [1, 2, 3]) == [[1, 2], [3]] and group_notes(pdf, [2, 3]) == [[2], [3]]
    reads = iter(["One", "Two"])  # the line on page 1, then page 2's
    title, text = read_note(pdf, [1, 2], lambda images: [next(reads)], tmp_path, "pictures")
    assert title == "5.3-4 Homework"
    assert text == ("# 5.3-4 Homework\n\n*Thursday, September 11, 2025 4:04 PM*\n\nOne\n\n*Attached file: PS1*\n\n"
                    "Two\n")
