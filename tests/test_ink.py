import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from im2tex.ink import (PdfStroke, Table, column_order, find_drawings, find_lines, find_tables, ink_lines, page_lines,
                        pdf_strokes, render_ink)


def _letters(x, y, n, h=6.0, w=4.0, gap=1.5):
    """n letter-sized stroke boxes written left to right from (x, y), y down."""
    return [(x + k * (w + gap), y, x + k * (w + gap) + w, y + h) for k in range(n)]


def test_strokes_written_together_form_one_line_with_their_fraction():
    first = _letters(10, 10, 5)
    fraction = [(50, 4, 56, 10), (48, 12, 60, 13), (50, 15, 56, 21)]  # numerator, bar, denominator
    second = _letters(10, 40, 6)
    lines = find_lines(first + fraction + second)
    assert lines == [list(range(8)), list(range(8, 14))]


def test_a_limit_added_later_joins_its_line():
    integral = [(10, 10, 14, 30)] + _letters(18, 15, 4)  # a tall integral sign and its integrand
    below = _letters(10, 60, 4)
    limit = [(9, 32, 13, 37)]  # lower limit, written after the next line
    lines = find_lines(integral + below + limit)
    assert lines == [[0, 1, 2, 3, 4, 9], [5, 6, 7, 8]]


def test_a_fraction_written_later_as_a_limit_joins_its_line():
    integral = [(10, 10, 14, 30)] + _letters(18, 15, 4)
    below = _letters(10, 60, 4)
    limit = [(15, -12, 18, -4), (14, -3, 19, -2), (15, -1, 18, 9)]  # 7 over 2, 3.5 letters tall, above the sign
    lines = find_lines(integral + below + limit)
    assert lines == [[0, 1, 2, 3, 4, 9, 10, 11], [5, 6, 7, 8]]


def test_a_fraction_bar_written_apart_joins_its_terms():
    top = _letters(10, 10, 4)
    numerator, denominator = _letters(50, 40, 4), _letters(50, 51, 4)
    bar = [(49, 48, 72, 49)]
    elsewhere, further = _letters(10, 100, 4), _letters(10, 130, 4)  # written in between: no stroke follows another
    lines = find_lines(top + numerator + elsewhere + bar + further + denominator)
    assert lines == [[0, 1, 2, 3], [4, 5, 6, 7, 12, 17, 18, 19, 20], [8, 9, 10, 11], [13, 14, 15, 16]]


def test_a_line_squeezed_in_later_is_read_where_it_is_written():
    first, third = _letters(10, 10, 6), _letters(10, 60, 6)
    second = _letters(10, 35, 6)  # written last, into the space left between them
    corner = _letters(200, 10, 3)  # later still, at the top of another column: still read after
    lines = find_lines(first + third + second + corner)
    assert lines == [list(range(6)), list(range(12, 18)), list(range(6, 12)), list(range(18, 21))]


def test_columns_and_a_far_side_note_stay_separate_in_writing_order():
    left = _letters(10, 10, 5) + _letters(10, 40, 5)
    right = _letters(200, 10, 5) + _letters(200, 40, 5)
    note = _letters(120, 40, 3)  # later, on the row of the second lines, far from both
    lines = find_lines(left + right + note)
    assert lines == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22]]


def test_a_step_written_back_in_an_earlier_column_is_read_with_it():
    left = [(10, 10, 40, 16), (10, 20, 40, 26)]
    right = [(60, 10, 90, 16), (60, 20, 90, 26), (60, 30, 90, 36)]
    back = (10, 31, 40, 37)  # then "= ..." under the left column, level with the right one's last line
    boxes = left + right + [back]
    assert column_order(boxes, 6.0, {3, 4, 5}) == [0, 1, 5, 2, 3, 4]
    assert column_order(boxes, 6.0, {3, 4}) == list(range(6))  # a line that doesn't go on a chain stays
    below = left + right[:2] + [(10, 40, 40, 46)]  # a step under the left column, below the right one's end:
    assert column_order(below, 6.0, {4}) == list(range(5))  # the right one may be added to the row ("+ 1/16")
    assert column_order(below, 6.0, {4}, {2}) == [0, 1, 4, 2, 3]  # unless it starts another problem


def test_an_integral_its_upper_limit_broke_off_joins_its_integrand():
    sign = [(10, 5, 11, 22), (11.5, 20, 12.5, 22.5)]  # the integral sign and its lower limit
    upper = [(11, 0, 13, 1.5)]  # its upper limit, written above it afterwards
    integrand = _letters(27, 11, 4, h=4.0, w=3.0, gap=1.0)
    assert find_lines(sign + upper + integrand) == [list(range(7))]


def test_a_fraction_written_with_its_bar_then_left_joins_its_denominator():
    numerator = _letters(10, 10, 3, h=4.0, w=3.0, gap=1.0) + [(9, 15, 22, 15.5)]  # written with its bar
    elsewhere = _letters(60, 10, 3, h=4.0, w=3.0, gap=1.0)  # then something on the right
    denominator = _letters(11, 16.5, 3, h=4.0, w=3.0, gap=1.0)  # then the denominator
    assert find_lines(numerator + elsewhere + denominator) == [[0, 1, 2, 3, 7, 8, 9], [4, 5, 6]]


def _box_stroke(x0, y0, x1, y1):
    return PdfStroke([np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)], (x0, y0, x1, y1), False)


def test_a_truth_table_is_found_and_read_down_its_columns():
    head = [_box_stroke(x, 9, x + 3, 12) for x in (10, 30, 50)]
    rule = _box_stroke(5, 14, 65, 14.6)
    cells = [_box_stroke(x, y + (2 if x == 50 else 0), x + 3, y + 3 + (2 if x == 50 else 0))
             for y in (17, 21, 25) for x in (10, 30, 50)]  # the last column written a little lower
    words = _box_stroke(12, 31, 32, 35)  # the sentence under the table, across its columns
    strokes = head + [rule] + cells + [words]
    [(idx, rows)] = find_tables(strokes)
    assert rows == [[[0], [1], [2]], [[4], [5], [6]], [[7], [8], [9]], [[10], [11], [12]]]
    assert idx == list(range(13))
    lines = ink_lines(strokes)
    assert isinstance(lines[0], Table) and lines[0].rows[1][2] == [cells[2]] and lines[1] == [words]


def test_two_rows_whose_columns_line_up_are_a_table():
    values = [_box_stroke(10, 10, 13, 14)] + [_box_stroke(x, 10, x + 2, 14) for x in (25, 35, 45, 55)]  # y 0 1 2 3
    probabilities = [_box_stroke(8, 18.5, 16, 22.5)] + [_box_stroke(x - 1, 18.5, x + 3, 22.5) for x in (25, 35, 45, 55)]
    [(idx, rows)] = find_tables(values + probabilities)
    assert rows == [[[0], [1], [2], [3], [4]], [[5], [6], [7], [8], [9]]]
    assert find_tables([_box_stroke(*b) for b in _letters(10, 10, 12) + _letters(10, 20, 12)]) == []  # writing


def test_lines_crossing_are_a_drawing_and_writing_is_not():
    letters = [_box_stroke(x, 50, x + 3, 54) for x in (10, 15, 20, 25)]
    lines = [_box_stroke(10, 20, 40, 20.5), _box_stroke(20, 5, 20.5, 35)]  # a level line across an upright one
    label = _box_stroke(30, 10, 33, 14)
    bar, integral = _box_stroke(60, 30, 80, 30.4), _box_stroke(58, 22, 59, 40)  # meeting at an end: writing
    [(idx, box)] = find_drawings(letters + lines + [label, bar, integral])
    assert idx == [4, 5, 6] and box == (10, 5, 40, 35)


def _pdf_with_squares(squares):
    """A one-page PDF with a filled square path per (x, y, size, grey level), in that drawing order."""
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(200, 100)
    for x, y, s, grey in squares:
        obj = pdfium_c.FPDFPageObj_CreateNewPath(x, y)
        for px, py in ((x + s, y), (x + s, y + s), (x, y + s)):
            pdfium_c.FPDFPath_LineTo(obj, px, py)
        pdfium_c.FPDFPath_Close(obj)
        pdfium_c.FPDFPageObj_SetFillColor(obj, grey, grey, grey, 255)
        pdfium_c.FPDFPath_SetDrawMode(obj, pdfium_c.FPDF_FILLMODE_WINDING, 0)
        pdfium_c.FPDFPage_InsertObject(page.raw, obj)
    pdfium_c.FPDFPage_GenerateContent(page.raw)
    return pdf, page


def test_pdf_strokes_reads_dark_filled_paths_in_order_with_y_down():
    pdf, page = _pdf_with_squares([(10, 80, 5, 0), (60, 40, 5, 230), (20, 80, 5, 40)])  # the middle one is light
    strokes = pdf_strokes(page)
    assert len(strokes) == 2
    assert np.allclose(strokes[0].box, (10, 15, 15, 20)) and np.allclose(strokes[1].box, (20, 15, 25, 20))
    assert not any(s.clipped for s in strokes)
    assert len(page_lines(page)) == 1  # both on one row, close together: one line


def test_render_ink_fills_outlines():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    img = render_ink([square, square + [20, 0]], filled=True)
    h, w = img.shape
    assert img[h // 2, int(w * 0.2)] < 64 and img[h // 2, w // 2] > 192  # inside a square / between them
