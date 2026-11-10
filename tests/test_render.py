import shutil
import threading
import time

import pytest

from im2tex.metrics import image_match
from im2tex.render import latex_batch
from im2tex.render.latex_batch import render

pytestmark = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")


def test_equivalent_sources_match_and_different_ones_dont():
    a, b, c = render([r"x^2", r"x^{2}", r"x_2"])
    assert image_match(a, b)
    assert not image_match(a, c)


def test_broken_formulas_are_isolated_and_order_is_kept():
    # \frac{a}{b never closes, so TeX reads to end of file and blames the wrong formula.
    formulas = ["y", r"\frac{a}{b", "z", r"\undefinedcommand", "w"]
    out = render(formulas)
    assert out[1] is None and out[3] is None
    for i in (0, 2, 4):
        assert image_match(out[i], render([formulas[i]])[0])
    assert not image_match(out[0], out[2])


def test_percent_comment_does_not_swallow_the_closing_delimiter():
    out = render([r"a % comment", "b"])
    assert image_match(out[0], render(["a"])[0])
    assert out[1] is not None


def test_comment_at_line_end_continues_the_formula():
    a, b = render(["x +%\ny", "x + y"])
    assert image_match(a, b)


def test_equation_numbers_are_dropped_instead_of_failing():
    a, b, c = render([r"x = 1 \tag{12}", r"x = 1 \tag*{(A. 4)}", "x = 1"])
    assert image_match(a, c) and image_match(b, c)


def test_trailing_space_widens_the_page_but_not_the_scale():
    # A page is a whole number of pixels; if the page were stretched to fill them, every width would get its
    # own scale and the same formula would land on different pixels.
    formulas = [r"\frac{a}{b} + \sum_{i=1}^{n} x_i^2 = 0", r"\frac{a}{b} + \sum_{i=1}^{n} x_i^2 = 0 \qquad \qquad"]
    formulas += [f + r"\hspace{%.2fpt}" % (0.13 * k) for k in range(1, 9) for f in formulas[:1]]
    out = render(formulas)
    assert all(image_match(out[0], o) for o in out[1:])


def test_parallel_batches_never_rasterize_at_once_and_render_like_one_thread(monkeypatch):
    active, peak, guard = [0], [0], threading.Lock()
    rasterize = latex_batch._rasterize

    def watched(page, dpi):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.005)  # long enough for unguarded threads to overlap
        try:
            return rasterize(page, dpi)
        finally:
            with guard:
                active[0] -= 1

    monkeypatch.setattr(latex_batch, "_rasterize", watched)
    formulas = [rf"x_{{{i}}} + \frac{{{i}}}{{y}}" for i in range(64)]
    parallel = render(formulas, batch_size=8, workers=8)
    assert peak[0] == 1
    assert all(p is not None and (p == s).all() for p, s in zip(parallel, render(formulas, workers=1)))
