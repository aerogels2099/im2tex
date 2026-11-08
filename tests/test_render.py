import shutil

import pytest

from im2tex.metrics import image_match
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
