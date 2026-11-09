import pytest

from im2tex.normalize import NormalizeError, normalize
from im2tex.page import PageError, normalize_page, page_metrics, split_page


def test_split_page_finds_inline_and_display_math_in_order():
    assert split_page("Let $x^2$ be\n$$y = 1$$\ncosts \\$5") == [
        ("text", "Let "), ("inline", "x^2"), ("text", " be\n"), ("display", "y = 1"), ("text", "\ncosts \\$5")]


@pytest.mark.parametrize("text", ["the $x^2 never closes", "$$a = b$", "$a$$b$"])
def test_split_page_rejects_unclosed_or_mixed_delimiters(text):
    with pytest.raises(PageError):
        split_page(text)


def test_normalize_page_collapses_whitespace_and_normalizes_each_formula():
    page = "Every  matrix\n\n$$A=A^T$$\nhas $x^2_i$  entries"
    assert normalize_page(page) == f"Every matrix $${normalize('A=A^T')}$$ has ${normalize('x^2_i')}$ entries"
    with pytest.raises(NormalizeError):
        normalize_page(r"bad $\frac{a$")
    assert normalize_page(r"bad $\frac{a$", strict=False) == r"bad $\frac{a$"


def test_page_metrics_score_text_and_formulas_separately():
    ref = "Let $x^2$ and $$y=1$$ hold"
    perfect = page_metrics(["Let  $x ^ {2}$\nand $$y = 1$$ hold"], [ref])  # same page, other spacing
    assert perfect["page_edit"] == 0 and perfect["text_cer"] == 0 and perfect["formula_recall"] == 1
    half = page_metrics(["Let $x^3$ and $$y=1$$ hold"], [ref])
    assert half["text_cer"] == 0 and half["formula_recall"] == 0.5 and half["formula_precision"] == 0.5
    assert 0 < half["page_edit"] < 0.1
    broken = page_metrics(["Let $x^2 and y hold"], [ref])  # unclosed $: read as text, no formulas found
    assert broken["formula_recall"] == 0 and broken["formula_precision"] == 0
