import numpy as np
import pytest

from im2tex.metrics import image_match, image_metrics, string_metrics


def test_perfect_predictions():
    refs = [r"\frac{a}{b}", r"x^{2}+y^{2}", r"\alpha"]
    m = string_metrics(refs, refs)
    assert m["bleu4"] == pytest.approx(1.0)
    assert m["edit_dist"] == 0
    assert m["exprate"] == 1


def test_exprate_counts_token_errors():
    refs = [r"\frac{a}{b}"] * 3
    preds = [r"\frac{a}{b}", r"\frac{a}{c}", r"\frac{x}{c}"]
    m = string_metrics(preds, refs)
    assert m["exprate"] == pytest.approx(1 / 3)
    assert m["exprate_le1"] == pytest.approx(2 / 3)
    assert m["exprate_le2"] == 1


def _glyph(extra_col=False):
    img = np.full((5, 5), 255, np.uint8)
    img[1:4, 2] = 0
    if extra_col:
        img[1, 3] = 0
    return img


def test_image_match_ignores_whitespace_borders():
    assert image_match(_glyph(), np.pad(_glyph(), 7, constant_values=255))
    assert not image_match(_glyph(), _glyph(extra_col=True))


def test_image_metrics_skips_failed_refs_and_counts_failed_preds():
    g = _glyph()
    m = image_metrics([g, None, g], [g, g, None])
    assert m["n_image"] == 2
    assert m["render_rate"] == 0.5
    assert m["image_match"] == 0.5
