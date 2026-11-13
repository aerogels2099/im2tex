import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import make_mixed_lines as m  # noqa: E402

from im2tex.page import split_page  # noqa: E402


def _line(words, gap=30, letter=12, h=128):
    """A fake written line: each letter a dark block, 3 px between letters, `gap` px between words, on gray paper.
    -> (image array, x of each word gap's middle)."""
    a = np.full((h, 20 + sum(len(w) * (letter + 3) for w in words) + gap * len(words)), 230, np.uint8)
    x, mids = 10, []
    for w in words:
        for _ in w:
            a[50:80, x: x + letter] = 20
            x += letter + 3
        mids.append(x - 3 + gap // 2)
        x += gap - 3
    return a, mids[:-1]


def test_word_cuts_finds_the_gaps_between_words():
    words = ["the", "map", "is", "linear"]
    a, mids = _line(words)
    cuts, widths = m.word_cuts(m.components(a)[0], words)
    assert np.allclose(cuts, mids, atol=1) and widths == [30, 30, 30]


def test_word_cuts_gives_up_when_gaps_and_spaces_disagree():
    a, _ = _line(["the", "mapis", "linear"])  # written as three words, labeled as four
    assert m.word_cuts(m.components(a)[0], ["the", "map", "is", "linear"]) is None
    a, _ = _line(["he", "said", '"no"'])
    assert m.word_cuts(m.components(a)[0], ["he", "said", '"no"']) is None  # quotes are often written apart


def test_symbol_px_lets_specks_barely_count():
    hwa = np.array([[40, 30, 900]] + [[3, 3, 30]] * 20)  # one symbol, twenty specks
    assert m.symbol_px(hwa) == 40


def test_whiten_makes_noisy_paper_white_and_keeps_ink_dark():
    rng = np.random.default_rng(0)
    a = rng.normal(225, 6, (60, 200)).clip(0, 255).astype(np.uint8)
    a[20:40, 50:150] = 30
    out = m.whiten(a, m.components(a)[0])
    assert np.mean(out[:10] == 255) > 0.85 and out[30, 100] < 40


def test_glue_puts_formulas_among_a_run_of_the_lines_words(tmp_path):
    words = ["Let", "the", "map", "be", "linear", "here"]
    a, mids = _line(words)
    Image.fromarray(a).save(tmp_path / "line.png")
    f = np.full((80, 120), 255, np.uint8)
    f[20:60, 10:40], f[20:60, 70:100] = 0, 0
    Image.fromarray(f).save(tmp_path / "f.png")
    m._init([], [([(str(tmp_path / "f.png"), "x _ { i }")], 1.0)], 0)
    line = (str(tmp_path / "line.png"), words, *m.word_cuts(m.components(a)[0], words))
    for seed in range(20):
        img, label = m.glue(np.random.default_rng(seed), line)
        parts = split_page(label)
        assert {c for k, c in parts if k == "inline"} == {" x _ { i } "}
        text = " ".join(c for k, c in parts if k == "text").split()
        assert text and any(words[i: i + len(text)] == text for i in range(len(words)))  # a run of its words
        assert img.dtype == np.uint8 and img.shape[0] >= a.shape[0]


def test_short_inline_math_has_a_letter_and_bare_numbers_are_text():
    assert all(m.short(f) for f in ["x", "T", "u _ { 1 }", r"\hat { \xi }", "x = 5", r"\varphi"])
    assert not any(m.short(f) for f in ["2 0 1 5", r"\sqrt { 2 }", "+", r"\frac { a } { b } + c"])
    assert all(m.bare_number(f) for f in ["2 0 1 5", "2 . 5", "1 , 0 0 0"])
    assert not any(m.bare_number(f) for f in ["- 5", "x", r"1 0 \%"])


def test_short_math_is_drawn_from_the_short_formulas_only(tmp_path):
    f = np.full((40, 40), 255, np.uint8)
    f[10:30, 10:30] = 0
    Image.fromarray(f).save(tmp_path / "f.png")
    m._init([], [([(str(tmp_path / "f.png"), r"\frac { a } { b } + c"), (str(tmp_path / "f.png"), "x")], 1.0)], 0)
    rng = np.random.default_rng(0)
    assert {m._formula(rng, short_math=True)[1] for _ in range(20)} == {"x"}
    assert {m._formula(rng)[1] for _ in range(50)} == {"x", r"\frac { a } { b } + c"}


def test_x_height_is_the_band_of_rows_most_ink_lies_in():
    a, _ = _line(["the", "map"])  # letters 30 px tall
    a[20:50, 12:15] = 20  # an ascender on the first letter, a thin stroke above the band
    assert m.x_height(m.components(a)[0]) == 30


def test_x_height_only_looks_at_the_main_line():
    assert all(m.x_height_only(f) for f in ["x", "u = a", r"\hat { c }", "x _ { 1 }", r"\omega ^ { 2 }"])
    assert not any(m.x_height_only(f) for f in ["T", "f", "2 x", r"\sqrt { x }", "x _ { 1 } y", "+"])
