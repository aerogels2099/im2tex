import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_gnhk as g  # noqa: E402

from im2tex.page import split_page  # noqa: E402


def test_label_attaches_punctuation_escapes_dollars_and_drops_crossed_out_words():
    assert g.label(["I", "paid", "%SC%", "$28", ",", "(", "cash", ")", "."]) == r"I paid \$28, (cash)."
    assert split_page(g.label(["$5", "and", "$6"])) == [("text", r"\$5 and \$6")]


def test_label_skips_unreadable_math_and_math_like_lines():
    for words in (["see", "%NA%"], ["%math%", "list"], ["x", "=", "2"], ["a+b=c"], ["go", "->", "home"],
                  ["%SC%"]):
        assert g.label(words) is None, words
    assert g.label(["well-known", "&", "/", "%"]) == "well-known & / %"


def _word(x0, y0, x1, y1, text="w"):
    return {"text": text, "polygon": {"x0": x0, "y0": y0, "x1": x1, "y1": y0, "x2": x1, "y2": y1, "x3": x0, "y3": y1},
            "line_idx": 0, "type": "H"}


def _rotate_word(word, angle, center):
    c, s = math.cos(angle), math.sin(angle)
    p = word["polygon"]
    for k in range(4):
        x, y = p[f"x{k}"] - center[0], p[f"y{k}"] - center[1]
        p[f"x{k}"], p[f"y{k}"] = center[0] + c * x - s * y, center[1] + s * x + c * y
    return word


def test_cut_straightens_a_tilted_line_and_orders_its_words():
    # three dark word blocks on a line tilted 20 degrees (down to the right), annotated out of order
    angle, center = math.radians(20), (300, 300)
    words = [_word(100 + 140 * k, 280, 200 + 140 * k, 320, text=t) for k, t in enumerate("abc")]
    words = [_rotate_word(w, angle, center) for w in words]
    image = Image.new("RGB", (600, 600), "white")
    d = ImageDraw.Draw(image)
    for w in words:
        d.polygon([tuple(x) for x in g.corners(w)], fill="black")
    shuffled = [words[2], words[0], words[1]]
    assert abs(math.degrees(g.direction(words)) - 20) < 0.5
    assert abs(math.degrees(g.direction(words[::-1])) - 20) < 0.5  # GNHK sometimes lists a line backwards
    assert [w["text"] for w in g.left_to_right(shuffled, angle)] == ["a", "b", "c"]
    crop = np.asarray(g.cut(image, shuffled, angle).convert("L"))
    ink_rows = np.flatnonzero((crop < 128).any(1))
    assert ink_rows.size <= 42  # straight: the 40 px word height, not the 160 px a tilted crop would span
    assert crop.shape[1] > 370  # all three words, end to end


def test_cut_paints_out_ink_outside_the_words():
    # one word reaches down 50 px further, so the crop spans ink of the next line under the first word
    words = [_word(100, 280, 200, 320), _word(240, 280, 340, 320), _word(380, 280, 480, 370)]
    image = Image.new("RGB", (600, 600), "white")
    d = ImageDraw.Draw(image)
    for w in words:
        d.polygon([tuple(x) for x in g.corners(w)], fill="black")
    d.rectangle((100, 345, 200, 365), fill="black")  # the next line, under the first word
    crop = np.asarray(g.cut(image, words, 0.0).convert("L"))
    # crop origin is (100 - 6, 280 - 6): 0.15 of the 40 px median word height
    assert (crop[71:92, 6:107] > 200).all()  # painted paper
    assert (crop[20:90, 300:370] < 128).mean() > 0.9  # the tall word itself stays


def test_cut_refuses_a_line_taller_than_its_words():
    words = [_word(0, 0, 100, 40), _word(150, 200, 250, 240)]  # "one line" whose words are 200 px apart
    assert g.cut(Image.new("RGB", (300, 300), "white"), words, 0.0) is None
