import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prepare_ink_pages import line_label, match  # noqa: E402


def test_line_label_writes_math_normalized_between_spaced_dollars():
    assert line_label("Case 1: $n$ is even") == "Case 1: $ n $ is even"
    assert line_label(r"$$= \frac{2^x}{\ln 2} |_0^1$$") == r"$ = \frac { 2 ^ { x } } { \ln 2 } | _ { 0 } ^ { 1 } $"
    assert line_label("3.2") == "3.2"


def test_match_pairs_lines_in_order_and_skips_a_line_found_in_two_pieces():
    labels = ["$$= 2k + 1$$", "$$n^2 + n + 3$$", "$$= 4k^2 - 2k + 3$$", "Case 2: $n$ is odd"]
    reads = ["$ = 2 k + 1 $", "$ n ^ { 2 } + n $", "$ + 3 $", "$ = 4 k ^ { 2 } - 2 k + 3 $", "Case 2 : $ n $ is odd"]
    assert match(reads, labels) == [(0, 0), (3, 2), (4, 3)]


def test_match_leaves_an_unwritten_found_line_unpaired():
    labels = ["$$x = 1$$", "$$y = 2$$"]
    reads = ["$ x = 1 $", "$ \\int \\frac { 1 } { 2 } d t $", "$ y = 2 $"]  # a stray scribble found as a line
    assert match(reads, labels) == [(0, 0), (2, 1)]
