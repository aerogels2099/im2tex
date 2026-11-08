from im2tex.tokenize_latex import tokenize


def test_commands_braces_and_chars():
    assert tokenize(r"\frac{a}{b}") == [r"\frac", "{", "a", "}", "{", "b", "}"]
    assert tokenize(r"\sum_{i=1}^n") == [r"\sum", "_", "{", "i", "=", "1", "}", "^", "n"]


def test_escapes_and_spacing_commands():
    assert tokenize(r"\left( x \right) \\ y\,z") == [r"\left", "(", "x", r"\right", ")", r"\\", "y", r"\,", "z"]
    assert tokenize(r"\operatorname*{d}") == [r"\operatorname*", "{", "d", "}"]


def test_whitespace_is_ignored():
    assert tokenize(r"\frac { a } { b }") == tokenize(r"\frac{a}{b}")
