import pytest

from im2tex.normalize import NormalizeError, normalize

CASES = [
    # scripts
    (r"x^2", r"x ^ { 2 }"),
    (r"x^2_i", r"x _ { i } ^ { 2 }"),
    (r"x'^2_i", r"x _ { i } ^ { \prime 2 }"),
    (r"f''(x)", r"f ^ { \prime \prime } ( x )"),
    (r"x^\frac12", r"x ^ { \frac { 1 } { 2 } }"),
    (r"x^{{a}}", r"x ^ { a }"),
    (r"{}^{14}C", r"{ } ^ { 1 4 } C"),
    # arguments and fractions
    (r"\frac12", r"\frac { 1 } { 2 }"),
    (r"\sqrt[3]x", r"\sqrt [ 3 ] { x }"),
    (r"{a \over b}", r"\frac { a } { b }"),
    (r"n \choose k", r"\binom { n } { k }"),
    # fonts
    (r"{\bf C}_i", r"\mathbf { C } _ { i }"),
    (r"{a \bf b \over c}", r"\frac { a \mathbf { b } } { \mathbf { c } }"),
    (r"\bm x", r"\boldsymbol { x }"),
    # synonyms and operators
    (r"a \le b \to \lbrace", r"a \leq b \rightarrow \{"),
    (r"\operatorname{sin} x", r"\sin x"),
    (r"\operatorname*{max}_i", r"\max _ { i }"),
    (r"\operatorname{max}", r"\operatorname { m a x }"),
    # braces that change nothing
    (r"{{x}}", r"x"),
    (r"a{{x}}", r"a { x }"),
    (r"{\frac{a}{b}}c", r"\frac { a } { b } c"),
    (r"\int {\cal D} \phi", r"\int \mathcal { D } \phi"),
    (r"{ \cal { S } }", r"\mathcal { S }"),
    (r"{a} \over b", r"\frac { a } { b }"),
    (r"x \sp 2 \sb i", r"x _ { i } ^ { 2 }"),
    # scopes: font switches stop at \right and at cell boundaries
    (r"\left( \cal x \right)^2", r"\left ( \mathcal { x } \right ) ^ { 2 }"),
    (r"\left\lbrace a \over b \right.", r"\left \{ \frac { a } { b } \right ."),
    (r"\begin{matrix} \bf a & b \\ c \end{matrix}", r"\begin{matrix} \mathbf { a } & b \\ c \end{matrix}"),
    # \mathop\to and \mathop{\to} render differently; so do \boldsymbol{{L}} and \boldsymbol{L}
    (r"\mathop\to_x", r"\mathop \rightarrow _ { x }"),
    (r"\mathop{\to}_x", r"\mathop { \rightarrow } _ { x }"),
    (r"\boldsymbol{{L}}", r"\boldsymbol { { L } }"),
    # commands that take an argument must keep it
    (r"\Bar{\mathcal{Z}}", r"\Bar { \mathcal { Z } }"),
    (r"\Gamma^\big( x \big\rbrace", r"\Gamma ^ { \big ( } x \big \}"),
    # text, lengths, invisible markup
    (r"\text{if }x", r"\text {if } x"),
    (r"\text{for all $x^2$}", r"\text {for all $x ^ { 2 }$}"),
    (r"\mbox{\bf  bold   text}", r"\mbox {\bf bold text}"),
    (r"\hspace { 0 . 5 i n } a", r"\hspace {0.5in} a"),
    (r"a\mkern-3mu b", r"a \mkern -3mu b"),
    (r"\mathrm{d}x\label{eq:1}\nonumber", r"\mathrm { d } x"),
    (r"\displaystyle\sum_i", r"\sum _ { i }"),
    (r"{\displaystyle a + {b}}", r"a + { b }"),
    (r"$$ x $$", r"x"),
    # environments
    (r"\begin{align} a &= b \\ c &= d \end{align}", r"\begin{aligned} a & = b \\ c & = d \end{aligned}"),
    (r"\begin{array}{c c} 1 & 2 \\[2pt] 3 & 4 \end{array}", r"\begin{array} {cc} 1 & 2 \\ [2pt] 3 & 4 \end{array}"),
    (r"\begin{equation} E = mc^2 \end{equation}", r"E = m c ^ { 2 }"),
    # an im2latex label and a pix2tex prediction of the same image normalize identically
    (r"\frac { d } { d s } { \bf C } _ { i }", r"\frac { d } { d s } \mathbf { C } _ { i }"),
    (r"{\frac{d}{d s}}\mathbf{C}_{i}", r"\frac { d } { d s } \mathbf { C } _ { i }"),
]


@pytest.mark.parametrize("src, expected", CASES)
def test_normalize(src, expected):
    assert normalize(src) == expected


@pytest.mark.parametrize("src", [src for src, _ in CASES])
def test_idempotent(src):
    once = normalize(src)
    assert normalize(once) == once


@pytest.mark.parametrize("src", [
    r"x^a^b", r"x_a_b", r"x^a'", r"\frac{a}{b", r"{a", r"a}", r"\frac{a}",
    r"\begin{matrix} a \end{pmatrix}", r"{a \over b \over c}", r"\kern x",
    r"\left( a", r"a \right)",
    r"x = \eqref{eq:1}",  # the image shows a resolved number the label can't reproduce
])
def test_invalid_input_raises(src):
    with pytest.raises(NormalizeError):
        normalize(src)
