r"""Rewrite LaTeX math into one canonical spelling, so formulas that render the same are written the same.

Every rule must leave the rendered image unchanged; scripts/check_normalizer.py verifies that on real data.
  - math tokens separated by single spaces; text inside \text{...} etc. kept verbatim
  - scripts braced, subscript first, primes as \prime:   x'^2_i       -> x _ { i } ^ { \prime 2 }
  - arguments braced:                                    \frac12      -> \frac { 1 } { 2 }
  - old font switches become commands:                   {\bf C}      -> \mathbf { C }
  - infix fractions become commands:                     {a \over b}  -> \frac { a } { b }
  - synonyms collapse:                                   \le -> \leq, \operatorname{sin} -> \sin
  - braces that change nothing are dropped:              {{x}} -> {x}, {\frac{a}{b}} -> \frac{a}{b}
  - invisible markup dropped:                            \label, \nonumber, \notag, \tag, \vspace
  - lengths kept whole:                                  \hspace { 0 . 5 i n } -> \hspace {0.5in}
  - display environments made inline:                    align -> aligned, equation -> unwrapped

Known render differences (~0.04% of im2latex): an old font switch under an accent (\hat{\cal E}) moves the
accent by a pixel, and a switch in the middle of a group (a \bf b + c) can change spacing at its edges.
"""
import re


class NormalizeError(ValueError):
    """Not valid LaTeX math: unbalanced braces, double superscript, missing argument, ..."""


_LEX = re.compile(r"\\[a-zA-Z]+|\\.|%[^\n]*|\s+|.", re.S)
_CONTROL_WORD = re.compile(r"\\[a-zA-Z]+")
_DIMEN = re.compile(r"[-+]*(?:\d+\.?\d*|\.\d+|\d*,\d+)(?:true)?(?:pt|pc|in|bp|cm|mm|dd|cc|sp|em|ex|mu)")

FONT_CMDS = [r"\mathbf", r"\mathrm", r"\mathit", r"\mathcal", r"\mathsf", r"\mathtt",
             r"\mathbb", r"\mathfrak", r"\mathscr", r"\mathnormal"]
FONT_SWITCHES = {r"\bf": r"\mathbf", r"\rm": r"\mathrm", r"\it": r"\mathit",
                 r"\cal": r"\mathcal", r"\sf": r"\mathsf", r"\tt": r"\mathtt"}
INFIX = {r"\over": r"\frac", r"\choose": r"\binom"}
SYNONYMS = {
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq", r"\to": r"\rightarrow", r"\gets": r"\leftarrow",
    r"\land": r"\wedge", r"\lor": r"\vee", r"\lnot": r"\neg", r"\owns": r"\ni",
    r"\lbrace": r"\{", r"\rbrace": r"\}", r"\lbrack": "[", r"\rbrack": "]", r"\vert": "|", r"\Vert": r"\|",
    r"\thinspace": r"\,", r"\medspace": r"\:", r"\>": r"\:", r"\thickspace": r"\;", r"\negthinspace": r"\!",
    r"\bm": r"\boldsymbol", r"\Bbb": r"\mathbb",
    "\\ ": "~", "\\\n": "~", "\\\t": "~",  # in math mode a control space and ~ are the same space
}
BRACED_SYNONYMS = {r"\dag": r"\dagger", r"\ddag": r"\ddagger"}  # in math \dag is {\dagger}: an Ord, not a Bin
# Ordinary (Ord-class) symbols: {x} and x render the same when no script follows.
ORDINARY = set(r"""\alpha \beta \gamma \delta \epsilon \varepsilon \zeta \eta \theta \vartheta \iota \kappa
\varkappa \lambda \mu \nu \xi \pi \varpi \rho \varrho \sigma \varsigma \tau \upsilon \phi \varphi \chi \psi
\omega \Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi \Omega \infty \partial \ell \hbar
\nabla \imath \jmath \aleph \emptyset \varnothing""".split())
# Common commands known to take no argument. A group after any *other* unparsed command might be its
# argument, and some commands (\textcircled, text accents like \b) treat {x} and x differently.
NO_ARG = ORDINARY | set(r"""\cdot \times \div \pm \mp \ast \star \circ \bullet \oplus \ominus \otimes \oslash
\odot \cup \cap \wedge \vee \setminus \wr \diamond \amalg \sqcup \sqcap \uplus \triangleleft \triangleright
\dagger \ddagger \leq \geq \neq \equiv \approx \sim \simeq \cong \propto \ll \gg \subset \supset \subseteq
\supseteq \in \ni \notin \mid \parallel \perp \models \vdash \dashv \prec \succ \preceq \succeq \asymp \doteq
\rightarrow \leftarrow \leftrightarrow \Rightarrow \Leftarrow \Leftrightarrow \longrightarrow \longleftarrow
\Longrightarrow \Longleftarrow \longleftrightarrow \Longleftrightarrow \mapsto \longmapsto \uparrow \downarrow
\updownarrow \Uparrow \Downarrow \nearrow \searrow \swarrow \nwarrow \hookrightarrow \hookleftarrow
\rightleftharpoons \iff \implies \sum \prod \coprod \int \iint \iiint \oint \bigcup \bigcap \bigoplus
\bigotimes \bigvee \bigwedge \bigsqcup \lim \sin \cos \tan \cot \sec \csc \sinh \cosh \tanh \log \ln \exp
\max \min \sup \inf \det \dim \ker \deg \arg \ldots \cdots \vdots \ddots \dots \quad \qquad \, \; \: \!
\langle \rangle \lfloor \rfloor \lceil \rceil \forall \exists \neg \top \bot \angle \triangle \Box \flat
\natural \sharp \prime \surd \Re \Im \wp \hslash \| \{ \} \displaystyle \textstyle \scriptstyle
\scriptscriptstyle \limits \nolimits \not \hline""".split())
OPS_NOLIMITS = set("arccos arcsin arctan arg cos cosh cot coth csc deg dim exp hom ker lg ln log "
                   "sec sin sinh tan tanh".split())
OPS_LIMITS = set("det gcd inf lim max min Pr sup".split())

# Argument specs: m = math {..}, o = optional math [..], O = optional raw [..],
# r = raw {..} (a length, colour or column spec: spaces removed), t = text {..} (kept verbatim),
# d = unbraced length (\kern3pt), D = a delimiter token, B = math {..} with inner braces kept as written,
# M = math arg kept braced or bare as written.
ARGS = {
    **{c: "m" for c in FONT_CMDS},
    **{c: "m" for c in (r"\Bbb", r"\operatorname")},
    # \boldsymbol{{L}} and \boldsymbol{L} render differently: amsbsy inspects its argument
    **{c: "B" for c in (r"\boldsymbol", r"\bm", r"\pmb")},
    **{c: "D" for c in (r"\middle", *(s + v for s in (r"\big", r"\Big", r"\bigg", r"\Bigg")
                                         for v in ("", "l", "r", "m")))},
    **{c: "mm" for c in (r"\frac", r"\dfrac", r"\tfrac", r"\binom", r"\dbinom", r"\tbinom",
                         r"\stackrel", r"\overset", r"\underset", r"\sideset")},
    **{c: "m" for c in (
        r"\hat", r"\tilde", r"\bar", r"\vec", r"\dot", r"\ddot", r"\dddot", r"\acute", r"\grave",
        r"\breve", r"\check", r"\mathring", r"\widehat", r"\widetilde", r"\overline", r"\underline",
        r"\overrightarrow", r"\overleftarrow", r"\overleftrightarrow", r"\underrightarrow",
        r"\underleftarrow", r"\overbrace", r"\underbrace", r"\phantom", r"\hphantom", r"\vphantom",
        r"\boxed", r"\cancel", r"\bcancel", r"\xcancel", r"\substack",
        r"\Hat", r"\Tilde", r"\Bar", r"\Vec", r"\Dot", r"\Ddot", r"\Acute", r"\Grave", r"\Breve", r"\Check")},
    **{c: "t" for c in (r"\text", r"\mbox", r"\hbox", r"\textrm", r"\textbf", r"\textit", r"\textsf",
                        r"\texttt", r"\textup", r"\textnormal", r"\textsc", r"\textsl", r"\textmd",
                        r"\emph", r"\fbox")},
    **{c: "d" for c in (r"\kern", r"\mkern", r"\hskip", r"\mskip")},
    r"\sqrt": "om", r"\smash": "Om", r"\xrightarrow": "om", r"\xleftarrow": "om", r"\cfrac": "Omm",
    r"\hspace": "r", r"\mspace": "r", r"\color": "r", r"\textcolor": "rm", r"\colorbox": "rt",
    r"\raisebox": "rt", r"\rule": "Orr", r"\cline": "r", r"\multicolumn": "rrm", r"\genfrac": "mmrrmm",
    r"\mathchoice": "mmmm", "\\\\": "O", r"\lefteqn": "m", r"\ensuremath": "m", r"\vcenter": "m",
    r"\setlength": "rr", r"\noalign": "r",
    # \mathop\to and \mathop{\to} render differently (a bare symbol is enlarged and centred), so keep as written
    **{c: "M" for c in (r"\mathop", r"\mathbin", r"\mathrel", r"\mathord", r"\mathopen", r"\mathclose",
                        r"\mathpunct", r"\mathinner")},
}
DROP = {r"\nonumber", r"\notag"}
DROP_ARGS = {r"\label": "r", r"\tag": "t", r"\vspace": "r"}
# The image shows something the label can't reproduce (a resolved reference number, a link, a picture).
REJECT = {r"\ref", r"\eqref", r"\cite", r"\url", r"\href", r"\includegraphics", r"\verb"}
ALIASES = {r"\sp": "^", r"\sb": "_"}  # plain TeX spellings of the script operators
STARRED = {r"\operatorname", r"\hspace", r"\vspace", r"\tag"}
# Commands whose expansion already wraps everything in braces, so {\cmd{..}} == \cmd{..}.
WRAPPED = {**{c: 1 for c in FONT_CMDS}, **{c: 2 for c in (r"\frac", r"\dfrac", r"\tfrac",
                                                            r"\binom", r"\dbinom", r"\tbinom")}}

ENV_ARGS = {"array": "r", "subarray": "r", "alignat": "r", "alignat*": "r", "alignedat": "r", "tabular": "r"}
ENV_UNWRAP = {"equation", "equation*", "displaymath", "math"}
ENV_RENAME = {  # display environments can't appear inside $...$; use their inline counterparts
    **dict.fromkeys(("align", "align*", "flalign", "flalign*", "split"), ("aligned", None)),
    **dict.fromkeys(("gather", "gather*", "multline", "multline*"), ("gathered", None)),
    **dict.fromkeys(("alignat", "alignat*"), ("alignedat", None)),
    **dict.fromkeys(("eqnarray", "eqnarray*"), ("array", "{rcl}")),
}


def normalize(latex: str) -> str:
    """Canonical spelling of a LaTeX math formula. Raises NormalizeError on invalid input."""
    p = _Parser(_strip_math_delimiters(latex.strip()))
    units = p.seq(stop=set())
    if len(units) == 1 and isinstance(units[0], _Group):
        units = units[0].inner
    if units[:1] == [[r"\displaystyle"]]:  # our renderer is already in display style
        units = units[1:]
    while units[-1:] == [["\\\\"]]:  # a trailing line break renders nothing
        units = units[:-1]
    return " ".join(_content(units))


def normalize_or_none(latex: str):
    """normalize(), or None for invalid input. Picklable, for multiprocessing."""
    try:
        return normalize(latex)
    except NormalizeError:
        return None


class _Group(list):
    """A free-standing {...} group; `inner` holds its content units."""

    def __init__(self, tokens, inner):
        super().__init__(tokens)
        self.inner = inner


class _Marker:
    """A switch that affects the rest of its group: \\bf-style fonts, or \\over-style infix."""

    def __init__(self, cmd, infix=False):
        self.cmd, self.infix = cmd, infix


class _Parser:
    def __init__(self, src):
        self.toks = [ALIASES.get(t, t) for t in _LEX.findall(src)]
        self.i = 0

    # -- token stream ---------------------------------------------------------------------------
    def peek(self, skip_space=True):
        while self.i < len(self.toks) and (self.toks[self.i][0] == "%" or
                                           (skip_space and self.toks[self.i].isspace())):
            self.i += 1
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self, skip_space=True):
        t = self.peek(skip_space)
        if t is None:
            raise NormalizeError("unexpected end of formula")
        self.i += 1
        return t

    def expect(self, tok):
        if (t := self.next()) != tok:
            raise NormalizeError(f"expected {tok!r}, got {t!r}")

    # -- math mode ------------------------------------------------------------------------------
    def seq(self, stop, unwrap=True):
        """Parse math up to (not including) a token in `stop`; return the finished units.

        unwrap=False keeps braces around lone symbols ({L}), for arguments of commands that inspect them."""
        items = []
        while (t := self.peek()) is not None and t not in stop:
            if t in ("^", "_", "'"):
                base = items.pop() if items and not isinstance(items[-1], _Marker) else []
                items.append(list(base) + self.scripts())
                continue
            u = self.atom()
            if u is None:
                continue
            if (isinstance(u, _Group) and len(u.inner) == 1 and _self_delimited(u.inner[0])
                    and not _maybe_argument(items[-1] if items else None)):
                u = u.inner[0]  # {\bf C} -> \mathbf{C}, {{x}} -> {x}
            items.append(u)
        # {x} -> x for a lone ordinary symbol, once we know no script attached to the group
        if unwrap:
            items = [u.inner[0] if isinstance(u, _Group) and len(u.inner) == 1 and _ordinary(u.inner[0])
                     and not _maybe_argument(items[k - 1] if k else None) else u
                     for k, u in enumerate(items)]
        return _finalize(items)

    def atom(self):
        t = self.next()
        if t == "{":
            inner = self.seq({"}"})
            self.expect("}")
            return _Group(["{", *_flat(inner), "}"], inner)
        if t in ("}", "$", r"\end", r"\right"):
            raise NormalizeError(f"unexpected {t!r}")
        if t in REJECT:
            raise NormalizeError(f"{t} cannot be read back from an image")
        if t == r"\begin":
            return self.env()
        if t == r"\left":  # \left..\right is a group: font switches and \over stop at \right
            left = self.delimiter()
            inner = self.seq({r"\right"})
            self.expect(r"\right")
            return [r"\left", left, *_flat(inner), r"\right", self.delimiter()]
        if t in FONT_SWITCHES:
            return _Marker(FONT_SWITCHES[t])
        if t in INFIX:
            return _Marker(INFIX[t], infix=True)
        if t in BRACED_SYNONYMS:
            return _Group(["{", BRACED_SYNONYMS[t], "}"], [[BRACED_SYNONYMS[t]]])
        star = t in STARRED and self.peek() == "*"
        if star:
            self.next()
        if t in DROP:
            return None
        if t in DROP_ARGS:
            self.args(DROP_ARGS[t])
            return None
        if t == r"\operatorname":
            return self.operatorname(star)
        name = t + "*" * star
        return [SYNONYMS.get(name, name), *self.args(ARGS.get(t, ""))]

    def scripts(self):
        sub = sup = None
        primes = []
        while (t := self.peek()) in ("^", "_", "'"):
            self.next()
            if t == "_":
                if sub is not None:
                    raise NormalizeError("double subscript")
                sub = self.math_arg()
            elif sup is not None:
                raise NormalizeError("double superscript")
            elif t == "'":
                primes.append(r"\prime")
            else:
                sup = self.math_arg()
        out = ["_", "{", *sub, "}"] if sub is not None else []
        if primes or sup is not None:
            out += ["^", "{", *primes, *(sup or []), "}"]
        return out

    def math_arg(self, unwrap=True):
        """A required math argument: {...}, or a single token together with its own arguments."""
        if self.peek() == "{":
            self.next()
            inner = self.seq({"}"}, unwrap)
            self.expect("}")
            return _content(inner) if unwrap else _flat(inner)
        if self.peek() in (None, "}", "^", "_", "&"):
            raise NormalizeError("missing argument")
        u = self.atom()
        if isinstance(u, _Marker):
            raise NormalizeError(f"{u.cmd} cannot be an argument")
        return _content([u]) if u else []

    def args(self, spec):
        out = []
        for kind in spec:
            if kind in "oO":
                if self.peek() != "[":
                    continue
                self.next()
                if kind == "o":
                    out += ["[", *_flat(self.seq({"]"})), "]"]
                    self.expect("]")
                else:
                    out.append("[" + self.raw_until("]") + "]")
            elif kind in "mB" or kind == "M" and self.peek() == "{":
                out += ["{", *self.math_arg(unwrap=kind != "B"), "}"]
            elif kind == "M":
                out += self.math_arg()
            elif kind == "D":
                out.append(self.delimiter())
            elif kind in "tr":  # kept as one glued token: {if }, {0.5in}
                if self.peek() == "{":
                    self.next()
                    body = self.text_until("}") if kind == "t" else self.raw_until("}")
                else:
                    body = self.next()
                out.append("{" + body + "}")
            elif kind == "d":
                out.append(self.dimen())
        return out

    def raw_until(self, close):
        """Content up to `close` with spaces removed: lengths, colour names, column specs."""
        parts, depth = [], 0
        while (t := self.next()) != close or depth:
            depth += (t == "{") - (t == "}")
            parts.append(t)
        return "".join(parts)

    def dimen(self):
        """An unbraced length such as the -3mu in \\mkern-3mu."""
        start = self.i
        chars = ""
        while (t := self.peek()) is not None and len(t) == 1 and not t.isspace() and len(chars) < 16:
            chars += t
            self.i += 1
        m = _DIMEN.match(chars)
        if not m:
            raise NormalizeError("expected a length")
        self.i = start
        for _ in range(len(m.group())):
            self.next()
        return m.group()

    def delimiter(self):
        t = self.next()
        return SYNONYMS.get(t, t)

    def operatorname(self, star):
        arg = self.math_arg()
        name = "".join(arg)
        if (name in OPS_LIMITS) if star else (name in OPS_NOLIMITS):
            return ["\\" + name]
        return [r"\operatorname*" if star else r"\operatorname", "{", *arg, "}"]

    def env(self):
        name = self.raw_until("}") if self.next() == "{" else None
        if not name:
            raise NormalizeError(r"bad \begin")
        args = self.args(ENV_ARGS.get(name, ""))
        body = []
        while True:  # each cell is its own group, so parse cell by cell
            body += _flat(self.seq({"&", "\\\\", r"\end"}))
            sep = self.next()
            if sep == r"\end":
                break
            body.append(sep)
            if sep == "\\\\":
                body += self.args("O")
        if body[-1:] == ["\\\\"]:  # a final \\ before \end adds no row
            body.pop()
        self.expect("{")
        if self.raw_until("}") != name:
            raise NormalizeError(rf"\begin{{{name}}} closed by a different \end")
        if name in ENV_UNWRAP:
            return body
        new, new_args = ENV_RENAME.get(name, (name, None))
        return [rf"\begin{{{new}}}", *([new_args] if new_args else args), *body, rf"\end{{{new}}}"]

    # -- text mode ------------------------------------------------------------------------------
    def text_until(self, close):
        """Text-mode content up to `close`, kept verbatim except that whitespace runs become one space."""
        out, depth, after_word = "", 0, False
        while (t := self.next(skip_space=False)) != close or depth:
            if t == "$":
                out += "$" + " ".join(_flat(self.seq({"$"}))) + "$"
                self.expect("$")
                after_word = False
                continue
            depth += (t == "{") - (t == "}")
            if t.isspace():
                if not after_word and not out.endswith(" "):  # TeX ignores spaces after control words
                    out += " "
                continue
            if after_word and t[0].isalpha():
                out += " "
            out += t
            after_word = bool(_CONTROL_WORD.fullmatch(t))
        return out


def _finalize(items):
    """Apply group-level switches: \\over splits the group in two; \\bf applies to the rest of it."""
    infix = [k for k, u in enumerate(items) if isinstance(u, _Marker) and u.infix]
    if len(infix) > 1:
        raise NormalizeError("ambiguous infix fraction")
    if not infix:
        return _apply_fonts(items, None)[0]
    k = infix[0]
    num, font = _apply_fonts(items[:k], None)
    den, _ = _apply_fonts(items[k + 1:], font)  # a font switch carries over into the denominator
    return [[items[k].cmd, "{", *_content(num), "}", "{", *_content(den), "}"]]


def _apply_fonts(items, font):
    out, buf = [], []
    for u in [*items, None]:
        if u is None or isinstance(u, _Marker):
            if font and buf:
                out.append([font, "{", *_content(buf), "}"])
            else:
                out.extend(buf)
            buf = []
            if u is not None:
                font = u.cmd
        else:
            buf.append(u)
    return out, font


def _self_delimited(unit):
    """True if braces around this unit change nothing: it is a group, or \\cmd{..} that braces itself."""
    if isinstance(unit, _Group):
        return True
    n = WRAPPED.get(unit[0]) if unit else None
    j = 1
    for _ in range(n or 0):
        if j >= len(unit) or unit[j] != "{":
            return False
        j = _closing(unit, j) + 1
    return n is not None and j == len(unit)


def _maybe_argument(prev):
    """Could a group right after `prev` be an argument we didn't parse? True after an unknown command."""
    return (isinstance(prev, list) and len(prev) == 1 and bool(_CONTROL_WORD.fullmatch(prev[0]))
            and prev[0] not in NO_ARG and prev[0] not in ARGS)


def _ordinary(unit):
    return len(unit) == 1 and (unit[0] in ORDINARY or (len(unit[0]) == 1 and unit[0].isalnum()))


def _closing(tokens, j):
    depth = 0
    for k in range(j, len(tokens)):
        depth += (tokens[k] == "{") - (tokens[k] == "}")
        if depth == 0:
            return k
    raise NormalizeError("unbalanced braces")


def _content(units):
    """Tokens of units that are about to be wrapped in braces; a lone group loses its own: {{a}} -> {a}."""
    if len(units) == 1 and isinstance(units[0], _Group):
        units = units[0].inner
    return _flat(units)


def _flat(units):
    return [t for u in units for t in u]


def _strip_math_delimiters(s):
    for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if s.startswith(left) and s.endswith(right) and len(s) >= len(left) + len(right):
            return s[len(left):len(s) - len(right)]
    return s
