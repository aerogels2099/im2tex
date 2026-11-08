"""Split LaTeX into tokens for metrics: \\commands, escaped symbols, then single characters."""
import re

_TOKEN = re.compile(r"\\[a-zA-Z]+\*?|\\.|\S")


def tokenize(latex: str) -> list[str]:
    return _TOKEN.findall(latex)
