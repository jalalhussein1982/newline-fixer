"""B1: the rule-based baseline (design section 4.3)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .lexicon import Lexicon
from .text import Gap

LIST_MARKER = re.compile(r"^(?:[•\-\*–]|\d{1,3}[.)])$")
SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?$")
TERMINAL = ".!?:;"
CLOSING = ")]}.,;:"
STOP = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "with",
    "our",
    "at",
    "by",
    "from",
    "as",
    "vs",
    "into",
    "over",
    "under",
    "per",
}
MAX_HEADING_TOKENS = 8


def _heading_like(line: Sequence[str], require_number: bool) -> bool:
    if not line or len(line) > MAX_HEADING_TOKENS:
        return False
    last = line[-1]
    if last[-1] in TERMINAL or last.lower() in STOP:
        return False
    if SECTION_NUMBER.match(line[0]):
        return len(line) >= 2
    if require_number:
        return False
    capitalized = [t for t in line if t[0].isalpha() and t[0].isupper()]
    acceptable = all(t.lower() in STOP or not t[0].isalpha() or t[0].isupper() for t in line)
    return acceptable and len(capitalized) >= 1


def rules_predict(tokens: Sequence[str], current: Sequence[Gap], lexicon: Lexicon) -> list[Gap]:
    out: list[Gap] = []
    line_start = 0
    for i, cur in enumerate(current):
        left, right = tokens[i], tokens[i + 1]
        has_break = cur in (Gap.NL, Gap.PARA)
        line = tokens[line_start : i + 1]
        if (
            has_break
            and left.isalpha()
            and right.isalpha()
            and lexicon.known(left + right)
            and not (lexicon.known(left) and lexicon.known(right))
        ):
            g = Gap.JOIN
        elif has_break and (right[0].islower() or right[0] in CLOSING):
            g = Gap.SPACE
        elif LIST_MARKER.match(right) and left[-1] in TERMINAL:
            g = Gap.NL
        elif right[0].isupper() and _heading_like(line, require_number=not has_break):
            g = Gap.PARA
        else:
            g = cur
        out.append(g)
        if g in (Gap.NL, Gap.PARA):
            line_start = i + 1
    return out


class RulesFixer:
    name = "rules"
    budget = 256

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon or Lexicon.bundled()

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        return rules_predict(tokens, current, self.lexicon)
