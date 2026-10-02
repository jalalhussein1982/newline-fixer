"""Tokens, gap classes, normalization and reconstruction (design section 2.1)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import IntEnum


class Gap(IntEnum):
    """What stands between two consecutive tokens."""

    JOIN = 0
    SPACE = 1
    NL = 2
    PARA = 3


GAP_STR: dict[Gap, str] = {Gap.JOIN: "", Gap.SPACE: " ", Gap.NL: "\n", Gap.PARA: "\n\n"}

_TOKEN_RE = re.compile(r"\S+")
_WS_RE = re.compile(r"\s+")


def classify_ws(ws: str) -> Gap:
    """Map a whitespace run to its gap class. The empty string is JOIN."""
    if ws == "":
        return Gap.JOIN
    newlines = ws.replace("\r\n", "\n").replace("\r", "\n").count("\n")
    if newlines >= 2:
        return Gap.PARA
    if newlines == 1:
        return Gap.NL
    return Gap.SPACE


def split(text: str) -> tuple[list[str], list[Gap]]:
    """Split text into non-whitespace tokens and the gap class between each pair."""
    tokens: list[str] = []
    gaps: list[Gap] = []
    end = 0
    for m in _TOKEN_RE.finditer(text):
        if tokens:
            gaps.append(classify_ws(text[end : m.start()]))
        tokens.append(m.group())
        end = m.end()
    return tokens, gaps


def join(tokens: Sequence[str], gaps: Sequence[Gap]) -> str:
    """Rebuild canonical text from tokens and gap classes."""
    expected = max(len(tokens) - 1, 0)
    if len(gaps) != expected:
        raise ValueError(f"{len(tokens)} tokens need {expected} gaps, got {len(gaps)}")
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i < len(gaps):
            parts.append(GAP_STR[gaps[i]])
    return "".join(parts)


def normalize(text: str) -> str:
    """Canonical whitespace form of text: split then join."""
    tokens, gaps = split(text)
    return join(tokens, gaps)


def content(text: str) -> str:
    """The text with every whitespace character removed."""
    return _WS_RE.sub("", text)


def gap_after_char(text: str) -> list[Gap]:
    """For each non-whitespace character, the gap class that follows it.

    Characters inside a token are followed by JOIN; the last character of the last
    token is followed by JOIN as well.
    """
    tokens, gaps = split(text)
    out: list[Gap] = []
    for i, tok in enumerate(tokens):
        out.extend([Gap.JOIN] * (len(tok) - 1))
        out.append(gaps[i] if i < len(gaps) else Gap.JOIN)
    return out


def derive_labels(corrupted: str, clean: str) -> list[Gap]:
    """Target gap class for every gap of `corrupted`, read off `clean` (design 2.2)."""
    if content(corrupted) != content(clean):
        raise ValueError("corrupted and clean text differ in non-whitespace content")
    after = gap_after_char(clean)
    tokens, _ = split(corrupted)
    labels: list[Gap] = []
    n = 0
    for tok in tokens[:-1]:
        n += len(tok)
        labels.append(after[n - 1])
    return labels


def unreachable_count(source: str, target: str) -> int:
    """Count target breaks that fall where `source` has no gap (design 2.3)."""
    if content(source) != content(target):
        raise ValueError("source and target differ in non-whitespace content")
    src = gap_after_char(source)
    tgt = gap_after_char(target)
    return sum(1 for s, t in zip(src, tgt, strict=True) if s is Gap.JOIN and t is not Gap.JOIN)
