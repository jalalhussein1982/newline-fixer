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
