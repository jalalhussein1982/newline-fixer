"""Seeded corruption of clean text (design section 3.3)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .text import GAP_STR, Gap, content, join, split


@dataclass(frozen=True)
class CorruptConfig:
    clean_fraction: float = 0.1
    remove_base: float = 0.6
    remove_scale: float = 0.4
    insert_min_chars: int = 40
    insert_max_chars: int = 200
    word_split_prob: float = 0.4
    double_prob: float = 0.1


@dataclass(frozen=True)
class Corrupted:
    text: str
    severity: float


def corrupt(clean: str, rng: random.Random, cfg: CorruptConfig | None = None) -> Corrupted:
    """Corrupt `clean`. Severity 0 means the canonical form of the input, unchanged."""
    cfg = cfg or CorruptConfig()
    tokens, gaps = split(clean)
    base = join(tokens, gaps)
    if len(tokens) < 2 or rng.random() < cfg.clean_fraction:
        return Corrupted(base, 0.0)
    severity = 1.0 - rng.random()  # in (0, 1]

    p_remove = cfg.remove_base + cfg.remove_scale * severity
    ws = [GAP_STR[g] for g in gaps]
    for i, g in enumerate(gaps):
        if g in (Gap.NL, Gap.PARA) and rng.random() < p_remove:
            ws[i] = " "

    chars = list(_interleave(tokens, ws))
    span = rng.uniform(cfg.insert_min_chars, cfg.insert_max_chars)
    n_insert = round(len(chars) * severity / span)
    positions = sorted((rng.randrange(len(chars)) for _ in range(n_insert)), reverse=True)
    for p in positions:
        _insert_newline(chars, p, rng, cfg)

    text = "".join(chars)
    if content(text) != content(clean):
        raise AssertionError("corruptor changed non-whitespace content")
    return Corrupted(text, severity)


def _interleave(tokens: Sequence[str], ws: Sequence[str]) -> str:
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i < len(ws):
            parts.append(ws[i])
    return "".join(parts)


def _insert_newline(chars: list[str], p: int, rng: random.Random, cfg: CorruptConfig) -> None:
    nl = "\n\n" if rng.random() < cfg.double_prob else "\n"
    if chars[p].isspace():
        _replace_run(chars, p, nl, rng)
    elif p > 0 and chars[p - 1].isspace():
        _replace_run(chars, p - 1, nl, rng)
    elif p > 0 and rng.random() < cfg.word_split_prob:
        chars[p:p] = list(nl)


def _replace_run(chars: list[str], p: int, nl: str, rng: random.Random) -> None:
    a = p
    while a > 0 and chars[a - 1].isspace():
        a -= 1
    b = p
    while b + 1 < len(chars) and chars[b + 1].isspace():
        b += 1
    chars[a : b + 1] = list(rng.choice([nl, nl + " ", " " + nl]))
