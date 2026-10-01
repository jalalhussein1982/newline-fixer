"""Windowing, merging and the single `fix` entry point (design section 4.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .models.base import Fixer
from .text import Gap, join, split


def make_windows(
    n: int,
    token_costs: Sequence[int],
    gap_costs: Sequence[int],
    overhead: int,
    budget: int,
) -> list[tuple[int, int]]:
    """Greedy windows [start, end) over n tokens, each within budget, overlapping by half.

    Raises ValueError if a token costs more than half the budget after overhead, or if
    any two adjacent tokens plus the gap between them exceed the budget after overhead,
    because such a gap could be left uncovered. With zero gap costs the half-budget
    per-token cap alone is sufficient.
    """
    if n == 0:
        return []
    if len(token_costs) != n or len(gap_costs) != max(n - 1, 0):
        raise ValueError("cost lists do not match token count")
    room = budget - overhead
    limit = room // 2
    for i, c in enumerate(token_costs):
        if c > limit:
            raise ValueError(f"token {i} costs {c}, above half the budget {limit}")
    for i in range(n - 1):
        pair = token_costs[i] + gap_costs[i] + token_costs[i + 1]
        if pair > room:
            raise ValueError(
                f"tokens {i} and {i + 1} with their gap cost {pair}, above room {room}"
            )
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = start + 1
        cost = token_costs[start]
        while end < n:
            extra = gap_costs[end - 1] + token_costs[end]
            if cost + extra > room:
                break
            cost += extra
            end += 1
        windows.append((start, end))
        if end >= n:
            return windows
        start += max(1, (end - start) // 2)


def assign_gaps(windows: Sequence[tuple[int, int]], n_gaps: int) -> list[int]:
    """Owner window per gap: the window containing both tokens whose center is nearest."""
    owner = [-1] * n_gaps
    best = [float("inf")] * n_gaps
    for w, (s, e) in enumerate(windows):
        center = (s + e - 1) / 2
        for g in range(s, min(e - 1, n_gaps)):
            d = abs(g + 0.5 - center)
            if d < best[g]:
                best[g] = d
                owner[g] = w
    if any(o < 0 for o in owner):
        raise ValueError("windowing left a gap without a prediction")
    return owner


def windows_for(
    fixer: Fixer, tokens: Sequence[str], current: Sequence[Gap]
) -> list[tuple[int, int]]:
    token_costs = [fixer.token_cost(t) for t in tokens]
    gap_costs = [fixer.gap_cost(g) for g in current]
    return make_windows(len(tokens), token_costs, gap_costs, fixer.overhead(), fixer.budget)


def predict_all(fixer: Fixer, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
    if len(tokens) < 2:
        return []
    return _merge(fixer, tokens, current, windows_for(fixer, tokens, current))


def _merge(
    fixer: Fixer,
    tokens: Sequence[str],
    current: Sequence[Gap],
    windows: Sequence[tuple[int, int]],
) -> list[Gap]:
    n = len(tokens)
    owner = assign_gaps(windows, n - 1)
    out = list(current)
    for w, (s, e) in enumerate(windows):
        pred = fixer.predict(tokens[s:e], current[s : e - 1])
        if len(pred) != e - s - 1:
            raise ValueError(f"{fixer.name} returned {len(pred)} gaps for {e - s} tokens")
        for g in range(s, e - 1):
            if owner[g] == w:
                out[g] = pred[g - s]
    return out


@dataclass(frozen=True)
class FixResult:
    text: str
    tokens: int
    gaps: int
    changed: int
    windows: int


def fix(text: str, fixer: Fixer) -> FixResult:
    """Tokenize, predict every gap with the fixer, rebuild canonical text."""
    tokens, current = split(text)
    if len(tokens) < 2:
        return FixResult(join(tokens, []), len(tokens), 0, 0, 0)
    windows = windows_for(fixer, tokens, current)
    pred = _merge(fixer, tokens, current, windows)
    changed = sum(1 for a, b in zip(current, pred, strict=True) if a != b)
    return FixResult(join(tokens, pred), len(tokens), len(current), changed, len(windows))
