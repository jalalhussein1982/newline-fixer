"""Document filters (design section 3.1)."""

from __future__ import annotations

TERMINAL = ".!?:;\"')»”"


def long_enough(text: str, min_chars: int = 200) -> bool:
    return len(text) >= min_chars


def has_structure(text: str) -> bool:
    """At least one paragraph break or line break survives normalization."""
    return "\n" in text


def is_hard_wrapped(text: str, threshold: float = 0.3) -> bool:
    """True when many lines end mid-sentence and the next line starts lowercase."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return False
    suspicious = 0
    for a, b in zip(lines, lines[1:], strict=False):
        if a[-1] not in TERMINAL and b[0].islower():
            suspicious += 1
    return suspicious / (len(lines) - 1) > threshold
