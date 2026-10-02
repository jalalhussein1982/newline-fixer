"""Cut documents into passages on paragraph boundaries (design section 5.1)."""

from __future__ import annotations


def cut_passages(text: str, min_chars: int = 300, max_chars: int = 800) -> list[str]:
    """Greedy, deterministic: accumulate paragraphs until adding one would exceed max."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        if len(p) > max_chars:
            if min_chars <= size:
                out.append("\n\n".join(buf))
            buf, size = [], 0
            continue
        extra = len(p) + (2 if buf else 0)
        if size + extra > max_chars:
            if size >= min_chars:
                out.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += extra
    if size >= min_chars:
        out.append("\n\n".join(buf))
    return out
