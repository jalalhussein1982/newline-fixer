"""Realistic passages from real PDF extractions (design section 5.1)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from ..text import content, normalize, split, unreachable_count
from .records import EvalItem

_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n[ \t]*(?=[a-z])")


def join_hyphenation(raw: str) -> tuple[str, int]:
    """`que-\\nries` -> `que\\nries`. Keeps the break so the service must still join it."""
    out, n = _HYPHEN_BREAK.subn("\n", raw)
    return out, n


def cut_raw_passages(
    raw: str,
    rng: random.Random,
    per_doc: int = 8,
    min_chars: int = 300,
    max_chars: int = 800,
    min_letter_ratio: float = 0.6,
) -> list[str]:
    """Cut on page breaks and blank lines, then take passages within bounds at random.

    Passages that are mostly non-letters (indexes, tables, formulas) are dropped before sampling.
    """
    chunks: list[str] = []
    for page in raw.split("\f"):
        buf: list[str] = []
        for block in re.split(r"\n[ \t]*\n", page):
            block = block.strip("\n")
            if not block.strip():
                continue
            candidate = "\n\n".join([*buf, block])
            if len(candidate) > max_chars:
                if len("\n\n".join(buf)) >= min_chars:
                    chunks.append("\n\n".join(buf))
                buf = [block] if len(block) <= max_chars else []
            else:
                buf.append(block)
        if len("\n\n".join(buf)) >= min_chars:
            chunks.append("\n\n".join(buf))
    chunks = [c.strip("\n") for c in chunks if min_chars <= len(c.strip("\n")) <= max_chars]
    chunks = [c for c in chunks if sum(ch.isalpha() for ch in c) / len(c) >= min_letter_ratio]
    return rng.sample(chunks, min(per_doc, len(chunks)))


PROPOSE_PROMPT = (
    "The text below was extracted from a PDF and its line breaks are wrong. Rewrite it with "
    "correct newlines: one blank line between paragraphs and after headings, a single newline "
    "before each list item, no line breaks inside sentences, and split words joined. Change "
    "ONLY whitespace. Every non-whitespace character must stay exactly as it is, in the same "
    "order. Output only the corrected text.\n\n"
)


def propose_target(
    input_text: str, model: str, client: Any | None = None, attempts: int = 3
) -> str:
    api: Any = client
    if api is None:
        import anthropic

        api = anthropic.Anthropic()
    for _ in range(attempts):
        msg = api.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": PROPOSE_PROMPT + input_text}],
        )
        text = normalize(msg.content[0].text)
        if content(text) == content(input_text):
            return text
    raise RuntimeError("proposal changed non-whitespace content three times")


def validate_pair(input_text: str, target: str) -> dict[str, int]:
    _, gaps = split(input_text)
    return {"unreachable": unreachable_count(input_text, target), "gaps": len(gaps)}


def load_reviewed(root: Path, role: str) -> list[EvalItem]:
    sources = json.loads((root / "sources.json").read_text())
    review = (
        json.loads((root / "review.json").read_text()) if (root / "review.json").exists() else {}
    )
    items: list[EvalItem] = []
    for src in sources[role]:
        doc_dir = root / src["doc"]
        if not doc_dir.exists():
            continue
        for inp in sorted(doc_dir.glob("*.input.txt")):
            nn = inp.name.split(".")[0]
            key = f"{src['doc']}/{nn}"
            if key not in review:
                continue
            raw = (doc_dir / f"{nn}.raw.txt").read_text()
            input_text = normalize(inp.read_text())
            target = normalize((doc_dir / f"{nn}.target.txt").read_text())
            if content(input_text) != content(target):
                raise ValueError(f"{key}: target changes non-whitespace content")
            stats = validate_pair(input_text, target)
            _, adjustments = join_hyphenation(raw)
            items.append(
                EvalItem(
                    key,
                    "realistic",
                    input_text,
                    target,
                    1.0,
                    {**review[key], "doc": src["doc"], "adjustments": adjustments, **stats},
                )
            )
    return items
