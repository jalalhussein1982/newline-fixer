"""Assemble, split, and derive evaluation sets (design sections 3.2, 3.3, 5.1)."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence

from ..corrupt import corrupt
from ..lexicon import Lexicon
from ..text import split
from .filters import has_structure, is_hard_wrapped, long_enough
from .passages import cut_passages
from .records import CleanDoc, EvalItem
from .splits import assign_splits, dedupe


def assemble(docs: Iterable[CleanDoc], seed: int) -> dict[str, list[CleanDoc]]:
    kept = [
        d
        for d in docs
        if long_enough(d.clean) and has_structure(d.clean) and not is_hard_wrapped(d.clean)
    ]
    kept = dedupe(kept)
    side = assign_splits((d.group for d in kept), seed=seed)
    out: dict[str, list[CleanDoc]] = {"train": [], "val": [], "test": []}
    for d in kept:
        out[side[d.group]].append(d)
    return out


def make_corrupted_set(docs: Sequence[CleanDoc], seed: int, limit: int | None) -> list[EvalItem]:
    rng = random.Random(seed)
    chosen = list(docs) if limit is None else rng.sample(list(docs), min(limit, len(docs)))
    items: list[EvalItem] = []
    for d in chosen:
        c = corrupt(d.clean, random.Random(f"{seed}:{d.id}"))
        items.append(EvalItem(d.id, d.source, c.text, d.clean, c.severity, {"group": d.group}))
    return items


def make_clean_set(docs: Sequence[CleanDoc], seed: int, n_passages: int) -> list[EvalItem]:
    rng = random.Random(seed)
    pool: list[tuple[CleanDoc, int, str]] = []
    for d in docs:
        for k, p in enumerate(cut_passages(d.clean)):
            pool.append((d, k, p))
    chosen = rng.sample(pool, min(n_passages, len(pool)))
    return [EvalItem(f"{d.id}#{k}", d.source, p, p, 0.0, {"group": d.group}) for d, k, p in chosen]


def build_lexicon(docs: Iterable[CleanDoc], min_count: int = 3) -> Lexicon:
    counts: Counter[str] = Counter()
    for d in docs:
        tokens, _ = split(d.clean)
        counts.update(t.lower() for t in tokens if t.isalpha())
    return Lexicon.from_counts(counts, min_count=min_count)
