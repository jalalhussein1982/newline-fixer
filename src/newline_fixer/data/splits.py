"""Deduplication and grouped splitting (design section 3.2)."""

from __future__ import annotations

import random
from collections.abc import Iterable

from .records import CleanDoc

PREFIX_CHARS = 200


def dedupe(docs: Iterable[CleanDoc]) -> list[CleanDoc]:
    seen_hash: set[str] = set()
    seen_prefix: set[str] = set()
    out: list[CleanDoc] = []
    for d in docs:
        prefix = d.clean[:PREFIX_CHARS]
        if d.sha256 in seen_hash or prefix in seen_prefix:
            continue
        seen_hash.add(d.sha256)
        seen_prefix.add(prefix)
        out.append(d)
    return out


def assign_splits(
    groups: Iterable[str], seed: int, val: float = 0.05, test: float = 0.05
) -> dict[str, str]:
    ids = sorted(set(groups))
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * val)
    n_test = round(len(ids) * test)
    out: dict[str, str] = {}
    for i, g in enumerate(ids):
        if i < n_val:
            out[g] = "val"
        elif i < n_val + n_test:
            out[g] = "test"
        else:
            out[g] = "train"
    return out
