"""Training examples: corrupted windows of the clean training split (design 3.4)."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..corrupt import CorruptConfig, corrupt
from ..data.records import CleanDoc
from ..text import Gap, derive_labels, split
from .vocab import Batch, CharVocab, WordVocab, collate, encode_window

IGNORE = -100


@dataclass(frozen=True)
class Example:
    tokens: list[str]
    current: list[Gap]
    target: list[Gap]


def examples_from_doc(
    doc: CleanDoc, rng: random.Random, budget: int, cfg: CorruptConfig | None = None
) -> list[Example]:
    corrupted = corrupt(doc.clean, rng, cfg)
    tokens, current = split(corrupted.text)
    target = derive_labels(corrupted.text, doc.clean)
    out: list[Example] = []
    for start in range(0, len(tokens), budget):
        chunk = tokens[start : start + budget]
        if len(chunk) < 2:
            continue
        gaps = slice(start, start + len(chunk) - 1)
        out.append(Example(list(chunk), list(current[gaps]), list(target[gaps])))
    return out


def epoch_examples(docs: Sequence[CleanDoc], seed: int, epoch: int, budget: int) -> list[Example]:
    out: list[Example] = []
    for doc in docs:
        out.extend(examples_from_doc(doc, random.Random(f"{seed}:{epoch}:{doc.id}"), budget))
    random.Random(f"{seed}:{epoch}").shuffle(out)
    return out


def class_counts(examples: Iterable[Example]) -> list[int]:
    counts = [0, 0, 0, 0]
    for e in examples:
        for g in e.target:
            counts[int(g)] += 1
    return counts


def batches(
    examples: Sequence[Example],
    words: WordVocab,
    chars: CharVocab,
    max_chars: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[Batch, Tensor]]:
    for i in range(0, len(examples), batch_size):
        group = examples[i : i + batch_size]
        encoded = [encode_window(e.tokens, e.current, words, chars, max_chars) for e in group]
        batch = collate(encoded, device)
        n = batch.word.shape[1]
        target = torch.full((len(group), n - 1), IGNORE, dtype=torch.long)
        for j, e in enumerate(group):
            target[j, : len(e.target)] = torch.tensor([int(g) for g in e.target], dtype=torch.long)
        yield batch, target.to(device)
