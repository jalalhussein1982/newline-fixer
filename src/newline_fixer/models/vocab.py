"""Word and character vocabularies and tensor encoding of token windows."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..text import Gap

PAD = 0
UNK = 1
NO_GAP = 4
_SPECIALS = ["<pad>", "<unk>"]


class _Vocab:
    def __init__(self, items: Sequence[str]) -> None:
        self.itos: list[str] = [*_SPECIALS, *items]
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.itos[len(_SPECIALS) :], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def _load_items(cls, path: Path) -> list[str]:
        items: list[str] = json.loads(path.read_text(encoding="utf-8"))
        return items


class WordVocab(_Vocab):
    @classmethod
    def build(cls, tokens: Iterable[str], size: int) -> WordVocab:
        counts = Counter(t.lower() for t in tokens)
        return cls([w for w, _ in counts.most_common(size)])

    @classmethod
    def load(cls, path: Path) -> WordVocab:
        return cls(cls._load_items(path))

    def encode(self, token: str) -> int:
        return self.stoi.get(token.lower(), UNK)


class CharVocab(_Vocab):
    @classmethod
    def build(cls, tokens: Iterable[str], min_count: int = 5) -> CharVocab:
        counts: Counter[str] = Counter()
        for t in tokens:
            counts.update(t)
        return cls(sorted(c for c, n in counts.items() if n >= min_count))

    @classmethod
    def load(cls, path: Path) -> CharVocab:
        return cls(cls._load_items(path))

    def encode(self, token: str, max_chars: int) -> list[int]:
        ids = [self.stoi.get(c, UNK) for c in token[:max_chars]]
        return ids + [PAD] * (max_chars - len(ids))


@dataclass(frozen=True)
class Encoded:
    word: list[int]
    char: list[list[int]]
    gap: list[int]
    n: int


def encode_window(
    tokens: Sequence[str],
    current: Sequence[Gap],
    words: WordVocab,
    chars: CharVocab,
    max_chars: int,
) -> Encoded:
    return Encoded(
        word=[words.encode(t) for t in tokens],
        char=[chars.encode(t, max_chars) for t in tokens],
        gap=[int(g) for g in current],
        n=len(tokens),
    )


@dataclass
class Batch:
    word: Tensor
    char: Tensor
    gap_after: Tensor
    lengths: Tensor


def collate(encoded: Sequence[Encoded], device: torch.device) -> Batch:
    b = len(encoded)
    n = max(e.n for e in encoded)
    c = len(encoded[0].char[0])
    word = torch.full((b, n), PAD, dtype=torch.long)
    char = torch.full((b, n, c), PAD, dtype=torch.long)
    gap_after = torch.full((b, n), NO_GAP, dtype=torch.long)
    lengths = torch.tensor([e.n for e in encoded], dtype=torch.long)
    for i, e in enumerate(encoded):
        word[i, : e.n] = torch.tensor(e.word, dtype=torch.long)
        char[i, : e.n] = torch.tensor(e.char, dtype=torch.long)
        if e.gap:
            gap_after[i, : e.n - 1] = torch.tensor(e.gap, dtype=torch.long)
    return Batch(word.to(device), char.to(device), gap_after.to(device), lengths)
