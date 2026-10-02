"""Hyperparameters of the from-scratch model (design section 4.4)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScratchConfig:
    vocab_size: int = 30_000
    word_dim: int = 128
    char_dim: int = 32
    char_filters: int = 64
    char_width: int = 3
    max_chars: int = 40
    gap_dim: int = 8
    hidden: int = 192
    layers: int = 2
    dropout: float = 0.2
    classifier_hidden: int = 256
    budget: int = 256

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ScratchConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


TINY = ScratchConfig(
    vocab_size=50,
    word_dim=8,
    char_dim=4,
    char_filters=6,
    char_width=3,
    max_chars=8,
    gap_dim=2,
    hidden=6,
    layers=1,
    dropout=0.0,
    classifier_hidden=8,
    budget=16,
)
