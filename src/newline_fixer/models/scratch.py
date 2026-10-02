"""M1 of the design: the from-scratch model behind the Fixer protocol."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

from ..text import Gap
from .device import select_device
from .scratch_config import ScratchConfig
from .scratch_net import GapTagger
from .vocab import CharVocab, WordVocab, collate, encode_window

WEIGHTS_ENV = "NF_WEIGHTS"
DEFAULT_WEIGHTS = "experiments/runs/current"


def default_weights() -> str:
    return os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)


class ScratchFixer:
    name = "scratch"

    def __init__(
        self,
        cfg: ScratchConfig,
        words: WordVocab,
        chars: CharVocab,
        net: GapTagger,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.words = words
        self.chars = chars
        self.net = net.to(device).eval()
        self.device = device
        self.budget = cfg.budget

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        if len(tokens) < 2:
            return []
        enc = encode_window(tokens, current, self.words, self.chars, self.cfg.max_chars)
        batch = collate([enc], self.device)
        with torch.no_grad():
            logits = self.net(batch)
        ids = logits[0, : len(current)].argmax(dim=-1).tolist()
        return [Gap(int(i)) for i in ids]

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.save(run_dir / "config.json")
        self.words.save(run_dir / "words.json")
        self.chars.save(run_dir / "chars.json")
        torch.save(self.net.state_dict(), run_dir / "model.pt")

    @classmethod
    def untrained(
        cls,
        cfg: ScratchConfig,
        words: WordVocab,
        chars: CharVocab,
        seed: int,
        device: torch.device | None = None,
    ) -> ScratchFixer:
        torch.manual_seed(seed)
        net = GapTagger(cfg, len(words), len(chars))
        return cls(cfg, words, chars, net, device or select_device())

    @classmethod
    def load(cls, source: str | Path, device: torch.device | None = None) -> ScratchFixer:
        run_dir = resolve_weights(source)
        if not (run_dir / "model.pt").exists():
            raise FileNotFoundError(
                f"no weights at {run_dir}; train with scripts/train_scratch.py or set {WEIGHTS_ENV}"
            )
        device = device or select_device()
        cfg = ScratchConfig.load(run_dir / "config.json")
        words = WordVocab.load(run_dir / "words.json")
        chars = CharVocab.load(run_dir / "chars.json")
        net = GapTagger(cfg, len(words), len(chars))
        net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        return cls(cfg, words, chars, net, device)


def resolve_weights(source: str | Path) -> Path:
    """A local directory. Task 8 extends this to Hub sources of the form hf:repo@revision."""
    return Path(source)
