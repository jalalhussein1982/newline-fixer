"""The fine-tuned encoder behind the Fixer protocol (design 4.5)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..text import Gap
from .device import select_device
from .finetune_encoding import (
    BUDGET,
    OVERHEAD,
    add_markers,
    collate,
    encode_window,
    gap_cost,
    token_cost,
)
from .scratch import resolve_weights

WEIGHTS_ENV_FINETUNED = "NF_WEIGHTS_FINETUNED"
WEIGHTS_ENV_ABLATION = "NF_WEIGHTS_FINETUNED_ABLATION"
DEFAULT_WEIGHTS_FINETUNED = "experiments/runs/finetuned"
DEFAULT_WEIGHTS_ABLATION = "experiments/runs/finetuned-ablation"
NUM_LABELS = 4


def default_weights_finetuned() -> str:
    return (
        os.environ.get("NF_WEIGHTS")
        or os.environ.get(WEIGHTS_ENV_FINETUNED)
        or DEFAULT_WEIGHTS_FINETUNED
    )


def default_weights_ablation() -> str:
    return os.environ.get(WEIGHTS_ENV_ABLATION) or DEFAULT_WEIGHTS_ABLATION


@dataclass(frozen=True)
class FinetunedConfig:
    pretrained: str
    random_init: bool = False
    budget: int = BUDGET
    max_len: int = BUDGET

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FinetunedConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _tokenizer_overrides(run_dir: Path) -> dict[str, object]:
    """Load kwargs for tokenizers saved by transformers 5 (Colab), which writes
    `extra_special_tokens` as a list; transformers 4 expects a mapping. The tokens themselves
    are in tokenizer.json, so naming them is enough."""
    path = run_dir / "tokenizer_config.json"
    if not path.exists():
        return {}
    extra = json.loads(path.read_text(encoding="utf-8")).get("extra_special_tokens")
    if isinstance(extra, list):
        return {"extra_special_tokens": {f"extra_{i}": t for i, t in enumerate(extra)}}
    return {}


class FinetunedFixer:
    name = "finetuned"

    def __init__(
        self,
        cfg: FinetunedConfig,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        name: str = "finetuned",
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.float().to(device).eval()
        self.budget = cfg.budget
        self.name = name
        self.weights_dir: Path | None = None

    def token_cost(self, token: str) -> int:
        return token_cost(self.tokenizer, token)

    def gap_cost(self, gap: Gap) -> int:
        return gap_cost(gap)

    def overhead(self) -> int:
        return OVERHEAD

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        if len(tokens) < 2:
            return []
        enc = encode_window(tokens, current, self.tokenizer, self.cfg.max_len)
        batch = collate([(enc, None)], self.tokenizer.pad_token_id, self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits[0]
        out = list(current)  # fallback for positions lost to overflow
        for i in range(len(current)):
            if i < len(enc.label_positions):
                out[i] = Gap(int(logits[enc.label_positions[i]].argmax().item()))
        return out

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(run_dir)
        self.tokenizer.save_pretrained(run_dir)
        self.cfg.save(run_dir / "fixer.json")

    @classmethod
    def from_parts(
        cls, cfg: FinetunedConfig, tokenizer: Any, model: Any, device: torch.device
    ) -> FinetunedFixer:
        return cls(cfg, tokenizer, model, device)

    @classmethod
    def from_pretrained_name(cls, cfg: FinetunedConfig, device: torch.device) -> FinetunedFixer:
        """Download the tokenizer and model named by cfg.pretrained (network)."""
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.pretrained)  # type: ignore[no-untyped-call]
        add_markers(tok)
        if cfg.random_init:
            config = AutoConfig.from_pretrained(cfg.pretrained, num_labels=NUM_LABELS)
            model = AutoModelForTokenClassification.from_config(config)  # type: ignore[no-untyped-call]
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                cfg.pretrained, num_labels=NUM_LABELS, dtype=torch.float32
            )
        model.resize_token_embeddings(len(tok))
        return cls(cfg, tok, model, device)

    @classmethod
    def load(
        cls,
        source: str | Path,
        device: torch.device | None = None,
        name: str = "finetuned",
    ) -> FinetunedFixer:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        run_dir = resolve_weights(source)
        if not (run_dir / "fixer.json").exists():
            raise FileNotFoundError(
                f"no weights at {run_dir}; train a fine-tuned run or set {WEIGHTS_ENV_FINETUNED}"
            )
        cfg = FinetunedConfig.load(run_dir / "fixer.json")
        tok = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            run_dir, **_tokenizer_overrides(run_dir)
        )
        model = AutoModelForTokenClassification.from_pretrained(run_dir)
        fixer = cls(cfg, tok, model, device or select_device(), name=name)
        fixer.weights_dir = run_dir
        return fixer
