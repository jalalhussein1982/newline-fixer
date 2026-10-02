"""Markers, subword costs and window encoding for the fine-tuned encoder (design 4.5)."""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..text import Gap

MARKERS: dict[Gap, str] = {Gap.NL: "[NL]", Gap.PARA: "[PP]"}
BUDGET = 512
OVERHEAD = 2  # CLS and SEP
TOKEN_CAP = (BUDGET - OVERHEAD) // 2 - 1
IGNORE = -100


def add_markers(tokenizer: Any) -> int:
    """Register the marker words as special tokens; returns the vocabulary size."""
    missing = [
        m for m in MARKERS.values() if tokenizer.convert_tokens_to_ids(m) == tokenizer.unk_token_id
    ]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return int(len(tokenizer))


def _pieces(tokenizer: Any, token: str) -> list[int]:
    ids: list[int] = tokenizer.encode(token, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.unk_token_id]
    return ids[:TOKEN_CAP]


# Memo: one inner dict per tokenizer (weak-keyed, so entries die with it); an inner dict
# is cleared when it passes 200k entries.
_COST_CACHE: weakref.WeakKeyDictionary[Any, dict[str, int]] = weakref.WeakKeyDictionary()
_COST_CACHE_MAX = 200_000


def token_cost(tokenizer: Any, token: str) -> int:
    """Subwords of the token alone: at least 1, at most TOKEN_CAP."""
    costs = _COST_CACHE.setdefault(tokenizer, {})
    if token not in costs:
        if len(costs) > _COST_CACHE_MAX:
            costs.clear()
        costs[token] = len(_pieces(tokenizer, token))
    return costs[token]


def gap_cost(gap: Gap) -> int:
    return 1 if gap in MARKERS else 0


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    attention_mask: list[int]
    label_positions: list[int]
    n_tokens: int
    overflowed: bool


def encode_window(
    tokens: Sequence[str], current: Sequence[Gap], tokenizer: Any, max_len: int = BUDGET
) -> Encoded:
    ids: list[int] = [tokenizer.cls_token_id]
    positions: list[int] = []
    overflowed = False
    marker_ids = {g: tokenizer.convert_tokens_to_ids(m) for g, m in MARKERS.items()}
    for i, token in enumerate(tokens):
        pieces = _pieces(tokenizer, token)
        if len(ids) + len(pieces) > max_len - 1:
            overflowed = True
            break
        positions.append(len(ids))
        ids.extend(pieces)
        if i < len(current) and current[i] in MARKERS:
            if len(ids) + 1 > max_len - 1:
                overflowed = True
                break
            ids.append(marker_ids[current[i]])
    ids.append(tokenizer.sep_token_id)
    return Encoded(ids, [1] * len(ids), positions, len(tokens), overflowed)


def labels_for(encoded: Encoded, target: Sequence[Gap]) -> list[int]:
    labels = [IGNORE] * len(encoded.input_ids)
    for i, pos in enumerate(encoded.label_positions):
        if i < encoded.n_tokens - 1 and i < len(target):
            labels[pos] = int(target[i])
    return labels


def collate(
    batch: Sequence[tuple[Encoded, list[int] | None]], pad_id: int, device: torch.device
) -> dict[str, Tensor]:
    width = max(len(e.input_ids) for e, _ in batch)
    n_pos = max(len(e.label_positions) for e, _ in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    pos = torch.full((len(batch), max(n_pos, 1)), -1, dtype=torch.long)
    has_labels = all(lab is not None for _, lab in batch)
    for b, (e, lab) in enumerate(batch):
        n = len(e.input_ids)
        ids[b, :n] = torch.tensor(e.input_ids)
        mask[b, :n] = 1
        if lab is not None:
            labels[b, :n] = torch.tensor(lab)
        if e.label_positions:
            pos[b, : len(e.label_positions)] = torch.tensor(e.label_positions)
    out = {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "label_positions": pos.to(device),
        "n_tokens": torch.tensor([e.n_tokens for e, _ in batch]).to(device),
    }
    if has_labels:
        out["labels"] = labels.to(device)
    return out
