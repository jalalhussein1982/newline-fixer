"""Pick the torch device: an explicit choice, else MPS when available, else CPU."""

from __future__ import annotations

import torch


def select_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
