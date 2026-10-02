"""Service settings from NF_* environment variables (design 6.2)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from ..models.base import Fixer
from ..models.registry import FIXER_NAMES, get_fixer

HUB_REPO = "jalalhussein1982/newline-fixer-scratch"
PUBLISHED_REVISION = "6c311e757d17e89c80b7b86908043637a4f56e28"  # scratch-v1, decision 0007
DEFAULT_MODEL = "rules"  # the decision rule of design 5.3; confirmed or changed by decision 0008


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    model_revision: str = PUBLISHED_REVISION
    weights: str | None = None
    max_chars: int = 100_000
    log_level: str = "INFO"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.model not in FIXER_NAMES:
            raise ValueError(f"unknown model {self.model!r}; NF_MODEL must be one of {FIXER_NAMES}")
        if self.max_chars <= 0:
            raise ValueError(f"NF_MAX_CHARS must be a positive integer, got {self.max_chars}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        raw_max = e.get("NF_MAX_CHARS", "100000")
        try:
            max_chars = int(raw_max)
        except ValueError as err:
            raise ValueError(f"NF_MAX_CHARS must be a positive integer, got {raw_max!r}") from err
        return cls(
            model=e.get("NF_MODEL", DEFAULT_MODEL),
            model_revision=e.get("NF_MODEL_REVISION", PUBLISHED_REVISION),
            weights=e.get("NF_WEIGHTS") or None,
            max_chars=max_chars,
            log_level=e.get("NF_LOG_LEVEL", "INFO").upper(),
            device=e.get("NF_DEVICE", "cpu"),
        )

    def weights_source(self) -> str | None:
        """Where the scratch weights come from; None for systems without weights."""
        if self.model != "scratch":
            return None
        return self.weights or f"hf:{HUB_REPO}@{self.model_revision}"


def load_fixer(settings: Settings) -> Fixer:
    """Build the served fixer. Learned models load on `settings.device` (CPU in production)."""
    source = settings.weights_source()
    if source is None:
        return get_fixer(settings.model)
    import torch

    from ..models.scratch import ScratchFixer

    return ScratchFixer.load(source, torch.device(settings.device))
