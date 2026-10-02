"""Service settings from NF_* environment variables (design 6.2)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from ..models.base import Fixer
from ..models.registry import FIXER_NAMES, get_fixer

HUB_REPO = "jalalhussein1982/newline-fixer-scratch"
PUBLISHED_REVISION = "6c311e757d17e89c80b7b86908043637a4f56e28"  # scratch-v1, decision 0007
HUB_REPO_FINETUNED = "jalalhussein1982/newline-fixer-finetuned"
PUBLISHED_REVISION_FINETUNED = (
    "11d6b26e80dfa2c9606702cd2755a63c9dce99ed"  # finetuned, decision 0010
)
SERVABLE = ("identity", "rules", "scratch", "finetuned")
DEFAULT_MODEL = "finetuned"  # decision 0010 (supersedes 0008); NF_MODEL=rules|scratch: baselines


@dataclass(frozen=True)
class Settings:
    """Service settings.

    `weights_by_model` holds (model, source) pairs from NF_WEIGHTS_SCRATCH and
    NF_WEIGHTS_FINETUNED (a tuple of pairs because the dataclass is frozen). `model_revision`
    defaults to the scratch revision; for `finetuned` that default means "not set" and the
    finetuned constant applies instead.
    """

    model: str = DEFAULT_MODEL
    model_revision: str = PUBLISHED_REVISION
    weights: str | None = None
    max_chars: int = 100_000
    log_level: str = "INFO"
    device: str = "cpu"
    weights_by_model: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.model not in FIXER_NAMES:
            raise ValueError(f"unknown model {self.model!r}; NF_MODEL must be one of {FIXER_NAMES}")
        if self.model not in SERVABLE:
            raise ValueError(f"model {self.model!r} is not servable; use one of {list(SERVABLE)}")
        if (
            self.model == "finetuned"
            and not self.weights
            and "finetuned" not in dict(self.weights_by_model)
            and not self._finetuned_revision()
        ):
            raise ValueError(
                "model 'finetuned' has no weights source: set NF_WEIGHTS, NF_WEIGHTS_FINETUNED "
                "or NF_MODEL_REVISION"
            )
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
            weights_by_model=tuple(
                (m, e[var])
                for m, var in (
                    ("scratch", "NF_WEIGHTS_SCRATCH"),
                    ("finetuned", "NF_WEIGHTS_FINETUNED"),
                )
                if e.get(var)
            ),
        )

    def _finetuned_revision(self) -> str:
        if self.model_revision == PUBLISHED_REVISION:  # the scratch default: not set
            return PUBLISHED_REVISION_FINETUNED
        return self.model_revision

    def weights_source(self) -> str | None:
        """Where the weights come from; None for systems without weights.

        NF_WEIGHTS, else NF_WEIGHTS_<MODEL>, else the Hub repo at the revision. An empty
        revision gives 'hf:<repo>' (the repo's main).
        """
        if self.model not in ("scratch", "finetuned"):
            return None
        if self.weights:
            return self.weights
        per_model = dict(self.weights_by_model).get(self.model)
        if per_model:
            return per_model
        if self.model == "scratch":
            repo, revision = HUB_REPO, self.model_revision
        else:
            repo = HUB_REPO_FINETUNED
            revision = self._finetuned_revision()
        return f"hf:{repo}@{revision}" if revision else f"hf:{repo}"


def load_fixer(settings: Settings) -> Fixer:
    """Build the served fixer. Learned models load on `settings.device` (CPU in production)."""
    source = settings.weights_source()
    if source is None:
        return get_fixer(settings.model)
    import torch

    device = torch.device(settings.device)
    if settings.model == "finetuned":
        from ..models.finetuned import FinetunedFixer

        return FinetunedFixer.load(source, device)
    from ..models.scratch import ScratchFixer

    return ScratchFixer.load(source, device)
