"""Name -> fixer. Learned models register here in later milestones."""

from __future__ import annotations

from collections.abc import Callable

from .base import Fixer
from .identity import IdentityFixer


def _rules() -> Fixer:
    from ..rules import RulesFixer

    return RulesFixer()


def _scratch() -> Fixer:
    from .scratch import ScratchFixer, default_weights

    return ScratchFixer.load(default_weights())


_REGISTRY: dict[str, Callable[[], Fixer]] = {
    "identity": IdentityFixer,
    "rules": _rules,
    "scratch": _scratch,
}
FIXER_NAMES = sorted(_REGISTRY)


def get_fixer(name: str) -> Fixer:
    try:
        return _REGISTRY[name]()
    except KeyError as e:
        raise ValueError(f"unknown fixer {name!r}; known: {FIXER_NAMES}") from e
