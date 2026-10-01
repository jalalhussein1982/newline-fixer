"""B0: keep the current gap classes; output is the normalized input."""

from __future__ import annotations

from collections.abc import Sequence

from ..text import Gap


class IdentityFixer:
    name = "identity"
    budget = 256

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        return list(current)
