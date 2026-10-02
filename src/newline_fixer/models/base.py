"""The one interface every system implements (design section 4.1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..text import Gap


class Fixer(Protocol):
    """Predicts a gap class for every gap in a window of tokens.

    Costs describe how many model input units a span occupies, so windowing can respect
    the model's budget. A fixer must cap `token_cost` so that any two adjacent tokens
    plus the gap between them fit in `budget - overhead()`; capping each token at half the
    remaining budget is sufficient when gap costs are zero. What it feeds the model for a
    capped token is truncated, the reconstruction always uses the original token.
    """

    name: str
    budget: int

    def token_cost(self, token: str) -> int: ...

    def gap_cost(self, gap: Gap) -> int: ...

    def overhead(self) -> int: ...

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]: ...
