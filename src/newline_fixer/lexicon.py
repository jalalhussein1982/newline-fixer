"""Word list used by the rules baseline (design section 4.3)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path


class Lexicon:
    def __init__(self, words: Iterable[str]) -> None:
        self._words = {w.strip().lower() for w in words if w.strip()}

    def known(self, word: str) -> bool:
        return word.lower() in self._words

    def __len__(self) -> int:
        return len(self._words)

    @classmethod
    def from_counts(cls, counts: Mapping[str, int], min_count: int = 3) -> Lexicon:
        return cls(w for w, c in counts.items() if c >= min_count and w.isalpha())

    @classmethod
    def from_file(cls, path: Path) -> Lexicon:
        return cls(path.read_text(encoding="utf-8").splitlines())

    @classmethod
    def bundled(cls) -> Lexicon:
        text = resources.files("newline_fixer.resources").joinpath("lexicon.txt").read_text("utf-8")
        return cls(text.splitlines())

    def write(self, path: Path) -> None:
        path.write_text("\n".join(sorted(self._words)) + "\n", encoding="utf-8")
