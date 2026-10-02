"""Dataset record types and JSONL IO (design section 3.2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..text import normalize


@dataclass(frozen=True)
class CleanDoc:
    id: str
    source: str
    source_ref: str
    group: str
    sha256: str
    clean: str

    @classmethod
    def make(cls, id: str, source: str, source_ref: str, group: str, text: str) -> CleanDoc:
        clean = normalize(text)
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        return cls(
            id=id, source=source, source_ref=source_ref, group=group, sha256=digest, clean=clean
        )


@dataclass(frozen=True)
class EvalItem:
    id: str
    source: str
    input: str
    target: str
    severity: float = 0.0
    meta: dict[str, object] = field(default_factory=dict)


def write_jsonl[T: (CleanDoc, EvalItem)](path: Path, rows: Iterable[T]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl[T: (CleanDoc, EvalItem)](path: Path, cls: type[T]) -> list[T]:
    names = {f.name for f in fields(cls)}
    out: list[T] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                out.append(cls(**{k: v for k, v in raw.items() if k in names}))
    return out
