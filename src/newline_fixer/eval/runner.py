"""Run systems over evaluation sets (design section 5)."""

from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ..data.records import EvalItem, read_jsonl
from ..models.base import Fixer
from ..models.registry import get_fixer
from ..text import derive_labels, join, normalize, split
from ..windows import predict_all
from .metrics import GapMetrics, paragraph_match

BANDS = (
    ("0", 0.0, 0.0),
    ("(0,0.33]", 0.0, 0.33),
    ("(0.33,0.66]", 0.33, 0.66),
    ("(0.66,1]", 0.66, 1.0),
)


def _band(severity: float) -> str:
    if severity == 0.0:
        return "0"
    for name, lo, hi in BANDS[1:]:
        if lo < severity <= hi:
            return name
    return BANDS[-1][0]


def evaluate_set(fixer: Fixer, items: Sequence[EvalItem]) -> dict[str, object]:
    overall = GapMetrics()
    by_band = {name: GapMetrics() for name, _, _ in BANDS}
    matched = total = 0
    changed_raw = changed_norm = 0
    for it in items:
        tokens, current = split(it.input)
        ref = derive_labels(it.input, it.target)
        pred = predict_all(fixer, tokens, current)
        text = join(tokens, pred)
        if len(pred) != len(ref):
            raise RuntimeError(
                f"{fixer.name} returned {len(pred)} gaps for {len(ref)} on item {it.id}"
            )
        overall.update(pred, ref, current)
        by_band[_band(it.severity)].update(pred, ref, current)
        m, t = paragraph_match(text, normalize(it.target))
        matched += m
        total += t
        changed_raw += text != it.input
        changed_norm += text != normalize(it.input)
    n = len(items)
    return {
        "gap": overall.to_dict(),
        "paragraph_match_rate": matched / total if total else 0.0,
        "string_changed_vs_raw": changed_raw / n if n else 0.0,
        "string_changed_vs_normalized": changed_norm / n if n else 0.0,
        "by_severity": {k: v.to_dict() for k, v in by_band.items()},
    }


def run(fixers: Sequence[str], sets: Sequence[str], sets_dir: Path) -> dict[str, object]:
    systems: dict[str, dict[str, object]] = {}
    for name in fixers:
        fixer = get_fixer(name)
        systems[name] = {
            s: evaluate_set(fixer, read_jsonl(sets_dir / f"{s}.jsonl", EvalItem)) for s in sets
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "sets": list(sets),
        "systems": systems,
    }
