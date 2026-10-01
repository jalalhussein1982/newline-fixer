"""Gap-level and paragraph metrics (design section 5.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..text import Gap

BREAKS = (Gap.NL, Gap.PARA)


@dataclass
class ClassCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    support: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class GapMetrics:
    per_class: dict[Gap, ClassCounts] = field(
        default_factory=lambda: {g: ClassCounts() for g in Gap}
    )
    n_gaps: int = 0
    n_changed: int = 0
    n_wrong_join: int = 0
    n_items: int = 0
    n_items_untouched: int = 0
    break_tp: int = 0
    break_fp: int = 0
    break_fn: int = 0

    def update(self, pred: Sequence[Gap], ref: Sequence[Gap], current: Sequence[Gap]) -> None:
        if not (len(pred) == len(ref) == len(current)):
            raise ValueError("pred, ref and current must have the same length")
        changed = 0
        for p, r, c in zip(pred, ref, current, strict=True):
            self.n_gaps += 1
            self.per_class[r].support += 1
            if p == r:
                self.per_class[p].tp += 1
            else:
                self.per_class[p].fp += 1
                self.per_class[r].fn += 1
                if p is Gap.JOIN:
                    self.n_wrong_join += 1
            if p != c:
                changed += 1
            pb, rb = p in BREAKS, r in BREAKS
            if pb and rb:
                self.break_tp += 1
            elif pb:
                self.break_fp += 1
            elif rb:
                self.break_fn += 1
        self.n_changed += changed
        self.n_items += 1
        if changed == 0:
            self.n_items_untouched += 1

    def macro_classes(self) -> list[Gap]:
        return [g for g in Gap if self.per_class[g].support > 0]

    def macro_f1(self) -> float:
        classes = self.macro_classes()
        return sum(self.per_class[g].f1 for g in classes) / len(classes) if classes else 0.0

    def break_f1(self) -> float:
        tp, fp, fn = self.break_tp, self.break_fp, self.break_fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    def wrong_join_per_1000(self) -> float:
        return 1000 * self.n_wrong_join / self.n_gaps if self.n_gaps else 0.0

    def damage_rate(self) -> float:
        return self.n_changed / self.n_gaps if self.n_gaps else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "n_items": self.n_items,
            "n_gaps": self.n_gaps,
            "macro_f1": self.macro_f1(),
            "macro_classes": [g.name for g in self.macro_classes()],
            "break_f1": self.break_f1(),
            "wrong_join_per_1000": self.wrong_join_per_1000(),
            "damage_rate": self.damage_rate(),
            "items_untouched_rate": self.n_items_untouched / self.n_items if self.n_items else 0.0,
            "per_class": {
                g.name: {
                    "support": c.support,
                    "precision": c.precision,
                    "recall": c.recall,
                    "f1": c.f1,
                }
                for g, c in self.per_class.items()
            },
        }


def paragraph_match(output: str, reference: str) -> tuple[int, int]:
    """(matched, total): reference paragraphs that occur verbatim among output paragraphs."""
    ref_paras = [p for p in reference.split("\n\n") if p]
    out_paras = {p for p in output.split("\n\n") if p}
    return sum(1 for p in ref_paras if p in out_paras), len(ref_paras)
