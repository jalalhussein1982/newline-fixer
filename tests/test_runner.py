from collections.abc import Sequence
from pathlib import Path

from newline_fixer.data.records import EvalItem, write_jsonl
from newline_fixer.eval.runner import evaluate_set, run
from newline_fixer.eval.table import render_table
from newline_fixer.models.registry import FIXER_NAMES, get_fixer
from newline_fixer.text import Gap

ITEMS = [
    EvalItem("1", "t", "a b\nc", "a b c", 0.2, {}),
    EvalItem("2", "t", "x\n\ny", "x\n\ny", 0.0, {}),
    EvalItem("3", "t", "Heading Body text here.", "Heading\n\nBody text here.", 0.9, {}),
]


def test_registry() -> None:
    assert set(FIXER_NAMES) >= {"identity", "rules"}
    assert get_fixer("identity").name == "identity"


def test_evaluate_set_identity() -> None:
    res = evaluate_set(get_fixer("identity"), ITEMS)
    gap = res["gap"]
    assert isinstance(gap, dict) and gap["n_items"] == 3 and gap["n_gaps"] == 6
    assert res["string_changed_vs_normalized"] == 0.0
    assert res["string_changed_vs_raw"] == 0.0
    bs = res["by_severity"]
    assert isinstance(bs, dict) and set(bs) == {"0", "(0,0.33]", "(0.33,0.66]", "(0.66,1]"}
    pm = res["paragraph_match_rate"]
    assert isinstance(pm, float) and 0.0 <= pm <= 1.0


def test_run_reads_sets_and_writes_structure(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "V9.jsonl", ITEMS)
    out = run(["identity", "rules"], ["V9"], tmp_path)
    systems = out["systems"]
    assert isinstance(systems, dict)
    assert set(systems) == {"identity", "rules"}
    assert "V9" in systems["identity"]
    md = render_table(out)
    assert "| V9" in md or "V9" in md
    assert "macro_f1" in md.lower() or "macro-F1" in md


class _Joiner:
    name = "joiner"
    budget = 256

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        return [Gap.JOIN] * len(current)


def test_evaluate_set_survives_join_predictions() -> None:
    res = evaluate_set(_Joiner(), [EvalItem("j", "t", "que ries", "queries", 0.5, {})])
    gap = res["gap"]
    assert isinstance(gap, dict) and gap["n_gaps"] == 1
    assert gap["per_class"]["JOIN"]["f1"] == 1.0
