import importlib.util
from pathlib import Path
from typing import Any

import pytest

from newline_fixer.eval.report import (
    per_class_table,
    realistic_facts,
    service_table,
    severity_table,
    summary_table,
    training_table,
)


def gap(f1: float) -> dict[str, Any]:
    return {
        "n_gaps": 100,
        "macro_f1": f1,
        "macro_classes": ["JOIN", "SPACE", "NL", "PARA"],
        "break_f1": f1,
        "wrong_join_per_1000": 0.5,
        "damage_rate": 0.01,
        "per_class": {
            c: {"support": 7, "precision": 0.5, "recall": 0.25, "f1": f1}
            for c in ("JOIN", "SPACE", "NL", "PARA")
        },
    }


def results() -> dict[str, Any]:
    band = {
        "n_items": 3,
        "n_gaps": 40,
        "macro_f1": 0.5,
        "wrong_join_per_1000": 0.0,
        "damage_rate": 0.0,
    }
    per_set = {
        "gap": gap(0.75),
        "paragraph_match_rate": 0.4,
        "by_severity": {"0": band, "(0,0.33]": band},
    }
    return {
        "git_commit": "abcdef0123456789",
        "dirty": False,
        "sets_sha256": {"V1": "1111111111111111", "V2": "2222222222222222"},
        "sets": ["V1", "V2"],
        "systems": {
            "rules": {"V1": per_set, "V2": per_set},
            "scratch": {"V1": per_set, "V2": per_set},
        },
    }


def test_summary_table_has_header_rows_and_commit() -> None:
    out = summary_table(results(), ["V1", "V2"], ["rules", "scratch"])
    assert out.startswith(
        "Rendered from commit `abcdef012345`, sets V1=111111111111, V2=222222222222."
    )
    assert out.count("\n| V1 | ") == 2 and out.count("\n| V2 | ") == 2
    assert "| 0.750 |" in out and "| 0.50 |" in out and "| 0.0100 |" in out
    assert "| changed gaps | paragraph match |" in out and "clean damage" not in out


def test_summary_table_marks_dirty_tree() -> None:
    r = results()
    r["dirty"] = True
    assert "(dirty tree)" in summary_table(r, ["V1"], ["rules"]).splitlines()[0]


def test_severity_table_one_row_per_band_and_system() -> None:
    out = severity_table(results(), "V1", ["rules", "scratch"])
    assert out.count("\n| 0 | ") == 2 and out.count("\n| (0,0.33] | ") == 2
    assert "| wrong-join /1k | changed gaps |" in out


def test_service_table_renders_dashes_for_missing() -> None:
    rec = {
        "label": "container-rules",
        "git_commit": "1234567890ab",
        "dirty": False,
        "systems": {
            "rules": {
                "disk_mb": None,
                "rss_mb": None,
                "throughput_chars_per_s": 1000,
                "latency_ms": {
                    "500": {"p50": 1.0, "p95": 2.0},
                    "2000": {"p50": 1.0, "p95": 2.0},
                    "10000": {"p50": 1.0, "p95": 2.0},
                },
            }
        },
    }
    out = service_table([rec])
    assert "| container-rules | rules |" in out and "| - | - |" in out and "1,000" in out


def test_training_table_one_row_per_record() -> None:
    rec = {
        "run_id": "r",
        "device": "cuda",
        "epochs": [{}, {}],
        "seconds": 120.0,
        "best_epoch": 2,
        "best": {"V1_macro_f1": 0.9, "V3_damage": 0.001},
        "n_params": 5_000_000,
        "git_commit": "abcdef012345ff",
    }
    out = training_table([rec])
    assert out.count("\n| r |") == 1 and "5,000,000" in out and "2.0" in out


def test_realistic_facts_counts_adjusted_and_excluded(tmp_path: Path) -> None:
    passages = {
        "adam": [("que-\nries", "que\nries"), ("same", "same"), ("x", "x")],
        "bert": [("a", "a"), ("b", "b")],
    }
    for doc, items in passages.items():
        d = tmp_path / doc
        d.mkdir()
        for i, (raw, inp) in enumerate(items):
            (d / f"{i:02d}.raw.txt").write_text(raw)
            (d / f"{i:02d}.input.txt").write_text(inp)
            (d / f"{i:02d}.target.txt").write_text(inp)
    sources = {"dev": [{"doc": "adam"}], "test": [{"doc": "bert"}]}
    review: dict[str, Any] = {
        "adam/00": {},
        "adam/01": {},
        "bert/00": {},
        "bert/01": {},
    }  # adam/02 is excluded
    facts = realistic_facts(tmp_path, review, sources)
    assert facts["V2"] == {"passages": 2, "adjusted": 1, "excluded": 1, "documents": ["adam"]}
    assert facts["T2"] == {"passages": 2, "adjusted": 0, "excluded": 0, "documents": ["bert"]}


def test_per_class_table_has_one_row_per_class_per_system() -> None:
    out = per_class_table(results(), ["V1"], ["rules", "scratch"])
    assert out.startswith("Rendered from commit `abcdef012345`")
    rows = [line for line in out.splitlines() if line.startswith("| V1 | ")]
    assert len(rows) == 8
    assert [r.split(" | ")[2] for r in rows[:4]] == ["JOIN", "SPACE", "NL", "PARA"]
    assert "| V1 | rules | NL | 7 | 0.500 | 0.250 | 0.750 |" in out


def test_report_tables_script_renders_every_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = Path(__file__).parent.parent
    if not (root / "experiments" / "results" / "test-sets.json").exists():
        pytest.skip("committed result records are absent")
    spec = importlib.util.spec_from_file_location(
        "report_tables_script", root / "scripts" / "report_tables.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    monkeypatch.chdir(root)
    script.main()
    out = capsys.readouterr().out
    for heading in (
        "## dev sets",
        "## test sets",
        "## T1 by severity band",
        "## per class, dev sets",
        "## per class, test sets",
        "## service",
        "## training",
        "## realistic sets",
    ):
        assert heading in out
    assert "Rendered from commit" in out
