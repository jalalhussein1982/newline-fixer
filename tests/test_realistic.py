import random
from pathlib import Path
from types import SimpleNamespace

from newline_fixer.data.realistic import (
    cut_raw_passages,
    join_hyphenation,
    load_reviewed,
    propose_target,
    validate_pair,
)

RAW = (
    "3.2.3 Applications of Attention in our Model\n"
    "The Transformer uses multi-head attention in three different ways:\n"
    '• In "encoder-decoder attention" layers, the que-\n'
    "ries come from the previous decoder layer, and the memory keys and val-\n"
    "ues come from the output of the encoder.\n"
)


def test_join_hyphenation() -> None:
    out, n = join_hyphenation(RAW)
    assert n == 2
    assert "que\nries" in out and "val\nues" in out
    assert "state-of-the-art" in join_hyphenation("a state-of-the-art\nresult")[0]


def test_cut_raw_passages_bounds() -> None:
    raw = "\f".join((f"Para {i}. " + "text " * 60 + "\n\n") * 3 for i in range(10))
    out = cut_raw_passages(raw, random.Random(0), per_doc=4)
    assert 1 <= len(out) <= 4
    for p in out:
        assert 300 <= len(p) <= 800
        assert "\f" not in p


def test_validate_pair_counts() -> None:
    assert validate_pair("a b\nc", "a\nb c") == {"unreachable": 0, "gaps": 2}
    assert validate_pair("a.B", "a.\n\nB")["unreachable"] == 1


def test_propose_target_rejects_content_change() -> None:
    bad = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **k: SimpleNamespace(content=[SimpleNamespace(text="changed words")])
        )
    )
    import pytest

    with pytest.raises(RuntimeError):
        propose_target("some words", "m", client=bad)
    good = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **k: SimpleNamespace(content=[SimpleNamespace(text="some\nwords")])
        )
    )
    assert propose_target("some words", "m", client=good) == "some\nwords"


def test_load_reviewed_only_returns_reviewed(tmp_path: Path) -> None:
    import json

    d = tmp_path / "doc1"
    d.mkdir()
    for nn, _ok in (("00", True), ("01", False)):
        (d / f"{nn}.raw.txt").write_text("a b")
        (d / f"{nn}.input.txt").write_text("a b")
        (d / f"{nn}.target.txt").write_text("a\nb")
    (tmp_path / "sources.json").write_text(
        json.dumps({"dev": [{"doc": "doc1", "title": "t", "url": "u"}], "test": []})
    )
    (tmp_path / "review.json").write_text(
        json.dumps({"doc1/00": {"reviewer": "JH", "date": "2026-10-02", "note": ""}})
    )
    items = load_reviewed(tmp_path, "dev")
    assert [i.id for i in items] == ["doc1/00"]
    assert items[0].meta["unreachable"] == 0
    assert load_reviewed(tmp_path, "test") == []


def test_cut_raw_passages_drops_non_prose() -> None:
    prose = "Word " + "word " * 78 + "end."
    dots = "Section 1.1 . . . . . . . . 12\n" * 13
    raw = prose + "\n\n" + dots
    out = cut_raw_passages(raw, random.Random(0), per_doc=8)
    assert out == [prose]


def test_load_reviewed_computed_meta_wins(tmp_path: Path) -> None:
    import json

    d = tmp_path / "doc1"
    d.mkdir()
    (d / "00.raw.txt").write_text("a b")
    (d / "00.input.txt").write_text("a b")
    (d / "00.target.txt").write_text("a\nb")
    (tmp_path / "sources.json").write_text(
        json.dumps({"dev": [{"doc": "doc1", "title": "t", "url": "u"}], "test": []})
    )
    (tmp_path / "review.json").write_text(
        json.dumps({"doc1/00": {"reviewer": "JH", "unreachable": 99}})
    )
    items = load_reviewed(tmp_path, "dev")
    assert items[0].meta["unreachable"] == 0
    assert items[0].meta["reviewer"] == "JH"
