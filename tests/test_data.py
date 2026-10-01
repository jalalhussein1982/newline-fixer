from pathlib import Path

from newline_fixer.data.filters import has_structure, is_hard_wrapped, long_enough
from newline_fixer.data.passages import cut_passages
from newline_fixer.data.records import CleanDoc, EvalItem, read_jsonl, write_jsonl
from newline_fixer.data.splits import assign_splits, dedupe

HARD_WRAPPED = (
    "It was the best of times, it was the worst of times, it was the age of\n"
    "wisdom, it was the age of foolishness, it was the epoch of belief, it was\n"
    "the epoch of incredulity, it was the season of Light, it was the season of\n"
    "Darkness, it was the spring of hope, it was the winter of despair.\n"
)
PARAGRAPHED = (
    "History\n\n"
    "The town was founded in 1820. It grew quickly after the railway arrived.\n\n"
    "Economy\n\n"
    "Farming remains the main activity. Tourism is growing.\n"
)


def test_clean_doc_make_normalizes_and_hashes() -> None:
    d = CleanDoc.make("x", "test", "ref", "g", "  a  b\n\n\nc ")
    assert d.clean == "a b\n\nc"
    assert len(d.sha256) == 64
    assert d == CleanDoc.make("x", "test", "ref", "g", "a b\n\nc")


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [
        CleanDoc.make("1", "s", "r", "g", "hello world"),
        CleanDoc.make("2", "s", "r2", "g2", "x\n\ny"),
    ]
    p = tmp_path / "docs.jsonl"
    assert write_jsonl(p, rows) == 2
    assert read_jsonl(p, CleanDoc) == rows
    items = [EvalItem("a", "s", "in put", "in\nput", 0.5, {"k": 1})]
    q = tmp_path / "items.jsonl"
    write_jsonl(q, items)
    assert read_jsonl(q, EvalItem) == items


def test_hard_wrap_filter() -> None:
    assert is_hard_wrapped(HARD_WRAPPED)
    assert not is_hard_wrapped(PARAGRAPHED)
    assert not is_hard_wrapped("one line only")


def test_length_and_structure_filters() -> None:
    assert not long_enough("short")
    assert long_enough("x" * 200)
    assert has_structure(PARAGRAPHED)
    assert not has_structure("a single paragraph with no breaks at all " * 5)


def test_dedupe_by_hash_and_prefix() -> None:
    a = CleanDoc.make("1", "s", "r", "g", "same text " * 30)
    b = CleanDoc.make("2", "s", "r", "g", "same text " * 30)
    c = CleanDoc.make("3", "s", "r", "g", "same text " * 30 + "but longer tail")
    d = CleanDoc.make("4", "s", "r", "g", "different " * 30)
    out = dedupe([a, b, c, d])
    assert [x.id for x in out] == ["1", "4"]


def test_assign_splits_is_deterministic_and_grouped() -> None:
    groups = [f"g{i}" for i in range(1000)]
    s1 = assign_splits(groups, seed=1)
    s2 = assign_splits(groups, seed=1)
    assert s1 == s2
    counts = {k: sum(1 for v in s1.values() if v == k) for k in ("train", "val", "test")}
    assert 40 <= counts["val"] <= 60 and 40 <= counts["test"] <= 60
    assert counts["train"] == 1000 - counts["val"] - counts["test"]
    assert assign_splits(groups, seed=2) != s1


def test_cut_passages_respects_bounds_and_boundaries() -> None:
    paras = [f"Paragraph {i}. " + "word " * 40 for i in range(12)]
    text = "\n\n".join(p.strip() for p in paras)
    out = cut_passages(text, min_chars=300, max_chars=800)
    assert out
    for p in out:
        assert 300 <= len(p) <= 800
        assert p == p.strip()
        assert all(part in text for part in p.split("\n\n"))
    assert cut_passages("too short") == []


def test_cut_passages_is_deterministic() -> None:
    text = "\n\n".join(f"Sentence number {i} is here." + " filler" * 20 for i in range(20))
    assert cut_passages(text) == cut_passages(text)
