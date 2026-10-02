from hypothesis import given
from hypothesis import strategies as st

from newline_fixer.text import GAP_STR, Gap, classify_ws, content, join, normalize, split


def test_classify_ws_basic() -> None:
    assert classify_ws("") is Gap.JOIN
    assert classify_ws(" ") is Gap.SPACE
    assert classify_ws("   ") is Gap.SPACE
    assert classify_ws("\n") is Gap.NL
    assert classify_ws("\n ") is Gap.NL
    assert classify_ws(" \n  ") is Gap.NL
    assert classify_ws("\n\n") is Gap.PARA
    assert classify_ws("\n \n") is Gap.PARA
    assert classify_ws("\n\n\n\n") is Gap.PARA


def test_classify_ws_windows_line_endings() -> None:
    assert classify_ws("\r\n") is Gap.NL
    assert classify_ws("\r\n\r\n") is Gap.PARA
    assert classify_ws("\r") is Gap.NL


def test_classify_ws_tabs_and_nbsp_are_space() -> None:
    assert classify_ws("\t") is Gap.SPACE
    assert classify_ws("\u00a0") is Gap.SPACE
    assert classify_ws("\u00a0\t\u00a0") is Gap.SPACE
    assert split("a\u00a0b") == (["a", "b"], [Gap.SPACE])


def test_split_example() -> None:
    tokens, gaps = split("3.2.3 Applications of Attention\n in our Model")
    assert tokens == ["3.2.3", "Applications", "of", "Attention", "in", "our", "Model"]
    assert gaps == [Gap.SPACE, Gap.SPACE, Gap.SPACE, Gap.NL, Gap.SPACE, Gap.SPACE]


def test_split_strips_leading_and_trailing_whitespace() -> None:
    assert split("  a b \n") == (["a", "b"], [Gap.SPACE])


def test_split_empty_and_single() -> None:
    assert split("") == ([], [])
    assert split("   \n ") == ([], [])
    assert split("word") == (["word"], [])


def test_join_rebuilds_canonical_text() -> None:
    assert join(["a", "b", "c", "d"], [Gap.JOIN, Gap.NL, Gap.PARA]) == "ab\nc\n\nd"
    assert join([], []) == ""
    assert join(["x"], []) == "x"


def test_join_rejects_wrong_gap_count() -> None:
    import pytest

    with pytest.raises(ValueError):
        join(["a", "b"], [])


def test_normalize_is_idempotent_and_canonical() -> None:
    raw = "  Hello \t world \n\n\n again\r\nend  "
    assert normalize(raw) == "Hello world\n\nagain\nend"
    assert normalize(normalize(raw)) == normalize(raw)


def test_content_removes_all_whitespace() -> None:
    assert content(" a\tb\nc d ") == "abcd"


@given(st.text())
def test_split_join_preserves_content(text: str) -> None:
    tokens, gaps = split(text)
    assert content(join(tokens, gaps)) == content(text)


@given(st.text())
def test_normalize_output_is_canonical(text: str) -> None:
    out = normalize(text)
    assert out == out.strip()
    tokens, gaps = split(out)
    assert join(tokens, gaps) == out
    for g in gaps:
        assert GAP_STR[g] in ("", " ", "\n", "\n\n")
