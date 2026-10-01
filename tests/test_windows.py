from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from newline_fixer.models.identity import IdentityFixer
from newline_fixer.text import Gap, content, normalize, split
from newline_fixer.windows import FixResult, assign_gaps, fix, make_windows, predict_all


def unit(n: int) -> list[int]:
    return [1] * n


def test_make_windows_single_window_when_it_fits() -> None:
    assert make_windows(10, unit(10), [0] * 9, 0, 256) == [(0, 10)]


def test_make_windows_overlap_by_half() -> None:
    assert make_windows(10, unit(10), [0] * 9, 0, 4) == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_make_windows_counts_gap_and_overhead_costs() -> None:
    # budget 6, overhead 2 leaves 4; tokens cost 1, gaps cost 1 -> 2 tokens + 1 gap = 3, 3 tokens = 5 > 4
    assert make_windows(5, unit(5), [1] * 4, 2, 6)[0] == (0, 2)


def test_make_windows_rejects_oversized_token() -> None:
    with pytest.raises(ValueError):
        make_windows(3, [1, 300, 1], [0, 0], 0, 256)


def test_make_windows_empty() -> None:
    assert make_windows(0, [], [], 0, 256) == []


def test_assign_gaps_covers_every_gap_once() -> None:
    windows = [(0, 4), (2, 6), (4, 8), (6, 10)]
    owner = assign_gaps(windows, 9)
    assert len(owner) == 9
    assert all(0 <= o < len(windows) for o in owner)
    assert owner[0] == 0 and owner[8] == 3
    assert owner[4] in (1, 2)


def test_assign_gaps_raises_when_uncovered() -> None:
    with pytest.raises(ValueError):
        assign_gaps([(0, 2), (3, 5)], 4)


class Flip:
    """Test fixer: turns every SPACE into NL, counts calls."""

    name = "flip"
    budget = 4

    def __init__(self) -> None:
        self.calls = 0

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        self.calls += 1
        return [Gap.NL if g is Gap.SPACE else g for g in current]


def test_predict_all_merges_windows() -> None:
    tokens = [f"t{i}" for i in range(10)]
    current = [Gap.SPACE] * 9
    flip = Flip()
    assert predict_all(flip, tokens, current) == [Gap.NL] * 9
    assert flip.calls == 4


def test_fix_identity_normalizes_only() -> None:
    res = fix("  a \t b\n\n\nc ", IdentityFixer())
    assert res == FixResult(text="a b\n\nc", tokens=3, gaps=2, changed=0, windows=1)


def test_fix_reports_changes() -> None:
    res = fix("a b c", Flip())
    assert res.text == "a\nb\nc"
    assert res.changed == 2


def test_fix_empty_whitespace_and_single_token() -> None:
    assert fix("", IdentityFixer()).text == ""
    assert fix("   \n\t ", IdentityFixer()).text == ""
    assert fix("  word  ", IdentityFixer()) == FixResult("word", 1, 0, 0, 0)


def test_fix_long_token_is_covered() -> None:
    url = "https://example.com/" + "x" * 2000
    text = " ".join(["a"] * 300 + [url] + ["b"] * 300)
    res = fix(text, Flip())
    assert content(res.text) == content(text)
    assert res.gaps == 600 and res.changed == 600


@given(st.text())
def test_fix_identity_equals_normalize(text: str) -> None:
    assert fix(text, IdentityFixer()).text == normalize(text)


@given(st.text())
def test_fix_preserves_content_with_flip(text: str) -> None:
    out = fix(text, Flip()).text
    assert content(out) == content(text)
    tokens, _ = split(text)
    assert len(tokens) < 2 or "\n" in out or " " not in out
