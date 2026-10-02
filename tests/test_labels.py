import pytest
from hypothesis import given
from hypothesis import strategies as st

from newline_fixer.text import (
    Gap,
    derive_labels,
    gap_after_char,
    join,
    normalize,
    split,
    unreachable_count,
)


def test_gap_after_char() -> None:
    assert gap_after_char("ab c\nd") == [Gap.JOIN, Gap.SPACE, Gap.NL, Gap.JOIN]
    assert gap_after_char("") == []
    assert gap_after_char("x") == [Gap.JOIN]


def test_derive_labels_on_example(example_input: str, example_output: str) -> None:
    tokens, current = split(example_input)
    labels = derive_labels(example_input, example_output)
    assert len(labels) == len(current)
    assert join(tokens, labels) == normalize(example_output)
    by_pair = {(tokens[i], tokens[i + 1]): labels[i] for i in range(len(labels))}
    assert by_pair[("Attention", "in")] is Gap.SPACE
    assert by_pair[("Model", "The")] is Gap.PARA
    assert by_pair[("ways:", "•")] is Gap.NL
    assert by_pair[("layers,", "the")] is Gap.SPACE
    assert by_pair[("que", "ries")] is Gap.JOIN


def test_derive_labels_rejects_content_mismatch() -> None:
    with pytest.raises(ValueError):
        derive_labels("a b", "a c")


def test_unreachable_count() -> None:
    assert unreachable_count("a b", "a\nb") == 0
    assert unreachable_count("paragraph.Second", "paragraph.\n\nSecond") == 1
    assert unreachable_count("a\nb", "ab") == 0


@given(st.text(min_size=1))
def test_derive_labels_roundtrip_from_clean(clean: str) -> None:
    canon = normalize(clean)
    tokens, _ = split(canon)
    assert join(tokens, derive_labels(canon, canon)) == canon
