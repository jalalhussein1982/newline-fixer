import random

from hypothesis import given, settings
from hypothesis import strategies as st

from newline_fixer.corrupt import CorruptConfig, Corrupted, corrupt
from newline_fixer.text import Gap, content, derive_labels, normalize, split

CLEAN = (
    "2.1 Background\n\n"
    "Recurrent models align positions to steps in computation time. They generate "
    "hidden states as a function of the previous state and the input.\n\n"
    "Three advantages:\n"
    "• total computational complexity per layer\n"
    "• the amount of computation that can be parallelized\n"
    "• the path length between long-range dependencies\n\n"
    "The Transformer uses attention in three different ways."
)


def test_same_seed_same_output() -> None:
    a = corrupt(CLEAN, random.Random(7))
    b = corrupt(CLEAN, random.Random(7))
    assert a == b


def test_content_preserved_and_severity_in_range() -> None:
    for seed in range(50):
        out = corrupt(CLEAN, random.Random(seed))
        assert isinstance(out, Corrupted)
        assert content(out.text) == content(CLEAN)
        assert 0.0 <= out.severity <= 1.0


def test_clean_fraction_one_returns_normalized_input() -> None:
    out = corrupt("  a  b\n\n\nc ", random.Random(1), CorruptConfig(clean_fraction=1.0))
    assert out == Corrupted("a b\n\nc", 0.0)


def test_short_input_is_untouched() -> None:
    assert corrupt("word", random.Random(1)) == Corrupted("word", 0.0)
    assert corrupt("", random.Random(1)) == Corrupted("", 0.0)


def test_produces_every_kind_of_noise() -> None:
    cfg = CorruptConfig(clean_fraction=0.0)
    seen_join = seen_leading_space = seen_removed_break = seen_wrong_para = False
    for seed in range(200):
        out = corrupt(CLEAN, random.Random(seed), cfg)
        labels = derive_labels(out.text, CLEAN)
        _, current = split(out.text)
        seen_join |= Gap.JOIN in labels
        seen_leading_space |= "\n " in out.text
        seen_removed_break |= any(
            c is Gap.SPACE and t in (Gap.NL, Gap.PARA) for c, t in zip(current, labels, strict=True)
        )
        seen_wrong_para |= any(
            c is Gap.PARA and t is not Gap.PARA for c, t in zip(current, labels, strict=True)
        )
    assert seen_join and seen_leading_space and seen_removed_break and seen_wrong_para


def test_severity_scales_noise() -> None:
    cfg = CorruptConfig(clean_fraction=0.0)
    changes: list[tuple[float, int]] = []
    for seed in range(300):
        out = corrupt(CLEAN, random.Random(seed), cfg)
        labels = derive_labels(out.text, CLEAN)
        _, current = split(out.text)
        n_changed = sum(1 for c, t in zip(current, labels, strict=True) if c != t)
        changes.append((out.severity, n_changed))
    low = [n for s, n in changes if s < 0.3]
    high = [n for s, n in changes if s > 0.7]
    assert sum(high) / len(high) > sum(low) / len(low)


@settings(max_examples=300)
@given(st.text(min_size=2), st.integers(min_value=0, max_value=10_000))
def test_invariant_on_arbitrary_unicode(text: str, seed: int) -> None:
    out = corrupt(text, random.Random(seed), CorruptConfig(clean_fraction=0.0))
    assert content(out.text) == content(text)
    assert (
        out.text == normalize(out.text)
        or "\n " in out.text
        or " \n" in out.text
        or "\n\n" in out.text
    )
