import pytest

from newline_fixer.eval.metrics import ClassCounts, GapMetrics, paragraph_match
from newline_fixer.text import Gap

J, S, N, P = Gap.JOIN, Gap.SPACE, Gap.NL, Gap.PARA


def test_class_counts_properties() -> None:
    c = ClassCounts(tp=2, fp=1, fn=1, support=3)
    assert c.precision == pytest.approx(2 / 3)
    assert c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3)
    assert ClassCounts().f1 == 0.0


def test_gap_metrics_counts() -> None:
    m = GapMetrics()
    m.update(pred=[S, N, J, P], ref=[S, S, J, N], current=[S, S, N, N])
    assert m.n_gaps == 4
    assert m.per_class[S] == ClassCounts(tp=1, fp=0, fn=1, support=2)
    assert m.per_class[N] == ClassCounts(tp=0, fp=1, fn=1, support=1)
    assert m.per_class[J] == ClassCounts(tp=1, fp=0, fn=0, support=1)
    assert m.per_class[P] == ClassCounts(tp=0, fp=1, fn=0, support=0)
    assert m.n_changed == 3
    assert m.n_wrong_join == 0
    assert m.macro_classes() == [J, S, N]
    assert m.n_items == 1 and m.n_items_untouched == 0


def test_wrong_join_and_untouched() -> None:
    m = GapMetrics()
    m.update(pred=[J, S], ref=[S, S], current=[S, S])
    m.update(pred=[S], ref=[S], current=[S])
    assert m.n_wrong_join == 1
    assert m.wrong_join_per_1000() == pytest.approx(1000 / 3)
    assert m.n_items == 2 and m.n_items_untouched == 1
    assert m.damage_rate() == pytest.approx(1 / 3)


def test_break_f1() -> None:
    m = GapMetrics()
    m.update(pred=[N, S, P, S], ref=[N, N, S, S], current=[S, S, S, S])
    # breaks: pred {0,2}, ref {0,1}: tp 1, fp 1, fn 1 -> f1 0.5
    assert m.break_f1() == pytest.approx(0.5)


def test_update_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        GapMetrics().update(pred=[S], ref=[S, S], current=[S, S])


def test_paragraph_match() -> None:
    ref = "Title\n\nFirst para.\n\nSecond para."
    assert paragraph_match("Title\n\nFirst para.\n\nSecond para.", ref) == (3, 3)
    assert paragraph_match("Title First para.\n\nSecond para.", ref) == (1, 3)
    assert paragraph_match("", ref) == (0, 3)


def test_to_dict_has_expected_keys() -> None:
    m = GapMetrics()
    m.update(pred=[S], ref=[S], current=[S])
    d = m.to_dict()
    assert set(d) >= {
        "n_gaps",
        "macro_f1",
        "macro_classes",
        "break_f1",
        "wrong_join_per_1000",
        "damage_rate",
        "per_class",
    }
