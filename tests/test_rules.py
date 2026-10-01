from pathlib import Path

from newline_fixer.lexicon import Lexicon
from newline_fixer.rules import RulesFixer, rules_predict
from newline_fixer.text import Gap, normalize, split
from newline_fixer.windows import fix

WORDS = ["the", "model", "queries", "attention", "come", "from", "a", "use", "usea"]


def lex() -> Lexicon:
    return Lexicon(WORDS)


def gaps_for(text: str) -> list[Gap]:
    tokens, current = split(text)
    return rules_predict(tokens, current, lex())


def test_lexicon_basics() -> None:
    lx = lex()
    assert lx.known("The") and lx.known("queries") and not lx.known("que")
    assert len(lx) == len(set(WORDS))
    assert Lexicon.from_counts({"a": 5, "b": 2}, min_count=3).known("a")
    assert not Lexicon.from_counts({"a": 5, "b": 2}, min_count=3).known("b")


def test_lexicon_file_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "lex.txt"
    lex().write(p)
    assert Lexicon.from_file(p).known("attention")


def test_rule1_joins_only_real_fragments() -> None:
    assert gaps_for("que\nries") == [Gap.JOIN]
    assert gaps_for("the\nmodel") == [Gap.SPACE]
    assert gaps_for("use\na") == [Gap.SPACE]  # both known, keep the boundary


def test_rule2_lowercase_continuation_is_space() -> None:
    assert gaps_for("layers,\n the") == [Gap.SPACE]
    assert gaps_for("word\n\n, more") == [Gap.SPACE, Gap.SPACE]


def test_rule3_list_marker_after_colon_or_terminal() -> None:
    assert gaps_for("ways: • In") == [Gap.NL, Gap.SPACE]
    assert gaps_for("done. - next") == [Gap.NL, Gap.SPACE]
    assert gaps_for("x and - y") == [Gap.SPACE] * 3


def test_rule4_numbered_heading_then_capital_is_para() -> None:
    assert gaps_for("3.2.3 Applications of Attention in our Model The Transformer")[-2] is Gap.PARA
    assert gaps_for("3.2.3 Applications of Attention")[0] is Gap.SPACE
    assert gaps_for("3.2.3 Applications of Attention")[2] is Gap.SPACE


def test_rule4_title_case_heading_only_when_break_present() -> None:
    assert gaps_for("Applications Of Attention\nThe Transformer")[2] is Gap.PARA
    assert gaps_for("Applications Of Attention The Transformer")[2] is Gap.SPACE


def test_rules_solve_the_example(example_input: str, example_output: str) -> None:
    out = fix(example_input, RulesFixer(lex()))
    assert out.text == normalize(example_output)


def test_bundled_lexicon_loads() -> None:
    assert len(Lexicon.bundled()) > 50
