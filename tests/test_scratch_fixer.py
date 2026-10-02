from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from newline_fixer.models.registry import FIXER_NAMES, get_fixer
from newline_fixer.models.scratch import WEIGHTS_ENV, ScratchFixer
from newline_fixer.models.scratch_config import TINY
from newline_fixer.models.vocab import CharVocab, WordVocab
from newline_fixer.text import Gap, content
from newline_fixer.windows import fix

CPU = torch.device("cpu")
WORDS = ["the", "cat", "sat", "on", "the", "mat", "and", "then", "the", "dog", "ran"]


def tiny_fixer(seed: int = 0) -> ScratchFixer:
    return ScratchFixer.untrained(
        TINY, WordVocab.build(WORDS, 20), CharVocab.build(WORDS, 1), seed, CPU
    )


def test_protocol_fields_and_predict_length() -> None:
    fx = tiny_fixer()
    assert fx.name == "scratch" and fx.budget == TINY.budget
    assert fx.token_cost("anything") == 1 and fx.gap_cost(Gap.NL) == 0 and fx.overhead() == 0
    out = fx.predict(["the", "cat", "sat"], [Gap.SPACE, Gap.NL])
    assert len(out) == 2 and all(isinstance(g, Gap) for g in out)
    assert fx.predict(["one"], []) == []
    assert len(fx.predict(["a", "b"], [Gap.SPACE])) == 1


def test_fix_preserves_content_on_long_and_unknown_tokens() -> None:
    fx = tiny_fixer()
    url = "https://example.com/" + "x" * 2000
    text = " ".join(["a"] * 40 + [url, "日本語", "😀"] + ["b"] * 40)
    res = fix(text, fx)
    assert content(res.text) == content(text)
    assert url in res.text and res.gaps == 82 and res.windows > 1


@settings(max_examples=25, deadline=None)
@given(st.text())
def test_fix_preserves_content_property(text: str) -> None:
    assert content(fix(text, tiny_fixer()).text) == content(text)


def test_save_load_roundtrip_gives_identical_predictions(tmp_path: Path) -> None:
    fx = tiny_fixer(seed=3)
    tokens, current = ["the", "cat", "sat", "on"], [Gap.NL, Gap.SPACE, Gap.PARA]
    fx.save(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "chars.json",
        "config.json",
        "model.pt",
        "words.json",
    ]
    again = ScratchFixer.load(tmp_path, CPU)
    assert again.predict(tokens, current) == fx.predict(tokens, current)
    assert ScratchFixer.load(str(tmp_path)).cfg == TINY


def test_registry_scratch_needs_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "scratch" in FIXER_NAMES
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="NF_WEIGHTS"):
        get_fixer("scratch")
    tiny_fixer().save(tmp_path)
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    assert get_fixer("scratch").name == "scratch"


def test_default_weights_prefers_nf_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    from newline_fixer.models.scratch import default_weights

    monkeypatch.setenv("NF_WEIGHTS_SCRATCH", "per-model")
    monkeypatch.setenv("NF_WEIGHTS", "global")
    assert default_weights() == "global"
    monkeypatch.delenv("NF_WEIGHTS")
    assert default_weights() == "per-model"
