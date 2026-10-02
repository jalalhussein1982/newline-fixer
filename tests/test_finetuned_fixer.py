from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import AutoModelForTokenClassification, DistilBertConfig

from newline_fixer.models.finetune_encoding import MARKERS, TOKEN_CAP, add_markers
from newline_fixer.models.finetuned import FinetunedConfig, FinetunedFixer
from newline_fixer.text import Gap, content, split
from newline_fixer.windows import fix, windows_for
from tests.tiny_tokenizer import tiny_tokenizer

CPU = torch.device("cpu")


def tiny_fixer(seed: int = 0) -> FinetunedFixer:
    tok = tiny_tokenizer()
    add_markers(tok)
    torch.manual_seed(seed)
    cfg = DistilBertConfig(  # type: ignore[no-untyped-call]
        vocab_size=len(tok),
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        max_position_embeddings=512,
        num_labels=4,
        pad_token_id=tok.pad_token_id,
    )
    model = AutoModelForTokenClassification.from_config(cfg)  # type: ignore[no-untyped-call]
    return FinetunedFixer.from_parts(
        FinetunedConfig(pretrained="tiny", random_init=True), tok, model, CPU
    )


def test_half_precision_model_is_cast_to_fp32() -> None:
    fx = tiny_fixer()
    half = fx.model.half()
    fx2 = FinetunedFixer.from_parts(fx.cfg, fx.tokenizer, half, CPU)
    assert all(p.dtype == torch.float32 for p in fx2.model.parameters())


def test_costs_and_budget() -> None:
    fx = tiny_fixer()
    assert fx.budget == 512 and fx.overhead() == 2
    assert fx.token_cost("the") == 1 and fx.token_cost("queries") == 2
    assert fx.gap_cost(Gap.NL) == 1 and fx.gap_cost(Gap.SPACE) == 0


def test_predict_returns_one_gap_per_gap() -> None:
    fx = tiny_fixer()
    out = fx.predict(["the", "queries", "model", "a"], [Gap.NL, Gap.SPACE, Gap.PARA])
    assert len(out) == 3 and all(isinstance(g, Gap) for g in out)
    assert fx.predict(["one"], []) == []


def test_overflow_falls_back_to_the_current_gaps() -> None:
    fx = tiny_fixer()
    tokens = ["the"] * 600
    current = [Gap.NL] * 599
    out = fx.predict(tokens, current)
    assert len(out) == 599
    assert out[-1] == Gap.NL  # lost to overflow: the current gap is kept


def test_long_token_is_capped_for_the_model_and_kept_in_the_output() -> None:
    fx = tiny_fixer()
    url = "s" * 3000
    assert fx.token_cost(url) <= TOKEN_CAP
    r = fix(f"the {url} model", fx)
    assert url in r.text and r.gaps == 2


def test_long_input_is_windowed_and_content_preserved() -> None:
    fx = tiny_fixer()
    text = "\n".join(["the queries model a b c"] * 400)  # NL gaps add marker costs
    assert len(windows_for(fx, *split(text))) > 1
    assert content(fix(text, fx).text) == content(text)


def test_save_and_load_round_trip_markers_and_predictions(tmp_path: Path) -> None:
    fx = tiny_fixer()
    fx.save(tmp_path)
    assert (tmp_path / "fixer.json").exists() and (tmp_path / "config.json").exists()
    loaded = FinetunedFixer.load(tmp_path, CPU)
    for m in MARKERS.values():
        assert loaded.tokenizer.convert_tokens_to_ids(m) == fx.tokenizer.convert_tokens_to_ids(m)
    assert (
        loaded.model.get_input_embeddings().weight.shape
        == fx.model.get_input_embeddings().weight.shape
    )
    tokens, current = ["the", "queries", "model", "a"], [Gap.NL, Gap.SPACE, Gap.PARA]
    assert loaded.predict(tokens, current) == fx.predict(tokens, current)
    assert loaded.weights_dir == tmp_path
    assert loaded.name == "finetuned"
    assert (
        FinetunedFixer.load(tmp_path, CPU, name="finetuned-ablation").name == "finetuned-ablation"
    )


def test_config_json_round_trip(tmp_path: Path) -> None:
    cfg = FinetunedConfig(pretrained="x/y", random_init=True, budget=300, max_len=200)
    cfg.save(tmp_path / "fixer.json")
    assert FinetunedConfig.load(tmp_path / "fixer.json") == cfg


@given(st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=300))
@settings(max_examples=40, deadline=None)
def test_content_preserved_for_any_text(text: str) -> None:
    fx = tiny_fixer()
    assert content(fix(text, fx).text) == content(text)


def test_tokenizer_overrides_map_list_form_extra_special_tokens(tmp_path: Path) -> None:
    from newline_fixer.models.finetuned import _tokenizer_overrides

    assert _tokenizer_overrides(tmp_path) == {}
    cfg = tmp_path / "tokenizer_config.json"
    cfg.write_text('{"extra_special_tokens": ["[NL]", "[PP]"]}', encoding="utf-8")
    assert _tokenizer_overrides(tmp_path) == {
        "extra_special_tokens": {"extra_0": "[NL]", "extra_1": "[PP]"}
    }
    cfg.write_text('{"extra_special_tokens": {"a": "[NL]"}}', encoding="utf-8")
    assert _tokenizer_overrides(tmp_path) == {}


def test_default_weights_prefers_nf_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    from newline_fixer.models.finetuned import default_weights_finetuned

    monkeypatch.setenv("NF_WEIGHTS_FINETUNED", "per-model")
    monkeypatch.setenv("NF_WEIGHTS", "global")
    assert default_weights_finetuned() == "global"
    monkeypatch.delenv("NF_WEIGHTS")
    assert default_weights_finetuned() == "per-model"
