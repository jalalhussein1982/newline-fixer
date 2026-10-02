import importlib.util
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "publish_weights_script", Path(__file__).parent.parent / "scripts" / "publish_weights.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = {
    "run_id": "r",
    "git_commit": "abcdef0123456789",
    "n_params": 1234,
    "seed": 1,
    "best_epoch": 2,
    "best": {"V1_macro_f1": 0.9, "V3_damage": 0.01},
}


def test_model_card_for_a_finetuned_record() -> None:
    card = load_script().model_card(
        {**BASE, "pretrained": "distilbert-base-cased", "random_init": False}, "u/r"
    )
    assert "`distilbert-base-cased`; random_init False" in card


def test_model_card_for_a_scratch_record_has_no_pretrained_line() -> None:
    card = load_script().model_card(BASE, "u/r")
    assert "random_init" not in card and "BiLSTM" in card and "0.900" in card
