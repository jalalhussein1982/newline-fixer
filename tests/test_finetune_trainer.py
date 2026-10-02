from pathlib import Path

import pytest
import torch

from newline_fixer.models.finetune_trainer import FinetuneTrainConfig, train_finetune
from newline_fixer.models.finetuned import FinetunedConfig, FinetunedFixer
from newline_fixer.text import Gap
from tests.test_finetuned_fixer import tiny_fixer
from tests.test_trainer import CPU, DEV, DOCS


def tiny_builder(cfg: FinetunedConfig, device: torch.device) -> FinetunedFixer:
    base = tiny_fixer()
    return FinetunedFixer.from_parts(cfg, base.tokenizer, base.model, device)


def test_train_tiny_end_to_end(tmp_path: Path) -> None:
    tcfg = FinetuneTrainConfig(
        run_id="t",
        pretrained="tiny",
        random_init=True,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        patience=5,
        seed=1,
        max_examples=16,
    )
    rec = train_finetune(tcfg, DOCS, DEV, tmp_path, CPU, build=tiny_builder)
    assert len(rec["epochs"]) == 2 and rec["pretrained"] == "tiny"  # type: ignore[arg-type]
    assert rec["random_init"] is True
    for k in (
        "run_id",
        "git_commit",
        "seed",
        "config",
        "train_config",
        "n_train_docs",
        "n_examples_epoch0",
        "class_counts_epoch0",
        "n_params",
        "device",
        "best",
        "best_epoch",
        "seconds",
        "weights_dir",
        "n_overflow_examples",
    ):
        assert k in rec
    assert (tmp_path / "fixer.json").exists() and (tmp_path / "run.json").exists()
    fx = FinetunedFixer.load(tmp_path, CPU)
    assert len(fx.predict(["a", "b", "c"], [Gap.SPACE, Gap.NL])) == 2


def test_early_stopping(tmp_path: Path) -> None:
    tcfg = FinetuneTrainConfig(
        run_id="e",
        pretrained="tiny",
        random_init=True,
        epochs=6,
        batch_size=4,
        lr=0.0,
        patience=1,
        seed=1,
        max_examples=8,
    )
    rec = train_finetune(tcfg, DOCS, DEV, tmp_path, CPU, build=tiny_builder)
    assert len(rec["epochs"]) == 2  # type: ignore[arg-type]


def test_ablation_record_differs_only_in_random_init(tmp_path: Path) -> None:
    def cfg(run_id: str, random_init: bool) -> FinetuneTrainConfig:
        return FinetuneTrainConfig(
            run_id=run_id,
            pretrained="tiny",
            random_init=random_init,
            epochs=1,
            batch_size=4,
            seed=1,
            max_examples=8,
        )

    ra = train_finetune(cfg("a", False), DOCS, DEV, tmp_path / "a", CPU, build=tiny_builder)
    rb = train_finetune(cfg("b", True), DOCS, DEV, tmp_path / "b", CPU, build=tiny_builder)
    ta, tb = ra["train_config"], rb["train_config"]
    assert isinstance(ta, dict) and isinstance(tb, dict)
    ka = {k: v for k, v in ta.items() if k not in ("run_id", "random_init")}
    kb = {k: v for k, v in tb.items() if k not in ("run_id", "random_init")}
    assert ka == kb and ra["random_init"] is False and rb["random_init"] is True
    assert ra["n_examples_epoch0"] == rb["n_examples_epoch0"]
    assert ra["class_counts_epoch0"] == rb["class_counts_epoch0"]


def test_dev_without_v1_is_rejected(tmp_path: Path) -> None:
    tcfg = FinetuneTrainConfig(run_id="n", pretrained="tiny", random_init=True, epochs=1)
    with pytest.raises(ValueError, match="V1"):
        train_finetune(tcfg, DOCS, {"V3": DEV["V3"]}, tmp_path, CPU, build=tiny_builder)
