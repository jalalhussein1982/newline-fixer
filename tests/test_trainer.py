from pathlib import Path

import torch

from newline_fixer.data.records import CleanDoc, EvalItem
from newline_fixer.eval.table import render_training_table
from newline_fixer.models.scratch import ScratchFixer
from newline_fixer.models.scratch_config import TINY
from newline_fixer.models.trainer import TrainConfig, class_weight_tensor, train
from newline_fixer.text import Gap

CPU = torch.device("cpu")
PARA = "The quick brown fox jumps over the lazy dog near the river bank today. "
TEXT = "Title Line\n\n" + PARA * 4 + "\n\nSecond Heading\n\n" + PARA * 4
DOCS = [CleanDoc.make(f"d{i}", "t", str(i), f"g{i}", f"{i} " + TEXT) for i in range(8)]
DEV = {
    "V1": [
        EvalItem(
            "v1",
            "t",
            "Title Line The quick\nbrown fox.",
            "Title Line\n\nThe quick brown fox.",
            0.5,
            {},
        )
    ],
    "V3": [
        EvalItem("v3", "t", "Clean text\n\nstays clean.", "Clean text\n\nstays clean.", 0.0, {})
    ],
}


def test_class_weight_tensor() -> None:
    assert class_weight_tensor([10, 70, 15, 5], "none", CPU) is None
    w = class_weight_tensor([10, 70, 15, 0], "inverse", CPU)
    assert w is not None and w.shape == (4,)
    assert w[1] < w[2] < w[0] and abs(float(w.mean()) - 1.0) < 1e-6


def test_train_tiny_end_to_end(tmp_path: Path) -> None:
    tcfg = TrainConfig(
        run_id="t", epochs=3, batch_size=4, lr=1e-2, patience=5, seed=1, char_min_count=1
    )
    rec = train(TINY, tcfg, DOCS, DEV, tmp_path, CPU)
    epochs = rec["epochs"]
    assert isinstance(epochs, list) and len(epochs) == 3
    first, last = epochs[0], epochs[-1]
    assert isinstance(first, dict) and isinstance(last, dict)
    assert last["train_loss"] < first["train_loss"]
    for k in (
        "run_id",
        "git_commit",
        "seed",
        "config",
        "train_config",
        "n_train_docs",
        "device",
        "best_epoch",
        "best",
        "weights_dir",
    ):
        assert k in rec
    assert set(first) >= {"epoch", "train_loss", "seconds", "V1_macro_f1", "V3_damage"}
    assert (tmp_path / "model.pt").exists() and (tmp_path / "run.json").exists()
    fx = ScratchFixer.load(tmp_path, CPU)
    assert len(fx.predict(["a", "b", "c"], [Gap.SPACE, Gap.NL])) == 2


def test_early_stopping_stops_after_patience(tmp_path: Path) -> None:
    tcfg = TrainConfig(
        run_id="e", epochs=8, batch_size=4, lr=0.0, patience=1, seed=1, char_min_count=1
    )
    rec = train(TINY, tcfg, DOCS, DEV, tmp_path, CPU)
    epochs = rec["epochs"]
    assert isinstance(epochs, list) and len(epochs) == 2  # epoch 1 is best, epoch 2 no better, stop


def test_render_training_table_has_one_row_per_run() -> None:
    recs: list[dict[str, object]] = [
        {
            "run_id": "a",
            "git_commit": "abc123def456",
            "train_config": {"class_weights": "none"},
            "best_epoch": 2,
            "best": {"V1_macro_f1": 0.7, "V3_damage": 0.001, "V2_macro_f1": None},
            "epochs": [{}, {}],
            "n_params": 5_000_000,
            "device": "mps",
            "seconds": 120.0,
        },
    ]
    md = render_training_table(recs)
    assert "| a |" in md and "0.700" in md and "0.0010" in md
