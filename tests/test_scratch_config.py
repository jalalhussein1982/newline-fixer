from pathlib import Path

import torch

from newline_fixer.models.device import select_device
from newline_fixer.models.scratch_config import TINY, ScratchConfig


def test_defaults_match_design_4_4() -> None:
    c = ScratchConfig()
    assert (c.vocab_size, c.word_dim, c.char_dim, c.char_filters, c.char_width) == (
        30_000,
        128,
        32,
        64,
        3,
    )
    assert (c.gap_dim, c.hidden, c.layers, c.dropout, c.classifier_hidden, c.budget) == (
        8,
        192,
        2,
        0.2,
        256,
        256,
    )
    assert c.max_chars == 40


def test_config_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    TINY.save(p)
    assert ScratchConfig.load(p) == TINY
    assert ScratchConfig() != TINY


def test_select_device_prefers_explicit_and_falls_back_to_cpu() -> None:
    assert select_device("cpu") == torch.device("cpu")
    d = select_device()
    assert d.type in ("cpu", "mps")
