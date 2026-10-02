from pathlib import Path

import torch

from newline_fixer.models.vocab import (
    NO_GAP,
    PAD,
    UNK,
    CharVocab,
    WordVocab,
    collate,
    encode_window,
)
from newline_fixer.text import Gap

TOKENS = ["the", "cat", "sat", "on", "the", "mat", "the", "end"]


def test_word_vocab_build_lowercases_and_caps_size() -> None:
    v = WordVocab.build(["The", "the", "cat", "Cat", "sat"], size=2)
    assert len(v) == 4  # pad, unk, the, cat
    assert v.encode("THE") == v.encode("the") == 2
    assert v.encode("sat") == UNK and v.encode("zzz") == UNK


def test_char_vocab_unknown_and_truncation() -> None:
    cv = CharVocab.build(["aaaaab", "aaaaab"], min_count=3)  # 'a' appears 10 times, 'b' twice
    ids = cv.encode("ab", max_chars=4)
    assert len(ids) == 4 and ids[0] != UNK and ids[1] == UNK and ids[2:] == [PAD, PAD]
    long = cv.encode("a" * 100, max_chars=4)
    assert len(long) == 4 and all(i == ids[0] for i in long)
    assert all(i == UNK for i in cv.encode("日本😀", max_chars=3))


def test_vocab_roundtrip(tmp_path: Path) -> None:
    wv = WordVocab.build(TOKENS, size=5)
    cv = CharVocab.build(TOKENS, min_count=1)
    wv.save(tmp_path / "w.json")
    cv.save(tmp_path / "c.json")
    assert WordVocab.load(tmp_path / "w.json").encode("the") == wv.encode("the")
    assert CharVocab.load(tmp_path / "c.json").encode("cat", 3) == cv.encode("cat", 3)


def test_encode_window_shapes() -> None:
    wv = WordVocab.build(TOKENS, size=10)
    cv = CharVocab.build(TOKENS, min_count=1)
    e = encode_window(["the", "cat"], [Gap.NL], wv, cv, max_chars=5)
    assert e.n == 2 and len(e.word) == 2 and len(e.char) == 2 and len(e.char[0]) == 5
    assert e.gap == [int(Gap.NL)]


def test_collate_pads_and_marks_no_gap() -> None:
    wv = WordVocab.build(TOKENS, size=10)
    cv = CharVocab.build(TOKENS, min_count=1)
    a = encode_window(["the", "cat", "sat"], [Gap.SPACE, Gap.PARA], wv, cv, 4)
    b = encode_window(["on", "mat"], [Gap.JOIN], wv, cv, 4)
    batch = collate([a, b], torch.device("cpu"))
    assert batch.word.shape == (2, 3) and batch.char.shape == (2, 3, 4)
    assert batch.gap_after.tolist() == [[1, 3, NO_GAP], [0, NO_GAP, NO_GAP]]
    assert batch.lengths.tolist() == [3, 2] and batch.lengths.device.type == "cpu"
    assert batch.word[1, 2].item() == PAD
