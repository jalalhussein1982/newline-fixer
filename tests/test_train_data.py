import random

import torch

from newline_fixer.corrupt import CorruptConfig
from newline_fixer.data.records import CleanDoc
from newline_fixer.models.train_data import (
    Example,
    batches,
    class_counts,
    epoch_examples,
    examples_from_doc,
)
from newline_fixer.models.vocab import CharVocab, WordVocab
from newline_fixer.text import Gap, content, join

PARA = "The quick brown fox jumps over the lazy dog near the river bank today. "
TEXT = "Title Line\n\n" + PARA * 6 + "\n\nSecond Heading\n\n" + PARA * 6
DOCS = [CleanDoc.make(f"d{i}", "t", str(i), f"g{i}", f"{i} " + TEXT) for i in range(6)]


def test_examples_cover_the_document_and_align() -> None:
    exs = examples_from_doc(DOCS[0], random.Random(1), budget=40)
    assert exs and all(2 <= len(e.tokens) <= 40 for e in exs)
    assert all(len(e.current) == len(e.target) == len(e.tokens) - 1 for e in exs)
    rebuilt = "".join(join(e.tokens, e.target) for e in exs)
    assert content(rebuilt) == content(DOCS[0].clean)


def test_epoch_examples_are_deterministic_and_vary_by_epoch() -> None:
    a = epoch_examples(DOCS, seed=1, epoch=0, budget=40)
    b = epoch_examples(DOCS, seed=1, epoch=0, budget=40)
    c = epoch_examples(DOCS, seed=1, epoch=1, budget=40)
    assert a == b and a != c and len(a) > len(DOCS)


def test_some_examples_are_clean() -> None:
    # The corruptor leaves clean_fraction (10%) of documents untouched; over 60 documents and
    # three epochs the chance that none is clean is below 1e-8, and the draws are seeded.
    many = [CleanDoc.make(f"m{i}", "t", str(i), f"m{i}", f"{i} " + TEXT) for i in range(60)]
    exs = [e for epoch in range(3) for e in epoch_examples(many, seed=2, epoch=epoch, budget=400)]
    assert any(e.current == e.target for e in exs)
    assert any(e.current != e.target for e in exs)


def test_class_counts_and_batches() -> None:
    exs = [
        Example(["a", "b", "c"], [Gap.SPACE, Gap.NL], [Gap.SPACE, Gap.PARA]),
        Example(["d", "e"], [Gap.NL], [Gap.JOIN]),
    ]
    assert class_counts(exs) == [1, 1, 0, 1]
    wv, cv = WordVocab.build(["a", "b", "c", "d", "e"], 10), CharVocab.build("abcde", 1)
    out = list(batches(exs, wv, cv, max_chars=3, batch_size=2, device=torch.device("cpu")))
    assert len(out) == 1
    batch, target = out[0]
    assert batch.word.shape == (2, 3) and target.tolist() == [[1, 3], [0, -100]]


def test_corrupt_config_is_honoured() -> None:
    exs = examples_from_doc(
        DOCS[1], random.Random(1), budget=400, cfg=CorruptConfig(clean_fraction=1.0)
    )
    assert all(e.current == e.target for e in exs)
