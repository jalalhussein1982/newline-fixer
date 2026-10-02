import torch
from transformers import PreTrainedTokenizerFast

from newline_fixer.models.finetune_encoding import (
    BUDGET,
    IGNORE,
    MARKERS,
    TOKEN_CAP,
    add_markers,
    collate,
    encode_window,
    gap_cost,
    labels_for,
    token_cost,
)
from newline_fixer.text import Gap
from tests.tiny_tokenizer import tiny_tokenizer


def tok() -> PreTrainedTokenizerFast:
    t = tiny_tokenizer()
    add_markers(t)
    return t


def test_markers_are_single_special_tokens() -> None:
    t = tiny_tokenizer()
    n = add_markers(t)
    assert n == len(t)
    for m in MARKERS.values():
        ids = t.convert_tokens_to_ids(m)
        assert isinstance(ids, int) and ids >= 0
        assert t.tokenize(m) == [m]
    assert add_markers(t) == n  # idempotent


def test_token_cost_counts_subwords_and_caps() -> None:
    t = tok()
    assert token_cost(t, "the") == 1
    assert token_cost(t, "queries") == 2
    assert token_cost(t, "\u0001") == 1  # zero-subword token still costs one
    assert 1 <= token_cost(t, "Attention" + "s" * 600) <= TOKEN_CAP
    assert (
        gap_cost(Gap.NL) == 1
        and gap_cost(Gap.PARA) == 1
        and gap_cost(Gap.SPACE) == 0
        and gap_cost(Gap.JOIN) == 0
    )


def test_encode_window_layout_and_label_positions() -> None:
    t = tok()
    tokens = ["the", "queries", "model"]
    current = [Gap.NL, Gap.SPACE]
    e = encode_window(tokens, current, t)
    ids = e.input_ids
    assert ids[0] == t.cls_token_id and ids[-1] == t.sep_token_id
    # the | [NL] | que ##ries | model
    nl = t.convert_tokens_to_ids(MARKERS[Gap.NL])
    assert ids[1:6] == [
        t.convert_tokens_to_ids("the"),
        nl,
        t.convert_tokens_to_ids("que"),
        t.convert_tokens_to_ids("##ries"),
        t.convert_tokens_to_ids("model"),
    ]
    assert e.label_positions == [1, 3, 5] and e.n_tokens == 3 and not e.overflowed
    assert e.attention_mask == [1] * len(ids)


def test_unknown_word_gets_unk_and_a_position() -> None:
    t = tok()
    e = encode_window(["a", "\u0001", "b"], [Gap.SPACE, Gap.SPACE], t)
    assert e.n_tokens == 3 and len(e.label_positions) == 3
    assert e.input_ids[e.label_positions[1]] == t.unk_token_id


def test_overflow_truncates_and_keeps_sep_last() -> None:
    t = tok()
    tokens = ["a"] * 600
    current = [Gap.NL] * 599  # markers double the length
    e = encode_window(tokens, current, t)
    assert (
        e.overflowed
        and BUDGET - 2 <= len(e.input_ids) <= BUDGET
        and e.input_ids[-1] == t.sep_token_id
    )
    assert all(p < BUDGET - 1 for p in e.label_positions)
    assert e.n_tokens == 600 and len(e.label_positions) <= 600


def test_labels_sit_on_first_subwords_only() -> None:
    t = tok()
    tokens, current, target = (
        ["the", "queries", "model"],
        [Gap.NL, Gap.SPACE],
        [Gap.SPACE, Gap.PARA],
    )
    e = encode_window(tokens, current, t)
    labels = labels_for(e, target)
    assert len(labels) == len(e.input_ids)
    assert labels[1] == int(Gap.SPACE) and labels[3] == int(Gap.PARA)
    assert labels[5] == IGNORE  # last token has no gap after it
    assert all(labels[i] == IGNORE for i in range(len(labels)) if i not in (1, 3))


def test_collate_pads_and_moves() -> None:
    t = tok()
    a = encode_window(["the", "model"], [Gap.SPACE], t)
    b = encode_window(["a", "b", "c"], [Gap.NL, Gap.SPACE], t)
    batch = collate(
        [(a, labels_for(a, [Gap.SPACE])), (b, labels_for(b, [Gap.PARA, Gap.NL]))],
        t.pad_token_id,
        torch.device("cpu"),
    )
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert (
        batch["input_ids"][0, -1].item() == t.pad_token_id
        and batch["labels"][0, -1].item() == IGNORE
    )
    assert batch["label_positions"].shape[0] == 2 and batch["label_positions"][0, -1].item() == -1
    assert batch["n_tokens"].tolist() == [2, 3]


class _StubTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    unk_token_id = 3
    pad_token_id = 0

    def __len__(self) -> int:
        return 20

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "\u0001":
            return []
        if text == "long":
            return list(range(10, 1010))
        return [7]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"[NL]": 4, "[PP]": 5}.get(token, self.unk_token_id)


def test_zero_subword_token_falls_back_to_unk() -> None:
    stub = _StubTokenizer()
    assert token_cost(stub, "\u0001") == 1
    e = encode_window(["a", "\u0001", "b"], [Gap.SPACE, Gap.SPACE], stub)
    assert len(e.label_positions) == 3
    assert e.input_ids[e.label_positions[1]] == stub.unk_token_id


def test_long_token_is_capped_at_token_cap() -> None:
    stub = _StubTokenizer()
    assert token_cost(stub, "long") == TOKEN_CAP
    e = encode_window(["long", "b"], [Gap.SPACE], stub)
    assert e.label_positions == [1, 1 + TOKEN_CAP] and not e.overflowed
