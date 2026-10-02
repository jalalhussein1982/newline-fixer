"""An offline WordPiece tokenizer for unit tests; no Hub access."""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "the",
    "que",
    "##ries",
    "model",
    "a",
    "b",
    "c",
    "in",
    "3",
    ".",
    "##2",
    "##3",
    "Attention",
    "##s",
]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {w: i for i, w in enumerate(VOCAB)}
    tok = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
