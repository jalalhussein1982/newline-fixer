# M5: Fine-Tuned Pretrained Encoder and the Pretraining Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, select, train, ablate, publish and evaluate the fine-tuned pretrained encoder of design 4.5 (the design's "M2" model, served as `finetuned`), run the pretraining ablation of design 4.6 and decision 0004, re-apply the decision rule of design 5.3 with the new numbers, and update the report, so the project's central hypothesis (does pretraining beat the rules on real text?) gets a measured answer.

**Architecture:** A `FinetunedFixer` wraps `AutoModelForTokenClassification` behind the `Fixer` protocol: the current whitespace is exposed by inserting marker words `[NL]` and `[PP]` (added as special tokens) between tokens whose current gap is NL or PARA; labels sit on the first subword of each real token and predict the gap after it; windowing uses subword costs (markers cost one unit, two special tokens of overhead, budget 512), so `windows.fix`, `eval.runner`, the service, the benchmark and the publish script need only a new registry entry and a per-model weights variable. Training reuses `epoch_examples` (fresh corruptions per epoch), a new trainer mirrors the scratch one (same record format, early stopping on V1 macro-F1, mixed precision on CUDA) and runs on Colab from a second notebook. Candidate selection, the chosen model's full run and the random-initialization ablation are three Colab runs the author launches; two decision records carry the numbers; the Dockerfile bakes both learned models' weights by pinned revision; the test sets are evaluated once for the two new systems and the report gains the three learned results decision 0004 promised.

**Tech Stack:** Python 3.12, uv, PyTorch, Hugging Face Transformers (`AutoModelForTokenClassification`, fast tokenizers), `sentencepiece` and `protobuf` (DeBERTa-v3 tokenizer), Colab T4, Hugging Face Hub, pytest, hypothesis, ruff, mypy strict, Docker. Existing modules: `text`, `windows`, `models.base`, `models.train_data` (`Example`, `epoch_examples`, `class_counts`), `models.trainer` (`TrainConfig` shape, `_dev_metrics`, record format), `models.scratch` (the pattern to mirror), `models.registry`, `service.config`, `eval.runner`, `eval.bench`, `eval.report`, `scripts/publish_weights.py`, `notebooks/train_scratch_colab.ipynb`.

**Spec:** `docs/02-design.md` 4.1 (windowing per model, long-token truncation), 4.5 (the model, verbatim below), 4.6 (ablation), 4.7 (run records), 5.2, 5.3 (decision rule), 6.2 (`NF_MODEL` `finetuned`), 6.4 (weights by revision), 9 (M5 row: "M2 selected, trained, ablated, published; report updated"), 10 (risk "uncased next-token cue lost by subword splitting: cased models only; current-gap markers"); decisions 0004 (three learned results in the report), 0006 (gate 0.0026), 0007, 0008 (what M5 must beat: rules V2 0.806; a new record supersedes 0008 only if the served model changes). Requirements Q1, Q2, Q4.

## Global Constraints

- Design 4.5, verbatim: `AutoModelForTokenClassification`; candidates `microsoft/deberta-v3-xsmall` and `distilbert-base-cased` (uncased models excluded); selection trains each candidate for one epoch on a 20k-window subset, measures validation macro-F1 and CPU latency per 256-token window, chooses by F1 subject to latency within three times that of M1 (the scratch model), recorded as a decision with the numbers; markers `[NL]` and `[PP]` between tokens whose current gap is NL or PARA, added as special tokens with the embedding matrix resized, carrying no label; labels on the first subword of each real token predicting the gap after it, other subwords and markers get the ignore index; learning rate 3e-5 to 5e-5, batch 16, up to three epochs, mixed precision on a Colab GPU, checkpoints saved to Drive, early stopping on validation macro-F1; windows follow the per-model budget rule of 4.1 with cost in subwords after marker insertion, budget 512.
- Design 4.1: a token whose cost alone exceeds the budget is truncated for the model input only, keeping its first units; reconstruction always uses the original token. `windows.make_windows` requires every token cost at most half the room, so `token_cost` is capped at `(512 - overhead) // 2 - 1 = 254` subwords and the encoder sees at most that many subwords of a long token.
- Decision 0004: the ablation trains the chosen encoder from random initialization with identical data, tokenizer, markers and schedule; the only difference is the weights. The report presents three learned results: scratch, fine-tuned, fine-tuned from random initialization.
- Design 5.3 and decision 0006: candidates have V3 clean damage at most 0.0026 and p50 under 300 ms for a 2,000-character input on the M1 Mac CPU; among candidates the highest V2 macro-F1 is served; wrong-join rate breaks ties; B1 stays if no learned model qualifies. Test sets are evaluated once per system and never used for a choice.
- Decision 0003: weights never in git; `experiments/runs/` stays git-ignored; records in `experiments/training/*.json` and results in `experiments/results/*.json` are committed; the Hub publish and the Colab runs are the author's actions (login, GPU session); the controller pauses for them.
- Configuration: `NF_MODEL` gains `finetuned` (`finetuned-ablation` is a registry entry for evaluation only and is rejected as a served model); each learned model has its own default weights source (`NF_WEIGHTS_SCRATCH`, `NF_WEIGHTS_FINETUNED`, `NF_WEIGHTS_FINETUNED_ABLATION`, each a local directory or `hf:repo@revision`); `NF_WEIGHTS` remains the override for the model selected by `NF_MODEL`; `NF_MODEL_REVISION` likewise applies to the selected model. Published defaults are constants in `service/config.py`.
- The run record format of design 4.7 is the scratch record's keys (`run_id`, `git_commit`, `started`, `seed`, `config`, `train_config`, `n_train_docs`, `n_examples_epoch0`, `class_counts_epoch0`, `n_params`, `device`, `epochs`, `best`, `best_epoch`, `seconds`, `weights_dir`, later `hub`) plus `pretrained`, `random_init`, `n_overflow_examples`, so `render_training_table` and `training_table` render it unchanged.
- Every random process takes an explicit seed; torch and CUDA seeds are set from the run seed; mixed precision only when the device is CUDA.
- Commit messages follow `type: summary`; every commit ends with a blank line and `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. mypy strict; ruff E, F, I, B, UP, SIM; `make fmt` before `make check`; `make check` passes at every commit. Unit tests never download from the Hub: encoder tests build a tiny WordPiece tokenizer with the `tokenizers` library and a tiny `DistilBertConfig` model in memory.
- The repository path contains U+2019 in "Jalal’s"; always quote it. If `uv run` reports `ModuleNotFoundError: newline_fixer`, run `chflags -R nohidden .venv` once.

## Review Focus

Inputs the spec implies but that no section spells out, each pinned by a test in the task that owns the code:

1. **A token longer than the half-budget cap** (a 2,000-character URL, hundreds of subwords): `token_cost` returns the cap, the encoder sees the first 254 subwords, and `fix()` returns the full original token unchanged. Test in Task 2.
2. **A window whose markers push the subword count past 512** (hundreds of NL gaps in a list): on the serving path `make_windows` counts gap costs so it cannot happen; on the training path the encoder truncates at 512 with no label position at or beyond 512, and the trainer counts and reports overflowed examples. Tests in Task 1 and Task 3.
3. **A token the tokenizer maps to zero subwords** (a lone control or private-use character): the token still gets a position (the `[UNK]` id) so `predict` returns exactly `len(current)` gaps for every input; the hypothesis content-preservation test runs through `fix()`. Test in Task 1 and Task 2.
4. **Markers must survive save and load**: after `save` and `load` the tokenizer maps `[NL]` and `[PP]` to the same single ids and the embedding matrix has the same size; otherwise a served model silently mis-tokenizes. Test in Task 2.
5. **The ablation must differ from the chosen run only by its initial weights**: the two records carry the same `pretrained`, config, schedule and seed, and `random_init` differs; a test on the trainer's record enforces it. Test in Task 3.

---

### Task 1: Dependencies, markers, subword encoding and costs

**Files:**
- Modify: `pyproject.toml` (`model` extra), `uv.lock`
- Create: `src/newline_fixer/models/finetune_encoding.py`, `tests/test_finetune_encoding.py`, `tests/tiny_tokenizer.py` (test helper)

**Interfaces:**
- Consumes: `text.Gap`, `train_data.Example`, `transformers.PreTrainedTokenizerFast`.
- Produces: constants `MARKERS: dict[Gap, str] = {Gap.NL: "[NL]", Gap.PARA: "[PP]"}`, `BUDGET = 512`, `OVERHEAD = 2`, `TOKEN_CAP = (BUDGET - OVERHEAD) // 2 - 1`, `IGNORE = -100`; `add_markers(tokenizer) -> int` (adds the two special tokens if absent, returns the new vocabulary size); `token_cost(tokenizer, token) -> int` (number of subwords of the token alone, at least 1, at most `TOKEN_CAP`; memoized per tokenizer with `functools.lru_cache` on `(id(tokenizer), token)` or a dict on the wrapper, implementer's choice, documented); `gap_cost(gap) -> int` (1 for NL and PARA, else 0); `@dataclass(frozen=True) Encoded(input_ids: list[int], attention_mask: list[int], label_positions: list[int], n_tokens: int, overflowed: bool)` where `label_positions[i]` is the input position of the first subword of token `i` (length `n_tokens`); `encode_window(tokens: Sequence[str], current: Sequence[Gap], tokenizer, max_len: int = BUDGET) -> Encoded` (CLS, then for each token its subwords truncated to `TOKEN_CAP`, then the marker of the gap after it if NL or PARA, then SEP; truncate the sequence to `max_len` keeping SEP last; `overflowed` true when truncation dropped anything; a token with zero subwords gets the unknown-token id); `labels_for(encoded: Encoded, target: Sequence[Gap]) -> list[int]` (length `len(input_ids)`, `IGNORE` everywhere except `label_positions[i] = int(target[i])` for `i < n_tokens - 1` and position < `len(input_ids)`); `collate(batch: Sequence[tuple[Encoded, list[int] | None]], pad_id: int, device) -> dict[str, Tensor]` (padded `input_ids`, `attention_mask`, optional `labels`, plus `label_positions` as a padded long tensor with `-1` padding and `n_tokens`).

- [ ] **Step 1: Dependencies**

In `pyproject.toml` extend the `model` extra to `["torch>=2.4,<3", "huggingface_hub>=0.25", "transformers>=4.45,<5", "sentencepiece>=0.2", "protobuf>=4"]`. Run `uv sync --all-extras`, then `uv run python -c "import transformers, tokenizers, sentencepiece; print(transformers.__version__)"`. Add `"transformers", "transformers.*", "tokenizers", "tokenizers.*", "sentencepiece"` to the mypy override list if mypy reports missing stubs (transformers ships partial typing; the override is expected).

- [ ] **Step 2: The test tokenizer helper**

`tests/tiny_tokenizer.py`:

```python
"""An offline WordPiece tokenizer for unit tests; no Hub access."""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "que", "##ries", "model", "a", "b", "c", "in", "3", ".", "##2", "##3", "Attention", "##s"]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {w: i for i, w in enumerate(VOCAB)}
    tok = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]"
    )
```

(`WordPiece` with `max_input_chars_per_word` default 100 maps a longer token to `[UNK]`; that is fine for the tests, the cap behaviour is tested with a token built from many `##s` pieces.)

- [ ] **Step 3: Write the failing tests**

`tests/test_finetune_encoding.py`:

```python
import torch

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


def tok():
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
    assert token_cost(t, "Attention" + "s" * 600) == TOKEN_CAP or token_cost(t, "Attention" + "s" * 600) == 1
    assert gap_cost(Gap.NL) == 1 and gap_cost(Gap.PARA) == 1 and gap_cost(Gap.SPACE) == 0 and gap_cost(Gap.JOIN) == 0


def test_encode_window_layout_and_label_positions() -> None:
    t = tok()
    tokens = ["the", "queries", "model"]
    current = [Gap.NL, Gap.SPACE]
    e = encode_window(tokens, current, t)
    ids = e.input_ids
    assert ids[0] == t.cls_token_id and ids[-1] == t.sep_token_id
    # the | [NL] | que ##ries | model
    nl = t.convert_tokens_to_ids(MARKERS[Gap.NL])
    assert ids[1:6] == [t.convert_tokens_to_ids("the"), nl, t.convert_tokens_to_ids("que"), t.convert_tokens_to_ids("##ries"), t.convert_tokens_to_ids("model")]
    assert e.label_positions == [1, 3, 5] and e.n_tokens == 3 and not e.overflowed
    assert e.attention_mask == [1] * len(ids)


def test_zero_subword_token_gets_unk_and_a_position() -> None:
    t = tok()
    e = encode_window(["a", "\u0001", "b"], [Gap.SPACE, Gap.SPACE], t)
    assert e.n_tokens == 3 and len(e.label_positions) == 3
    assert e.input_ids[e.label_positions[1]] == t.unk_token_id


def test_overflow_truncates_and_keeps_sep_last() -> None:
    t = tok()
    tokens = ["a"] * 600
    current = [Gap.NL] * 599  # markers double the length
    e = encode_window(tokens, current, t)
    assert e.overflowed and len(e.input_ids) == BUDGET and e.input_ids[-1] == t.sep_token_id
    assert all(p < BUDGET - 1 for p in e.label_positions)
    assert e.n_tokens == 600 and len(e.label_positions) <= 600


def test_labels_sit_on_first_subwords_only() -> None:
    t = tok()
    tokens, current, target = ["the", "queries", "model"], [Gap.NL, Gap.SPACE], [Gap.SPACE, Gap.PARA]
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
    batch = collate([(a, labels_for(a, [Gap.SPACE])), (b, labels_for(b, [Gap.PARA, Gap.NL]))], t.pad_token_id, torch.device("cpu"))
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert batch["input_ids"][0, -1].item() == t.pad_token_id and batch["labels"][0, -1].item() == IGNORE
    assert batch["label_positions"].shape[0] == 2 and batch["label_positions"][0, -1].item() == -1
    assert batch["n_tokens"].tolist() == [2, 3]
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_finetune_encoding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newline_fixer.models.finetune_encoding'`

- [ ] **Step 5: Implement `finetune_encoding.py`**

```python
"""Markers, subword costs and window encoding for the fine-tuned encoder (design 4.5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..text import Gap

MARKERS: dict[Gap, str] = {Gap.NL: "[NL]", Gap.PARA: "[PP]"}
BUDGET = 512
OVERHEAD = 2  # CLS and SEP
TOKEN_CAP = (BUDGET - OVERHEAD) // 2 - 1
IGNORE = -100


def add_markers(tokenizer: Any) -> int:
    """Register the marker words as special tokens; returns the vocabulary size."""
    missing = [m for m in MARKERS.values() if tokenizer.convert_tokens_to_ids(m) == tokenizer.unk_token_id]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return len(tokenizer)


def _pieces(tokenizer: Any, token: str) -> list[int]:
    ids: list[int] = tokenizer.encode(token, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.unk_token_id]
    return ids[:TOKEN_CAP]


_COST_CACHE: dict[tuple[int, str], int] = {}


def token_cost(tokenizer: Any, token: str) -> int:
    key = (id(tokenizer), token)
    if key not in _COST_CACHE:
        if len(_COST_CACHE) > 200_000:
            _COST_CACHE.clear()
        _COST_CACHE[key] = len(_pieces(tokenizer, token))
    return _COST_CACHE[key]


def gap_cost(gap: Gap) -> int:
    return 1 if gap in MARKERS else 0


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    attention_mask: list[int]
    label_positions: list[int]
    n_tokens: int
    overflowed: bool


def encode_window(tokens: Sequence[str], current: Sequence[Gap], tokenizer: Any, max_len: int = BUDGET) -> Encoded:
    ids = [tokenizer.cls_token_id]
    positions: list[int] = []
    overflowed = False
    for i, token in enumerate(tokens):
        pieces = _pieces(tokenizer, token)
        if len(ids) + len(pieces) > max_len - 1:
            overflowed = True
            break
        positions.append(len(ids))
        ids.extend(pieces)
        if i < len(current) and current[i] in MARKERS:
            if len(ids) + 1 > max_len - 1:
                overflowed = True
                break
            ids.append(tokenizer.convert_tokens_to_ids(MARKERS[current[i]]))
    ids.append(tokenizer.sep_token_id)
    return Encoded(ids, [1] * len(ids), positions, len(tokens), overflowed)


def labels_for(encoded: Encoded, target: Sequence[Gap]) -> list[int]:
    labels = [IGNORE] * len(encoded.input_ids)
    for i, pos in enumerate(encoded.label_positions):
        if i < encoded.n_tokens - 1 and i < len(target):
            labels[pos] = int(target[i])
    return labels


def collate(batch: Sequence[tuple[Encoded, list[int] | None]], pad_id: int, device: torch.device) -> dict[str, Tensor]:
    width = max(len(e.input_ids) for e, _ in batch)
    n_pos = max(len(e.label_positions) for e, _ in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    pos = torch.full((len(batch), max(n_pos, 1)), -1, dtype=torch.long)
    has_labels = all(lab is not None for _, lab in batch)
    for b, (e, lab) in enumerate(batch):
        n = len(e.input_ids)
        ids[b, :n] = torch.tensor(e.input_ids)
        mask[b, :n] = 1
        if lab is not None:
            labels[b, :n] = torch.tensor(lab)
        if e.label_positions:
            pos[b, : len(e.label_positions)] = torch.tensor(e.label_positions)
    out = {"input_ids": ids.to(device), "attention_mask": mask.to(device), "label_positions": pos.to(device),
           "n_tokens": torch.tensor([e.n_tokens for e, _ in batch])}
    if has_labels:
        out["labels"] = labels.to(device)
    return out
```

Note on `test_zero_subword_token_gets_unk_and_a_position`: the `\u0001` token must produce no pieces with the tiny tokenizer (WhitespaceSplit keeps it as one word; WordPiece maps an unknown word to `[UNK]`, one piece, so `_pieces` returns `[unk_id]` either way); the test asserts the position holds the unk id. Both paths satisfy it.

Note on `test_overflow_truncates_and_keeps_sep_last`: with 600 tokens and 599 NL markers the encoder stops at the first token that does not fit and appends SEP, so `len(input_ids) <= BUDGET`; the test's `== BUDGET` holds only when the stop lands exactly on the limit. Make the assertion `len(e.input_ids) <= BUDGET` and `>= BUDGET - 2` instead and say so in your report; the plan author's equality was too strict.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_finetune_encoding.py -v`
Expected: all PASS.

- [ ] **Step 7: `make fmt && make check`, then commit**

```bash
git add pyproject.toml uv.lock src/newline_fixer/models/finetune_encoding.py tests/test_finetune_encoding.py tests/tiny_tokenizer.py
git commit -m "feat: transformers extra; markers, subword costs and window encoding for the fine-tuned encoder"
```

---

### Task 2: `FinetunedFixer`, save and load, registry entries, per-model weights variables

**Files:**
- Create: `src/newline_fixer/models/finetuned.py`, `tests/test_finetuned_fixer.py`
- Modify: `src/newline_fixer/models/registry.py` (`finetuned`, `finetuned-ablation`), `src/newline_fixer/models/scratch.py` (`default_weights` reads `NF_WEIGHTS_SCRATCH` first, then `NF_WEIGHTS`, then `experiments/runs/current`), `src/newline_fixer/service/config.py` (`weights_source` per model; `finetuned` allowed, `finetuned-ablation` rejected as a served model; `HUB_REPO_FINETUNED`, `PUBLISHED_REVISION_FINETUNED = ""` until Task 7), `tests/test_service_config.py`, `README.md` (configuration table rows)

**Interfaces:**
- Consumes: Task 1; `models.base.Fixer`; `models.scratch.resolve_weights` (the `hf:` source resolver, reuse it, do not duplicate); `transformers.AutoModelForTokenClassification`, `AutoTokenizer`, `AutoConfig`.
- Produces: `@dataclass(frozen=True) FinetunedConfig(pretrained: str, random_init: bool = False, budget: int = BUDGET, max_len: int = BUDGET)` with `save(path)`, `load(path)` (JSON `fixer.json` in the run directory); `class FinetunedFixer` with `name = "finetuned"`, `budget`, `token_cost`, `gap_cost`, `overhead() -> 2`, `predict(tokens, current) -> list[Gap]` (one window, `torch.no_grad`, logits gathered at `label_positions`, argmax, exactly `len(current)` entries; a position missing because of overflow falls back to the current gap), `save(run_dir)` (`model.save_pretrained`, `tokenizer.save_pretrained`, `fixer.json`), `classmethod load(source: str | Path, device: torch.device | None = None) -> FinetunedFixer` (via `resolve_weights`, `weights_dir` set), `classmethod from_pretrained_name(cfg: FinetunedConfig, device) -> FinetunedFixer` (downloads the tokenizer and model, adds markers, resizes embeddings, `num_labels=4`; `random_init` builds the model from the config with `AutoModelForTokenClassification.from_config` and the same resized vocabulary), `classmethod from_parts(cfg, tokenizer, model, device)` (for tests), `weights_dir: Path | None`; registry: `"finetuned"` loads `default_weights_finetuned()` = `NF_WEIGHTS_FINETUNED` or `NF_WEIGHTS` or `experiments/runs/finetuned`; `"finetuned-ablation"` loads `NF_WEIGHTS_FINETUNED_ABLATION` or `experiments/runs/finetuned-ablation` and returns a fixer whose `name` is `"finetuned-ablation"`; `Settings.weights_source()`: `NF_WEIGHTS` if set, else `NF_WEIGHTS_<MODEL>` if set, else `hf:<repo>@<revision>` for scratch and finetuned (revision from `NF_MODEL_REVISION` when set, else the published constant), else None; `Settings(model="finetuned-ablation")` raises `ValueError("not servable")`.

- [ ] **Step 1: Write the failing tests**

`tests/test_finetuned_fixer.py` (all offline: a tiny `DistilBertConfig` model and the tiny tokenizer):

```python
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import AutoModelForTokenClassification, DistilBertConfig

from newline_fixer.models.finetune_encoding import MARKERS, TOKEN_CAP, add_markers
from newline_fixer.models.finetuned import FinetunedConfig, FinetunedFixer
from newline_fixer.text import Gap, content
from newline_fixer.windows import fix, windows_for
from tests.tiny_tokenizer import tiny_tokenizer

CPU = torch.device("cpu")


def tiny_fixer(seed: int = 0) -> FinetunedFixer:
    tok = tiny_tokenizer()
    add_markers(tok)
    torch.manual_seed(seed)
    cfg = DistilBertConfig(vocab_size=len(tok), dim=16, n_layers=1, n_heads=2, hidden_dim=32, max_position_embeddings=512, num_labels=4, pad_token_id=tok.pad_token_id)
    model = AutoModelForTokenClassification.from_config(cfg)
    return FinetunedFixer.from_parts(FinetunedConfig(pretrained="tiny", random_init=True), tok, model, CPU)


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


def test_long_token_is_capped_for_the_model_and_kept_in_the_output() -> None:
    fx = tiny_fixer()
    url = "s" * 3000
    assert fx.token_cost(url) <= TOKEN_CAP
    r = fix(f"the {url} model", fx)
    assert url in r.text and r.gaps == 2


def test_long_input_is_windowed_and_content_preserved() -> None:
    fx = tiny_fixer()
    text = "\n".join(["the queries model a b c"] * 400)  # NL gaps add marker costs
    assert len(windows_for(fx, *__import__("newline_fixer.text", fromlist=["split"]).split(text))) > 1
    assert content(fix(text, fx).text) == content(text)


def test_save_and_load_round_trip_markers_and_predictions(tmp_path: Path) -> None:
    fx = tiny_fixer()
    fx.save(tmp_path)
    assert (tmp_path / "fixer.json").exists() and (tmp_path / "config.json").exists()
    loaded = FinetunedFixer.load(tmp_path, CPU)
    for m in MARKERS.values():
        assert loaded.tokenizer.convert_tokens_to_ids(m) == fx.tokenizer.convert_tokens_to_ids(m)
    assert loaded.model.get_input_embeddings().weight.shape == fx.model.get_input_embeddings().weight.shape
    tokens, current = ["the", "queries", "model", "a"], [Gap.NL, Gap.SPACE, Gap.PARA]
    assert loaded.predict(tokens, current) == fx.predict(tokens, current)
    assert loaded.weights_dir == tmp_path


@given(st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=300))
@settings(max_examples=40, deadline=None)
def test_content_preserved_for_any_text(text: str) -> None:
    fx = tiny_fixer()
    assert content(fix(text, fx).text) == content(text)
```

Replace the `__import__` trick in `test_long_input_is_windowed_and_content_preserved` with a proper `from newline_fixer.text import split` at the top; it is written inline here only to keep the listing compact.

Add to `tests/test_service_config.py`:

```python
def test_weights_source_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from newline_fixer.service.config import HUB_REPO_FINETUNED, PUBLISHED_REVISION_FINETUNED
    s = Settings.from_env({"NF_MODEL": "finetuned", "NF_WEIGHTS_FINETUNED": "experiments/runs/finetuned"})
    assert s.weights_source() == "experiments/runs/finetuned"
    s = Settings.from_env({"NF_MODEL": "finetuned", "NF_WEIGHTS": "x", "NF_WEIGHTS_FINETUNED": "y"})
    assert s.weights_source() == "x"
    s = Settings.from_env({"NF_MODEL": "finetuned", "NF_MODEL_REVISION": "abc"})
    assert s.weights_source() == f"hf:{HUB_REPO_FINETUNED}@abc"
    s = Settings.from_env({"NF_MODEL": "scratch", "NF_WEIGHTS_SCRATCH": "z"})
    assert s.weights_source() == "z"
    with pytest.raises(ValueError, match="not servable"):
        Settings(model="finetuned-ablation")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_finetuned_fixer.py tests/test_service_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newline_fixer.models.finetuned'` and `ImportError` for the new constants.

- [ ] **Step 3: Implement `finetuned.py`**

Structure (mirror `scratch.py`; full code is the implementer's, these parts are fixed):

```python
"""The fine-tuned encoder behind the Fixer protocol (design 4.5)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..text import Gap
from .device import select_device
from .finetune_encoding import BUDGET, OVERHEAD, add_markers, collate, encode_window, gap_cost, token_cost
from .scratch import resolve_weights

WEIGHTS_ENV_FINETUNED = "NF_WEIGHTS_FINETUNED"
WEIGHTS_ENV_ABLATION = "NF_WEIGHTS_FINETUNED_ABLATION"
DEFAULT_WEIGHTS_FINETUNED = "experiments/runs/finetuned"
DEFAULT_WEIGHTS_ABLATION = "experiments/runs/finetuned-ablation"


def default_weights_finetuned() -> str:
    return os.environ.get(WEIGHTS_ENV_FINETUNED) or os.environ.get("NF_WEIGHTS") or DEFAULT_WEIGHTS_FINETUNED


def default_weights_ablation() -> str:
    return os.environ.get(WEIGHTS_ENV_ABLATION) or DEFAULT_WEIGHTS_ABLATION


@dataclass(frozen=True)
class FinetunedConfig:
    pretrained: str
    random_init: bool = False
    budget: int = BUDGET
    max_len: int = BUDGET

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> FinetunedConfig: ...


class FinetunedFixer:
    name = "finetuned"

    def __init__(self, cfg: FinetunedConfig, tokenizer: Any, model: Any, device: torch.device, name: str = "finetuned") -> None:
        self.cfg, self.tokenizer, self.device = cfg, tokenizer, device
        self.model = model.to(device).eval()
        self.budget = cfg.budget
        self.name = name
        self.weights_dir: Path | None = None

    def token_cost(self, token: str) -> int: return token_cost(self.tokenizer, token)
    def gap_cost(self, gap: Gap) -> int: return gap_cost(gap)
    def overhead(self) -> int: return OVERHEAD

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        if len(tokens) < 2:
            return []
        enc = encode_window(tokens, current, self.tokenizer, self.cfg.max_len)
        batch = collate([(enc, None)], self.tokenizer.pad_token_id, self.device)
        with torch.no_grad():
            logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[0]
        out = list(current)  # fallback for positions lost to overflow
        for i in range(len(current)):
            if i < len(enc.label_positions):
                out[i] = Gap(int(logits[enc.label_positions[i]].argmax().item()))
        return out
```

`from_pretrained_name(cfg, device)`: `tok = AutoTokenizer.from_pretrained(cfg.pretrained)`; `add_markers(tok)`; if `cfg.random_init`: `config = AutoConfig.from_pretrained(cfg.pretrained, num_labels=4)`; `model = AutoModelForTokenClassification.from_config(config)`; else `model = AutoModelForTokenClassification.from_pretrained(cfg.pretrained, num_labels=4)`; then `model.resize_token_embeddings(len(tok))`. `save(run_dir)`: `self.model.save_pretrained(run_dir)`, `self.tokenizer.save_pretrained(run_dir)`, `self.cfg.save(run_dir / "fixer.json")`. `load(source, device)`: `run_dir = resolve_weights(source)`; `cfg = FinetunedConfig.load(run_dir / "fixer.json")`; `tok = AutoTokenizer.from_pretrained(run_dir)`; `model = AutoModelForTokenClassification.from_pretrained(run_dir)`; `fixer = cls(cfg, tok, model, device or select_device())`; `fixer.weights_dir = run_dir`. The ablation registry entry calls `load(default_weights_ablation())` and sets `name = "finetuned-ablation"` (pass `name=` through a `load(..., name=...)` keyword).

Registry additions:

```python
def _finetuned() -> Fixer:
    from .finetuned import FinetunedFixer, default_weights_finetuned
    return FinetunedFixer.load(default_weights_finetuned())


def _finetuned_ablation() -> Fixer:
    from .finetuned import FinetunedFixer, default_weights_ablation
    return FinetunedFixer.load(default_weights_ablation(), name="finetuned-ablation")
```

`service/config.py`: constants `HUB_REPO_FINETUNED = "jalalhussein1982/newline-fixer-finetuned"`, `PUBLISHED_REVISION_FINETUNED = ""` (Task 7 fills it; an empty revision means `weights_source()` returns `hf:<repo>` without `@`, which `resolve_weights` already accepts as the moving main; `Settings.__post_init__` raises if `model == "finetuned"` and both the revision and any weights variable are empty, so a half-configured image fails at startup, not silently); `SERVABLE = ("identity", "rules", "scratch", "finetuned")`; `weights_source()` per the Interfaces block; `from_env` reads `NF_WEIGHTS_SCRATCH`, `NF_WEIGHTS_FINETUNED` into a `weights_by_model: dict[str, str]` field (frozen dataclass: use a tuple of pairs or `types.MappingProxyType`; the implementer picks and documents). README configuration table gains `NF_WEIGHTS_SCRATCH` and `NF_WEIGHTS_FINETUNED` rows and says `NF_MODEL` accepts `finetuned`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_finetuned_fixer.py tests/test_service_config.py tests/test_scratch_fixer.py tests/test_service_api.py -v`
Expected: all PASS; the hypothesis test runs through `fix()` with the tiny model.

- [ ] **Step 5: `make fmt && make check`, then commit**

```bash
git add src/newline_fixer/models/finetuned.py src/newline_fixer/models/registry.py src/newline_fixer/models/scratch.py src/newline_fixer/service/config.py tests/test_finetuned_fixer.py tests/test_service_config.py README.md
git commit -m "feat: FinetunedFixer behind the Fixer protocol; registry entries; per-model weights variables"
```

---

### Task 3: Trainer, training script, records

**Files:**
- Create: `src/newline_fixer/models/finetune_trainer.py`, `scripts/train_finetune.py`, `tests/test_finetune_trainer.py`
- Modify: `src/newline_fixer/models/trainer.py` (`_dev_metrics` takes `Fixer`, exported as `dev_metrics`), `scripts/publish_weights.py` (uploads every file of the run directory except `*.log`; model card reads `pretrained`/`random_init` when present), `experiments/README.md` only via `scripts/results_table.py` later

**Interfaces:**
- Consumes: Task 1 (`encode_window`, `labels_for`, `collate`, `IGNORE`), Task 2 (`FinetunedFixer`, `FinetunedConfig`), `train_data.epoch_examples`, `train_data.class_counts`, `trainer.dev_metrics`, `data.records`.
- Produces: `@dataclass(frozen=True) FinetuneTrainConfig(run_id: str, pretrained: str, random_init: bool = False, epochs: int = 3, batch_size: int = 16, lr: float = 5e-5, weight_decay: float = 0.01, warmup_fraction: float = 0.06, patience: int = 1, seed: int = 1, grad_clip: float = 1.0, max_docs: int | None = None, max_examples: int | None = None, token_budget: int = 192)`; `train_finetune(tcfg, train_docs, dev, run_dir, device) -> dict[str, object]` (the record; schedule: AdamW, linear warmup then linear decay over `epochs × steps`; `torch.autocast("cuda", dtype=torch.float16)` with `GradScaler` when the device is CUDA; per epoch: fresh `epoch_examples(docs, seed, epoch, token_budget)`, capped at `max_examples` by the shuffled order, encoded with labels, `n_overflow_examples` counted; after each epoch `dev_metrics(fixer, dev)`, `print(json.dumps(row))`, save on a new best V1 macro-F1, `run.json` written every epoch (as the scratch trainer does), early stop after `patience` epochs without improvement); CLI `uv run python scripts/train_finetune.py --run-id <id> --model <name> [--random-init] [--epochs 3 --batch 16 --lr 5e-5 --seed 1 --max-docs N --max-examples N --device cuda|cpu]` writing weights to `experiments/runs/<run-id>/` and the record to `experiments/training/<run-id>.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_finetune_trainer.py` (CPU, tiny model, a dozen tiny docs; mirror `tests/test_trainer.py` for `DOCS`/`DEV` fixtures and reuse them by import if they are module-level there, otherwise copy the smallest version):

```python
def test_train_tiny_end_to_end(tmp_path: Path) -> None:
    tcfg = FinetuneTrainConfig(run_id="t", pretrained="tiny", random_init=True, epochs=2, batch_size=4, lr=1e-3, patience=5, seed=1, max_examples=16)
    rec = train_finetune(tcfg, DOCS, DEV, tmp_path, CPU, build=tiny_builder)  # `build` injects the tiny model for tests
    assert len(rec["epochs"]) == 2 and rec["pretrained"] == "tiny" and rec["random_init"] is True
    for k in ("run_id", "git_commit", "seed", "config", "train_config", "n_train_docs", "n_examples_epoch0", "class_counts_epoch0", "n_params", "device", "best", "best_epoch", "seconds", "weights_dir", "n_overflow_examples"):
        assert k in rec
    assert (tmp_path / "fixer.json").exists() and (tmp_path / "run.json").exists()
    fx = FinetunedFixer.load(tmp_path, CPU)
    assert len(fx.predict(["a", "b", "c"], [Gap.SPACE, Gap.NL])) == 2


def test_early_stopping(tmp_path: Path) -> None:
    tcfg = FinetuneTrainConfig(run_id="e", pretrained="tiny", random_init=True, epochs=6, batch_size=4, lr=0.0, patience=1, seed=1, max_examples=8)
    rec = train_finetune(tcfg, DOCS, DEV, tmp_path, CPU, build=tiny_builder)
    assert len(rec["epochs"]) == 2


def test_ablation_record_differs_only_in_random_init(tmp_path: Path) -> None:
    a = FinetuneTrainConfig(run_id="a", pretrained="tiny", random_init=False, epochs=1, batch_size=4, seed=1, max_examples=8)
    b = FinetuneTrainConfig(run_id="b", pretrained="tiny", random_init=True, epochs=1, batch_size=4, seed=1, max_examples=8)
    ra = train_finetune(a, DOCS, DEV, tmp_path / "a", CPU, build=tiny_builder)
    rb = train_finetune(b, DOCS, DEV, tmp_path / "b", CPU, build=tiny_builder)
    ka = {k: v for k, v in ra["train_config"].items() if k not in ("run_id", "random_init")}
    kb = {k: v for k, v in rb["train_config"].items() if k not in ("run_id", "random_init")}
    assert ka == kb and ra["random_init"] is False and rb["random_init"] is True
    assert ra["n_examples_epoch0"] == rb["n_examples_epoch0"] and ra["class_counts_epoch0"] == rb["class_counts_epoch0"]


def test_dev_without_v1_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="V1"):
        train_finetune(FinetuneTrainConfig(run_id="n", pretrained="tiny", random_init=True, epochs=1), DOCS, {"V3": DEV["V3"]}, tmp_path, CPU, build=tiny_builder)
```

`tiny_builder(cfg: FinetunedConfig, device) -> FinetunedFixer` constructs the tiny fixer as in Task 2's tests (ignores `random_init` for the model but records it); `train_finetune`'s `build` parameter defaults to `FinetunedFixer.from_pretrained_name`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_finetune_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newline_fixer.models.finetune_trainer'`

- [ ] **Step 3: Implement the trainer**

Mirror `trainer.py` closely (read it first): the record dict with the same keys plus `pretrained`, `random_init`, `n_overflow_examples`; `write_record()` after every epoch; `dev_metrics(fixer, dev)` from `trainer.py` (rename `_dev_metrics` to `dev_metrics`, type the first parameter as `Fixer`, keep a `_dev_metrics = dev_metrics` alias if anything imports the old name). Training step per batch: `out = model(input_ids=..., attention_mask=..., labels=...)`; `loss = out.loss` (the HF head applies the ignore index); under CUDA wrap the forward in `torch.autocast(device_type="cuda", dtype=torch.float16)` and use `torch.amp.GradScaler("cuda")`; `nn.utils.clip_grad_norm_` after unscale; linear schedule via `torch.optim.lr_scheduler.LambdaLR`. `n_params = sum(p.numel() for p in model.parameters())`. Seeds: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`.

`scripts/train_finetune.py`: argparse as in `scripts/train_scratch.py`, reads `data/clean/train.jsonl` and `data/sets/V1.jsonl`, `V3.jsonl`, `select_device(args.device)`, calls `train_finetune`, writes the record to `experiments/training/<run-id>.json`, prints the best line.

`scripts/publish_weights.py`: replace the fixed `FILES` list with "every regular file in the run directory whose name does not end in `.log`" (`allow_patterns=None`, `ignore_patterns=["*.log", ".cache/**"]`), and let `model_card` print `pretrained` and `random_init` lines when the record has them; keep the existing scratch behaviour otherwise. Test: extend the publish test only if one exists for `publish_weights.py` (there is none; add a two-assert test of `model_card` on a finetuned-shaped record).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_finetune_trainer.py tests/test_trainer.py -v` → all PASS. Then a CPU smoke of the real script on a tiny pretrained model to prove the download path: `uv run python scripts/train_finetune.py --run-id ft-smoke --model distilbert-base-cased --max-docs 20 --max-examples 32 --epochs 1 --device cpu` (downloads about 260 MB once; expected under three minutes; prints one epoch row; writes `experiments/runs/ft-smoke/` and `experiments/training/ft-smoke.json`). Then delete both (`rm -rf experiments/runs/ft-smoke experiments/training/ft-smoke.json`) so no smoke record is committed; put the epoch row in your report.

- [ ] **Step 5: `make fmt && make check`, then commit**

```bash
git status --short   # nothing under experiments/runs/, no ft-smoke record
git add src/newline_fixer/models/finetune_trainer.py src/newline_fixer/models/trainer.py scripts/train_finetune.py scripts/publish_weights.py tests/test_finetune_trainer.py tests/test_publish_weights.py
git commit -m "feat: fine-tuning trainer with mixed precision and early stopping; training script; publish uploads any run directory"
```

---

### Task 4: Colab notebook for the three runs, per-window latency script, README

**Files:**
- Create: `notebooks/train_finetune_colab.ipynb`, `scripts/select_encoder.py`, `tests/test_select_encoder.py`
- Modify: `README.md` (Development: "Train the fine-tuned encoder" paragraph), `Makefile` (no change unless a target is handy)

**Interfaces:**
- Consumes: `scripts/train_finetune.py` (Task 3), `notebooks/train_scratch_colab.ipynb` (the pattern: clone, Drive, data zip, `train()` helper with the two-minute Drive sync, zip of runs and records), `FinetunedFixer.load`, `ScratchFixer.load`, `data/sets/V3.jsonl`, `experiments/training/*.json`.
- Produces: a notebook with cells: GPU check; clone branch `m5-finetuned-model` and `pip install --no-deps -e .` plus `pip install -q sentencepiece protobuf` (Colab ships torch and transformers); Drive mount and the `clean-train.zip` from `My Drive/newline-fixer/`; the `train()` helper calling `python -u scripts/train_finetune.py ... --device cuda` with the Drive sync; **Selection cells**: `train("ft-deberta-select", "microsoft/deberta-v3-xsmall", epochs=1, max_examples=20000)` and `train("ft-distilbert-select", "distilbert-base-cased", epochs=1, max_examples=20000)`; a zip/download cell for the selection runs (`m5-select.zip`); **Full-run cells** (run after decision 0009), parameterised by `MODEL = "<set after decision 0009>"`: `train("finetuned", MODEL, epochs=3)` and `train("finetuned-ablation", MODEL, epochs=3, random_init=True)`; a zip/download cell (`m5-runs.zip`). `scripts/select_encoder.py --runs ft-deberta-select,ft-distilbert-select --out experiments/results/m5-candidates.json`: for scratch (`experiments/runs/current`) and each run, loads the fixer on CPU, builds one 256-token window from the first V3 passages (tokens 0..255, current gaps as they are), times `predict` (3 warm-ups, 20 timed, p50 and p95 ms), reads `best.V1_macro_f1` from the run's record, writes `{"scratch_p50_ms", "limit_ms": 3 × scratch p50, "candidates": {run: {"pretrained", "V1_macro_f1", "p50_ms", "p95_ms", "within_limit": bool, "n_params"}}}` and prints a Markdown table; `eval.report` gains `candidates_table(record) -> str`. Tests: `candidates_table` formatting from a dict; the window builder returns exactly 256 tokens.

- [ ] **Step 1: Write the failing test** for `candidates_table` and the window builder (`select_window(items: Sequence[EvalItem], n_tokens: int = 256) -> tuple[list[str], list[Gap]]` in `eval/bench.py`), then implement both, then the script (no test for the script itself; it needs real weights).

- [ ] **Step 2: Write the notebook** by copying `train_scratch_colab.ipynb` with `python3` and editing the cells as listed; `python3 -c` compile-check every code cell with `!`/`%` lines stripped, as the M3 session did. The `train()` helper gains `model`, `random_init` and `max_examples` parameters mapped to the CLI flags.

- [ ] **Step 3: README** under Development: "Train the fine-tuned encoder (design 4.5): `uv run python scripts/train_finetune.py --run-id finetuned --model microsoft/deberta-v3-xsmall --epochs 3` on a GPU; on the M1 use the Colab notebook `notebooks/train_finetune_colab.ipynb`, which runs the two candidate selections, then the chosen model and its random-initialization ablation, and hands back the run directories. `uv run python scripts/select_encoder.py --runs ft-deberta-select,ft-distilbert-select --out experiments/results/m5-candidates.json` measures CPU latency per 256-token window against the scratch model's."

- [ ] **Step 4: `make fmt && make check`, then commit**

```bash
git add notebooks/train_finetune_colab.ipynb scripts/select_encoder.py src/newline_fixer/eval/bench.py src/newline_fixer/eval/report.py tests/test_select_encoder.py tests/test_report_tables.py README.md
git commit -m "feat: Colab notebook for the fine-tuning runs; encoder selection script with per-window CPU latency"
```

Then push the branch (`git push -u origin m5-finetuned-model`) so Colab can clone it; this push is of a feature branch and is part of the task.

---

### Task 5: Candidate selection runs (author) and decision 0009

**Files:**
- Produce (committed): `experiments/training/ft-deberta-select.json`, `experiments/training/ft-distilbert-select.json`, `experiments/results/m5-candidates.json`, `experiments/README.md` (re-rendered; `results_table.py` must skip or render the candidates file: add a branch in `scripts/results_table.py` that renders `m5-candidates.json` with `candidates_table` instead of `render_table`)
- Create: `docs/decisions/0009-encoder-candidate.md`

**The author's step (controller pauses):** open the notebook from the branch in Colab, T4 runtime, run the setup cells and the two selection cells, download `m5-select.zip`, unzip at the repository root.

- [ ] **Step 1 (implementer, after the zip is in place):** `uv run python scripts/select_encoder.py --runs ft-deberta-select,ft-distilbert-select --out experiments/results/m5-candidates.json`, then `uv run python scripts/results_table.py`. Read the table. The limit is three times the scratch model's p50 per 256-token window on this Mac's CPU.

- [ ] **Step 2: Decision 0009** (`docs/decisions/0009-encoder-candidate.md`, four parts, every number from the candidates file and the two records): candidates, their one-epoch V1 macro-F1, p50/p95 per window, parameter counts, within-limit flags; the rule: highest V1 macro-F1 among those within the limit; if neither is within the limit, the faster one is chosen and the record says the latency condition of design 4.5 is not met, what ratio it reached, and that the serving rule of 5.3 (300 ms at 2,000 characters on the host) is applied separately in decision 0010; the chosen `MODEL` string for the notebook.

- [ ] **Step 3: commit**

```bash
git status --short   # nothing under experiments/runs/
git add experiments/training/ft-deberta-select.json experiments/training/ft-distilbert-select.json experiments/results/m5-candidates.json experiments/README.md scripts/results_table.py docs/decisions/0009-encoder-candidate.md
git commit -m "data: encoder candidate selection runs and latency; decision 0009"
```

Push the branch again so the notebook's full-run cells (which read `MODEL`) can be used from a clone that carries the decision.

---

### Task 6: Full run and ablation (author), dev evaluation, decision 0010, served default

**Files:**
- Produce (committed): `experiments/training/finetuned.json`, `experiments/training/finetuned-ablation.json`, `experiments/results/m5-finetuned.json` (dev sets V1, V2, V3 for `identity,rules,scratch,finetuned,finetuned-ablation`), `experiments/README.md`
- Create: `docs/decisions/0010-served-model-after-m5.md`
- Modify: `src/newline_fixer/service/config.py` (`DEFAULT_MODEL`) and `Dockerfile` (`NF_MODEL` default) only if the rule picks `finetuned`; `README.md` (served-model sentence)

**The author's step (controller pauses):** in the notebook set `MODEL` to decision 0009's choice, run the full-run cells (about ten minutes each on a T4 for xsmall; the record is synced to Drive every two minutes), download `m5-runs.zip`, unzip at the repository root (`experiments/runs/finetuned/`, `experiments/runs/finetuned-ablation/`, the two records).

- [ ] **Step 1 (implementer):** `uv run python scripts/evaluate.py --systems identity,rules,scratch,finetuned,finetuned-ablation --sets V1,V2,V3 --out experiments/results/m5-finetuned.json` (CPU; the two transformer systems take a few minutes), `uv run python scripts/results_table.py`. Then the host benchmark for the new system: `uv run python scripts/bench.py --systems identity,rules,scratch,finetuned --label m1-mac-cpu --out experiments/bench/m1-mac-cpu.json` (re-measures all four so the host rows share one record; the previous file is replaced; git history keeps it), re-render.

- [ ] **Step 2: Apply the rule of design 5.3** literally with `finetuned` added: candidates = V3 damage at most 0.0026 and host p50 at 2,000 characters under 300 ms; highest V2 macro-F1; wrong-join tie-break. Write `docs/decisions/0010-served-model-after-m5.md`: context (0008 served B1; M5 adds `finetuned`; the ablation is not a serving candidate, it is evidence); options (keep B1; serve finetuned; serve scratch); the table (system, V3 damage, host p50 @2,000, candidate?, V2 macro-F1, V2 wrong-join /1k, V1 macro-F1) for rules, scratch, finetuned, finetuned-ablation; the decision and whether it supersedes 0008 ("Status: accepted; supersedes 0008" only if the served model changes, else "accepted; 0008 stands"); the ablation paragraph of decision 0004 (fine-tuned vs random-init with identical data and schedule: the V1/V2/V3 differences are the measured value of pretraining); consequences (`DEFAULT_MODEL`, the image default, what the report says, what M6 would try: ONNX only if latency is the reason a better model is not served).

- [ ] **Step 3:** align `DEFAULT_MODEL`, the Dockerfile `NF_MODEL` and the README sentence with the decision; `make fmt && make check`; commit:

```bash
git add experiments/training/finetuned.json experiments/training/finetuned-ablation.json experiments/results/m5-finetuned.json experiments/bench/m1-mac-cpu.json experiments/README.md docs/decisions/0010-served-model-after-m5.md src/newline_fixer/service/config.py Dockerfile README.md
git commit -m "data: fine-tuned encoder and its ablation evaluated on the dev sets; decision 0010"
```

---

### Task 7: Publish (author), image with both models, container benchmark, test sets once, report update

**Files:**
- Modify: `src/newline_fixer/service/config.py` (`PUBLISHED_REVISION_FINETUNED`), `Dockerfile` (second weights directory and build args: `ARG NF_FINETUNED_REVISION`, download under the same `WITH_WEIGHTS` guard to `/app/weights-finetuned`; runtime `ENV NF_WEIGHTS_SCRATCH=/app/weights NF_WEIGHTS_FINETUNED=/app/weights-finetuned`, and no global `NF_WEIGHTS`), `README.md` (Docker paragraph: both models baked; `NF_MODEL=finetuned`), `scripts/container_check.py` (no change; run it with `--model finetuned` too), `scripts/report_tables.py` (`merge_results` across `test-sets.json` and `test-sets-m5.json` with equal set hashes; training and service tables pick up the new records; a `## candidates` section), `src/newline_fixer/eval/report.py` (`merge_results(a, b) -> dict` requiring equal `sets_sha256`, concatenating `systems`, provenance listing both commits), `report.md` (sections 1, 5, 7, 8, 9, 10, 11, 12), `docs/03-implementation-plan.md` (M5 row)
- Produce (committed): `experiments/results/test-sets-m5.json` (T0–T3 for `finetuned,finetuned-ablation`, once), `experiments/bench/container-finetuned.json`, `experiments/README.md`, `experiments/training/finetuned.json` (+`hub` key)

**The author's step (controller pauses):** `uv run python scripts/publish_weights.py --run-id finetuned --repo jalalhussein1982/newline-fixer-finetuned`; paste the revision. (The ablation weights are not published; they are evidence, not a servable model; the record says so.)

- [ ] **Step 1:** set `PUBLISHED_REVISION_FINETUNED`; Dockerfile and README changes; `make container-check` (rules), then `uv run python scripts/container_check.py --no-build --model finetuned` (health must reach 200; the example may or may not match, report which), image size; container benchmark `uv run python scripts/bench.py --url http://localhost:8000 --label container-finetuned --out experiments/bench/container-finetuned.json` with `-e NF_MODEL=finetuned`; re-render.

- [ ] **Step 2: test sets once:** `uv run python scripts/evaluate.py --systems finetuned,finetuned-ablation --sets T0,T1,T2,T3 --out experiments/results/test-sets-m5.json`; `results_table.py`; `merge_results` in `report_tables.py` so the report's test tables show all five systems; a unit test for `merge_results` (equal hashes required; systems concatenated; a mismatch raises).

- [ ] **Step 3: report update.** Re-render all tables and re-paste verbatim with provenance lines and the weights captions (scratch and finetuned revisions). Section 1: the verdict sentence gains the fine-tuned result. Section 5: the fine-tuned encoder paragraph (candidates, decision 0009 numbers, schedule, Colab, parameters) and the ablation paragraph. Section 7: the Q2 verdict paragraph rewritten with five systems (state plainly whether the fine-tuned model adds value on V2 and T2 over both baselines and over scratch; the ablation's numbers as the measured value of pretraining); per-class tables now five systems. Section 8: the new service rows (host and container). Section 9: decisions 0009 and 0010. Section 10: failures of the fine-tuned model (the example, wrong joins, latency if it fails the gate). Section 11: updated. Section 12: the second notebook, the new Hub repo and revision. `docs/03-implementation-plan.md` M5 row: `plans/2026-10-02-m5-finetuned-model.md`. The design's "Changes since v2" note gains one sentence if anything in 4.5 was not followed (for example the learning rate chosen).

- [ ] **Step 4:** `make fmt && make check`; `uv run python scripts/report_tables.py` verbatim check of every table; commit:

```bash
git status --short   # nothing under experiments/runs/
git add src/newline_fixer/service/config.py Dockerfile README.md scripts/report_tables.py src/newline_fixer/eval/report.py tests/test_report_tables.py experiments/results/test-sets-m5.json experiments/bench/container-finetuned.json experiments/README.md experiments/training/finetuned.json report.md docs/03-implementation-plan.md docs/02-design.md
git commit -m "docs: fine-tuned encoder published, baked into the image, benchmarked and evaluated once on the test sets; report updated"
```

---

## Self-review against the spec

- **Design 4.5:** model class, both candidates, selection by one-epoch F1 and per-window CPU latency within 3× M1 (Tasks 4, 5), markers as special tokens with resized embeddings (Tasks 1, 2), labels on first subwords with ignore elsewhere (Task 1), lr/batch/epochs/mixed precision/Drive checkpoints/early stopping (Task 3, 4), subword-cost windows at 512 (Task 2). **4.6 / decision 0004:** ablation with identical data and schedule (Task 3 test, Task 6), three learned results in the report (Task 7). **4.7:** records in the scratch format (Task 3). **5.3:** decision 0010 (Task 6). **6.2 / 6.4:** `NF_MODEL=finetuned`, per-model weights, both models baked by revision (Tasks 2, 7). **9 (M5 row):** selected, trained, ablated, published, report updated (Tasks 5, 6, 7). **Q4:** notebook, scripts, pinned revisions.
- **Not in this milestone:** ONNX export (design 6.4 stretch; mentioned in 0010 only as the M6 route if latency blocks serving), the Space (M6), the bundle (the author produces it from `main` after this milestone's PR, `make bundle`).
- **Author's actions, where the controller pauses:** Colab selection runs (before Task 5 step 1), Colab full runs (before Task 6 step 1), Hub publish (before Task 7 step 1). Each is one notebook session or one command, as in M2 and M4.
- **Review Focus coverage:** 1 → Task 2 (`test_long_token_is_capped…`); 2 → Task 1 (`test_overflow…`) and Task 3 (`n_overflow_examples` in the record); 3 → Task 1 (`test_zero_subword…`) and Task 2 (hypothesis through `fix()`); 4 → Task 2 (`test_save_and_load_round_trip…`); 5 → Task 3 (`test_ablation_record_differs_only_in_random_init`).
- **Type consistency:** `Encoded`, `encode_window`, `labels_for`, `collate`, `token_cost`, `gap_cost`, `add_markers`, `FinetunedConfig`, `FinetunedFixer` (`from_parts`, `from_pretrained_name`, `load(source, device, name=)`, `save`), `FinetuneTrainConfig`, `train_finetune(tcfg, docs, dev, run_dir, device, build=)`, `dev_metrics`, `candidates_table`, `select_window`, `merge_results` are used with the same names and signatures in every task that touches them.
