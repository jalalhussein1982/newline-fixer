# M2: From-Scratch Model (BiLSTM Gap Tagger) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, evaluate and record the from-scratch model M1 of the design (a character-aware BiLSTM gap tagger), wired into the existing `Fixer` protocol so it is served and evaluated by exactly the same code as the baselines.

**Architecture:** A PyTorch module predicts one of four gap classes for every gap in a window of at most 256 tokens from a word embedding, a character CNN and the current gap class. `ScratchFixer` wraps it behind the `Fixer` protocol with unit token cost and budget 256, so `windows.fix` and `eval.runner` need no change. Training examples are produced on the fly from the clean training split with the seeded corruptor, a fresh corruption per epoch; the model is selected by macro-F1 on V1 with early stopping and must pass the V3 clean-damage gate; weights live in a git-ignored run directory and later on the Hub, never in git; every run writes a small committed JSON record.

**Tech Stack:** Python 3.12, uv, PyTorch (CPU and Apple MPS), pytest, hypothesis, ruff, mypy strict. Existing modules: `text`, `corrupt`, `windows`, `models.base`, `eval.metrics`, `eval.runner`, `data.records`.

**Spec:** `docs/02-design.md` sections 3.4, 4.1, 4.4, 4.7, 5.2, 5.3, 6.4 (weights), 7; decision records 0003 (weights on the Hub, not in git), 0004, 0006 (the frozen baseline to beat). Requirements: `docs/01-requirements.md`.

## Global Constraints

- Python `>=3.12`. Package `newline_fixer`, layout `src/newline_fixer/`. mypy strict over `src`, `tests`, `scripts`; ruff rules E, F, I, B, UP, SIM; `make fmt` before `make check`; `make check` passes at every commit.
- Gap classes are exactly `JOIN=0, SPACE=1, NL=2, PARA=3` (`text.Gap`); the model emits four logits in that order.
- Every system preserves the non-whitespace character sequence; the model never produces text, only gap classes, and `windows.fix` rebuilds from the original tokens.
- `Fixer` contract (models/base.py): `name`, `budget`, `token_cost`, `gap_cost`, `overhead`, `predict(tokens, current) -> list[Gap]` with exactly `len(current)` entries; any two adjacent tokens plus their gap must fit in `budget - overhead`. M1 uses unit token cost, zero gap cost, zero overhead, budget 256 (design 4.1).
- Design 4.4 hyperparameters, verbatim: word vocabulary 30k most frequent lowercased training tokens plus unknown, dimension 128; character embedding 32, 64 filters of width 3, max-pooled; current gap class embedding dimension 8 attached to the token before the gap; two-layer bidirectional LSTM, hidden 192 per direction, dropout 0.2; classifier on the concatenation of the hidden states on both sides of the gap and the gap embedding, one hidden layer of 256, then four logits; about five million parameters; cross-entropy; AdamW, learning rate 2e-3, batch 32 windows, up to eight epochs, early stopping on validation macro-F1; MPS when available, CPU otherwise.
- Design 3.4: training windows are at most 256 tokens cut on gap boundaries; an example is tokens, current gap classes, target gap classes.
- Design 4.1: a token longer than the model's input limit is truncated for the model input only; reconstruction uses the original token.
- Every random process takes an explicit `random.Random` or seed; torch seeds are set from the run seed. No global random state.
- Dev sets V1, V2, V3 are used for every choice. Test sets T0, T1, T2, T3 are not evaluated in this milestone.
- Decision 0003: weights are never committed. `experiments/runs/` stays git-ignored (weights); `experiments/training/*.json` (run records) and `experiments/results/*.json` are committed.
- Decision 0006 numbers to beat on the dev sets: macro-F1 V1 0.635, V2 0.806; clean-damage gate on V3 0.0026; wrong joins 0.00 per thousand.
- Commit messages follow `type: summary` (`feat`, `test`, `chore`, `data`, `docs`); every commit ends with a blank line and `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- The repository path contains U+2019 in "Jalal’s"; always use the absolute path in quotes. If `uv run` reports `ModuleNotFoundError: newline_fixer`, run `chflags -R nohidden .venv` once.

## Review Focus

Inputs the spec implies but that no section spells out, each pinned by a test in the task that owns the code:

1. **A two-token window (one gap)** must pass through packing, the LSTM and the classifier and return exactly one class. Test in Task 3 and Task 4.
2. **A token longer than the character limit** (a 2,000-character URL) is truncated for the character CNN only; `fix()` returns the full token unchanged. Test in Task 4.
3. **Characters and words never seen in training** (CJK, emoji, rare symbols) map to the unknown ids and never raise. Test in Task 2 and Task 4.
4. **Clean text must stay clean**: every training epoch contains uncorrupted windows (the corruptor's `clean_fraction`) whose targets equal their inputs, and the trainer records V3 damage every epoch so a model that damages clean text is visible before it is selected. Test in Task 5 and Task 6.
5. **No MPS on the machine** (CI, Linux): device selection falls back to CPU and the saved weights load on CPU regardless of where they were trained (`map_location`). Test in Task 1 and Task 4.

---

### Task 1: PyTorch dependency, model configuration, device selection

**Files:**
- Modify: `pyproject.toml` (optional dependency group `model`)
- Create: `src/newline_fixer/models/scratch_config.py`, `src/newline_fixer/models/device.py`, `tests/test_scratch_config.py`
- Modify: `README.md` (one sentence under Development)

**Interfaces:**
- Produces: `ScratchConfig` frozen dataclass with the design 4.4 defaults and `save(path)`, `load(path)`; `TINY: ScratchConfig` for tests; `select_device(prefer: str | None = None) -> torch.device`.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
model = ["torch>=2.4,<3"]
```

and, so that Linux (CI) installs the CPU wheel instead of the default CUDA build with its multi-gigabyte NVIDIA dependencies, add at the end of the file:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Run `uv sync --all-extras` (on macOS this downloads the standard wheel with MPS support, a few hundred megabytes; expected once). Confirm `uv lock` resolved a `+cpu` torch for Linux: `grep -n "download.pytorch.org" uv.lock | head -3` prints at least one line. Then `uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`.

- [ ] **Step 2: Write the failing tests** in `tests/test_scratch_config.py`

```python
from pathlib import Path

import torch

from newline_fixer.models.device import select_device
from newline_fixer.models.scratch_config import TINY, ScratchConfig


def test_defaults_match_design_4_4() -> None:
    c = ScratchConfig()
    assert (c.vocab_size, c.word_dim, c.char_dim, c.char_filters, c.char_width) == (30_000, 128, 32, 64, 3)
    assert (c.gap_dim, c.hidden, c.layers, c.dropout, c.classifier_hidden, c.budget) == (8, 192, 2, 0.2, 256, 256)
    assert c.max_chars == 40


def test_config_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "config.json"
    TINY.save(p)
    assert ScratchConfig.load(p) == TINY
    assert TINY != ScratchConfig()


def test_select_device_prefers_explicit_and_falls_back_to_cpu() -> None:
    assert select_device("cpu") == torch.device("cpu")
    d = select_device()
    assert d.type in ("cpu", "mps")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_scratch_config.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'newline_fixer.models.device'`

- [ ] **Step 4: Implement `src/newline_fixer/models/scratch_config.py`**

```python
"""Hyperparameters of the from-scratch model (design section 4.4)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScratchConfig:
    vocab_size: int = 30_000
    word_dim: int = 128
    char_dim: int = 32
    char_filters: int = 64
    char_width: int = 3
    max_chars: int = 40
    gap_dim: int = 8
    hidden: int = 192
    layers: int = 2
    dropout: float = 0.2
    classifier_hidden: int = 256
    budget: int = 256

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ScratchConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


TINY = ScratchConfig(
    vocab_size=50,
    word_dim=8,
    char_dim=4,
    char_filters=6,
    char_width=3,
    max_chars=8,
    gap_dim=2,
    hidden=6,
    layers=1,
    dropout=0.0,
    classifier_hidden=8,
    budget=16,
)
```

- [ ] **Step 5: Implement `src/newline_fixer/models/device.py`**

```python
"""Pick the torch device: an explicit choice, else MPS when available, else CPU."""

from __future__ import annotations

import torch


def select_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 6: README** — under "Development", after the data sentence, add: "The from-scratch model needs the `model` extra (PyTorch); `uv sync --all-extras` installs it. Training uses Apple MPS when available and falls back to CPU."

- [ ] **Step 7: Run `make fmt && make check`** — expected: all pass (mypy sees torch's own type hints; no overrides needed).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/newline_fixer/models/scratch_config.py src/newline_fixer/models/device.py tests/test_scratch_config.py README.md
git commit -m "chore: torch extra; from-scratch model config and device selection"
```

---

### Task 2: Vocabularies and window encoding

**Files:**
- Create: `src/newline_fixer/models/vocab.py`, `tests/test_vocab.py`

**Interfaces:**
- Consumes: `text.Gap`.
- Produces:
  - constants `PAD = 0`, `UNK = 1`, `NO_GAP = 4` (gap id for "no gap after this token")
  - `class WordVocab` with `__init__(words: Sequence[str])` (specials prepended), `classmethod build(tokens: Iterable[str], size: int) -> WordVocab`, `encode(token: str) -> int`, `__len__`, `save(path)`, `classmethod load(path)`
  - `class CharVocab` with the same shape: `build(tokens: Iterable[str], min_count: int = 5)`, `encode(token: str, max_chars: int) -> list[int]` (ids of the first `max_chars` characters, padded with `PAD` to `max_chars`)
  - `@dataclass(frozen=True) Encoded(word: list[int], char: list[list[int]], gap: list[int], n: int)`
  - `encode_window(tokens: Sequence[str], current: Sequence[Gap], words: WordVocab, chars: CharVocab, max_chars: int) -> Encoded`
  - `@dataclass Batch(word: Tensor, char: Tensor, gap_after: Tensor, lengths: Tensor)` and `collate(encoded: Sequence[Encoded], device: torch.device) -> Batch`; `gap_after[b, i]` is the current gap class after token `i` and `NO_GAP` after the last token and in padding; `lengths` stays on the CPU for packing.

- [ ] **Step 1: Write the failing tests**

```python
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

TOKENS = "the cat sat on the mat the end".split()


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vocab.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'newline_fixer.models.vocab'`

- [ ] **Step 3: Implement `src/newline_fixer/models/vocab.py`**

```python
"""Word and character vocabularies and tensor encoding of token windows."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..text import Gap

PAD = 0
UNK = 1
NO_GAP = 4
_SPECIALS = ["<pad>", "<unk>"]


class _Vocab:
    def __init__(self, items: Sequence[str]) -> None:
        self.itos: list[str] = [*_SPECIALS, *items]
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos[len(_SPECIALS) :], ensure_ascii=False), encoding="utf-8")

    @classmethod
    def _load_items(cls, path: Path) -> list[str]:
        items: list[str] = json.loads(path.read_text(encoding="utf-8"))
        return items


class WordVocab(_Vocab):
    @classmethod
    def build(cls, tokens: Iterable[str], size: int) -> WordVocab:
        counts = Counter(t.lower() for t in tokens)
        return cls([w for w, _ in counts.most_common(size)])

    @classmethod
    def load(cls, path: Path) -> WordVocab:
        return cls(cls._load_items(path))

    def encode(self, token: str) -> int:
        return self.stoi.get(token.lower(), UNK)


class CharVocab(_Vocab):
    @classmethod
    def build(cls, tokens: Iterable[str], min_count: int = 5) -> CharVocab:
        counts: Counter[str] = Counter()
        for t in tokens:
            counts.update(t)
        return cls(sorted(c for c, n in counts.items() if n >= min_count))

    @classmethod
    def load(cls, path: Path) -> CharVocab:
        return cls(cls._load_items(path))

    def encode(self, token: str, max_chars: int) -> list[int]:
        ids = [self.stoi.get(c, UNK) for c in token[:max_chars]]
        return ids + [PAD] * (max_chars - len(ids))


@dataclass(frozen=True)
class Encoded:
    word: list[int]
    char: list[list[int]]
    gap: list[int]
    n: int


def encode_window(
    tokens: Sequence[str],
    current: Sequence[Gap],
    words: WordVocab,
    chars: CharVocab,
    max_chars: int,
) -> Encoded:
    return Encoded(
        word=[words.encode(t) for t in tokens],
        char=[chars.encode(t, max_chars) for t in tokens],
        gap=[int(g) for g in current],
        n=len(tokens),
    )


@dataclass
class Batch:
    word: Tensor
    char: Tensor
    gap_after: Tensor
    lengths: Tensor


def collate(encoded: Sequence[Encoded], device: torch.device) -> Batch:
    b = len(encoded)
    n = max(e.n for e in encoded)
    c = len(encoded[0].char[0])
    word = torch.full((b, n), PAD, dtype=torch.long)
    char = torch.full((b, n, c), PAD, dtype=torch.long)
    gap_after = torch.full((b, n), NO_GAP, dtype=torch.long)
    lengths = torch.tensor([e.n for e in encoded], dtype=torch.long)
    for i, e in enumerate(encoded):
        word[i, : e.n] = torch.tensor(e.word, dtype=torch.long)
        char[i, : e.n] = torch.tensor(e.char, dtype=torch.long)
        if e.gap:
            gap_after[i, : e.n - 1] = torch.tensor(e.gap, dtype=torch.long)
    return Batch(word.to(device), char.to(device), gap_after.to(device), lengths)
```

- [ ] **Step 4: Run `make fmt && make check`** — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/newline_fixer/models/vocab.py tests/test_vocab.py
git commit -m "feat: word and character vocabularies; window encoding and batching"
```

---

### Task 3: The network

**Files:**
- Create: `src/newline_fixer/models/scratch_net.py`, `tests/test_scratch_net.py`

**Interfaces:**
- Consumes: `ScratchConfig`, `Batch`, `NO_GAP`.
- Produces: `class CharCNN(nn.Module)` with `forward(char_ids: Tensor[B, N, C]) -> Tensor[B, N, filters]`; `class GapTagger(nn.Module)` with `__init__(cfg: ScratchConfig, n_words: int, n_chars: int)` and `forward(batch: Batch) -> Tensor[B, N-1, 4]` (logits for the gap after tokens 0..N-2); `count_parameters(module) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
import torch

from newline_fixer.models.scratch_config import TINY, ScratchConfig
from newline_fixer.models.scratch_net import CharCNN, GapTagger, count_parameters
from newline_fixer.models.vocab import NO_GAP, Batch


def make_batch(lengths: list[int], max_chars: int = 8) -> Batch:
    b, n = len(lengths), max(lengths)
    g = torch.full((b, n), NO_GAP, dtype=torch.long)
    for i, length in enumerate(lengths):
        g[i, : length - 1] = 1
    return Batch(
        word=torch.randint(2, 50, (b, n)),
        char=torch.randint(2, 20, (b, n, max_chars)),
        gap_after=g,
        lengths=torch.tensor(lengths),
    )


def test_char_cnn_shape() -> None:
    cnn = CharCNN(n_chars=20, char_dim=4, filters=6, width=3)
    out = cnn(torch.randint(0, 20, (2, 5, 8)))
    assert out.shape == (2, 5, 6)


def test_gap_tagger_logits_shape_and_two_token_window() -> None:
    net = GapTagger(TINY, n_words=50, n_chars=20)
    logits = net(make_batch([5, 2]))
    assert logits.shape == (2, 4, 4)
    single = net(make_batch([2]))
    assert single.shape == (1, 1, 4)


def test_gradients_reach_every_parameter() -> None:
    net = GapTagger(TINY, n_words=50, n_chars=20)
    logits = net(make_batch([4, 3]))
    logits.sum().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert missing == []


def test_full_config_is_about_five_million_parameters() -> None:
    net = GapTagger(ScratchConfig(), n_words=30_002, n_chars=300)
    n = count_parameters(net)
    assert 4_500_000 < n < 7_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scratch_net.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/models/scratch_net.py`**

```python
"""The from-scratch gap tagger: word embedding + character CNN + BiLSTM (design 4.4)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .scratch_config import ScratchConfig
from .vocab import NO_GAP, PAD, Batch

N_CLASSES = 4


class CharCNN(nn.Module):
    def __init__(self, n_chars: int, char_dim: int, filters: int, width: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=PAD)
        self.conv = nn.Conv1d(char_dim, filters, width, padding=width // 2)

    def forward(self, char_ids: Tensor) -> Tensor:
        b, n, c = char_ids.shape
        x = self.emb(char_ids.reshape(b * n, c)).transpose(1, 2)  # [B*N, dim, C]
        x = torch.relu(self.conv(x))  # [B*N, filters, C]
        return x.max(dim=2).values.reshape(b, n, -1)


class GapTagger(nn.Module):
    def __init__(self, cfg: ScratchConfig, n_words: int, n_chars: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(n_words, cfg.word_dim, padding_idx=PAD)
        self.char_cnn = CharCNN(n_chars, cfg.char_dim, cfg.char_filters, cfg.char_width)
        self.gap_emb = nn.Embedding(NO_GAP + 1, cfg.gap_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(
            cfg.word_dim + cfg.char_filters + cfg.gap_dim,
            cfg.hidden,
            num_layers=cfg.layers,
            bidirectional=True,
            batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(4 * cfg.hidden + cfg.gap_dim, cfg.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden, N_CLASSES),
        )

    def forward(self, batch: Batch) -> Tensor:
        gap = self.gap_emb(batch.gap_after)  # [B, N, gap_dim]
        x = torch.cat([self.word_emb(batch.word), self.char_cnn(batch.char), gap], dim=-1)
        x = self.drop(x)
        packed = pack_padded_sequence(x, batch.lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=batch.word.shape[1])
        left, right = h[:, :-1], h[:, 1:]  # hidden states on both sides of each gap
        feats = torch.cat([left, right, gap[:, :-1]], dim=-1)
        logits: Tensor = self.head(feats)
        return logits


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
```

- [ ] **Step 4: Run `make fmt && make check`** — expected: all pass. If mypy complains that `self.head(feats)` returns `Any`, the explicit `logits: Tensor` annotation above is the intended fix; do not add ignores.

- [ ] **Step 5: Commit**

```bash
git add src/newline_fixer/models/scratch_net.py tests/test_scratch_net.py
git commit -m "feat: BiLSTM gap tagger with character CNN and gap embedding"
```

---

### Task 4: `ScratchFixer`, save and load, registry entry

**Files:**
- Create: `src/newline_fixer/models/scratch.py`, `tests/test_scratch_fixer.py`
- Modify: `src/newline_fixer/models/registry.py`

**Interfaces:**
- Consumes: `Fixer` protocol, `windows.fix`, `ScratchConfig`, vocabularies, `GapTagger`, `select_device`.
- Produces:
  - `class ScratchFixer` implementing `Fixer` (name `"scratch"`, `budget = cfg.budget`, unit token cost, zero gap cost and overhead) with `__init__(cfg, words, chars, net, device)`, `predict`, `save(run_dir: Path) -> None` (writes `config.json`, `words.json`, `chars.json`, `model.pt`), `classmethod load(source: str | Path, device: torch.device | None = None) -> ScratchFixer` (a local directory in this task; Task 8 adds `hf:` sources), `classmethod untrained(cfg, words, chars, seed, device) -> ScratchFixer`
  - `WEIGHTS_ENV = "NF_WEIGHTS"`, `DEFAULT_WEIGHTS = "experiments/runs/current"`, `default_weights() -> str`
  - registry name `scratch` resolving to `ScratchFixer.load(default_weights())`, raising `FileNotFoundError` with an explanatory message when the directory is missing.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from newline_fixer.models.registry import FIXER_NAMES, get_fixer
from newline_fixer.models.scratch import WEIGHTS_ENV, ScratchFixer
from newline_fixer.models.scratch_config import TINY
from newline_fixer.models.vocab import CharVocab, WordVocab
from newline_fixer.text import Gap, content
from newline_fixer.windows import fix

CPU = torch.device("cpu")
WORDS = "the cat sat on the mat and then the dog ran".split()


def tiny_fixer(seed: int = 0) -> ScratchFixer:
    return ScratchFixer.untrained(TINY, WordVocab.build(WORDS, 20), CharVocab.build(WORDS, 1), seed, CPU)


def test_protocol_fields_and_predict_length() -> None:
    fx = tiny_fixer()
    assert fx.name == "scratch" and fx.budget == TINY.budget
    assert fx.token_cost("anything") == 1 and fx.gap_cost(Gap.NL) == 0 and fx.overhead() == 0
    out = fx.predict(["the", "cat", "sat"], [Gap.SPACE, Gap.NL])
    assert len(out) == 2 and all(isinstance(g, Gap) for g in out)
    assert fx.predict(["one"], []) == []
    assert len(fx.predict(["a", "b"], [Gap.SPACE])) == 1


def test_fix_preserves_content_on_long_and_unknown_tokens() -> None:
    fx = tiny_fixer()
    url = "https://example.com/" + "x" * 2000
    text = " ".join(["a"] * 40 + [url, "日本語", "😀"] + ["b"] * 40)
    res = fix(text, fx)
    assert content(res.text) == content(text)
    assert url in res.text and res.gaps == 82 and res.windows > 1


@settings(max_examples=25, deadline=None)
@given(st.text())
def test_fix_preserves_content_property(text: str) -> None:
    assert content(fix(text, tiny_fixer()).text) == content(text)


def test_save_load_roundtrip_gives_identical_predictions(tmp_path: Path) -> None:
    fx = tiny_fixer(seed=3)
    tokens, current = ["the", "cat", "sat", "on"], [Gap.NL, Gap.SPACE, Gap.PARA]
    fx.save(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chars.json", "config.json", "model.pt", "words.json"]
    again = ScratchFixer.load(tmp_path, CPU)
    assert again.predict(tokens, current) == fx.predict(tokens, current)
    assert ScratchFixer.load(str(tmp_path)).cfg == TINY


def test_registry_scratch_needs_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "scratch" in FIXER_NAMES
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="NF_WEIGHTS"):
        get_fixer("scratch")
    tiny_fixer().save(tmp_path)
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path))
    assert get_fixer("scratch").name == "scratch"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scratch_fixer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'newline_fixer.models.scratch'`

- [ ] **Step 3: Implement `src/newline_fixer/models/scratch.py`**

```python
"""M1 of the design: the from-scratch model behind the Fixer protocol."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import torch

from ..text import Gap
from .device import select_device
from .scratch_config import ScratchConfig
from .scratch_net import GapTagger
from .vocab import CharVocab, WordVocab, collate, encode_window

WEIGHTS_ENV = "NF_WEIGHTS"
DEFAULT_WEIGHTS = "experiments/runs/current"


def default_weights() -> str:
    return os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)


class ScratchFixer:
    name = "scratch"

    def __init__(
        self,
        cfg: ScratchConfig,
        words: WordVocab,
        chars: CharVocab,
        net: GapTagger,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.words = words
        self.chars = chars
        self.net = net.to(device).eval()
        self.device = device
        self.budget = cfg.budget

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        if len(tokens) < 2:
            return []
        enc = encode_window(tokens, current, self.words, self.chars, self.cfg.max_chars)
        batch = collate([enc], self.device)
        with torch.no_grad():
            logits = self.net(batch)
        ids = logits[0, : len(current)].argmax(dim=-1).tolist()
        return [Gap(int(i)) for i in ids]

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.save(run_dir / "config.json")
        self.words.save(run_dir / "words.json")
        self.chars.save(run_dir / "chars.json")
        torch.save(self.net.state_dict(), run_dir / "model.pt")

    @classmethod
    def untrained(
        cls,
        cfg: ScratchConfig,
        words: WordVocab,
        chars: CharVocab,
        seed: int,
        device: torch.device | None = None,
    ) -> ScratchFixer:
        torch.manual_seed(seed)
        net = GapTagger(cfg, len(words), len(chars))
        return cls(cfg, words, chars, net, device or select_device())

    @classmethod
    def load(cls, source: str | Path, device: torch.device | None = None) -> ScratchFixer:
        run_dir = resolve_weights(source)
        if not (run_dir / "model.pt").exists():
            raise FileNotFoundError(
                f"no weights at {run_dir}; train with scripts/train_scratch.py or set {WEIGHTS_ENV}"
            )
        device = device or select_device()
        cfg = ScratchConfig.load(run_dir / "config.json")
        words = WordVocab.load(run_dir / "words.json")
        chars = CharVocab.load(run_dir / "chars.json")
        net = GapTagger(cfg, len(words), len(chars))
        net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        return cls(cfg, words, chars, net, device)


def resolve_weights(source: str | Path) -> Path:
    """A local directory. Task 8 extends this to Hub sources of the form hf:repo@revision."""
    return Path(source)
```

- [ ] **Step 4: Register it** — in `src/newline_fixer/models/registry.py` add, next to `_rules`:

```python
def _scratch() -> Fixer:
    from .scratch import ScratchFixer, default_weights

    return ScratchFixer.load(default_weights())
```

and `"scratch": _scratch` in `_REGISTRY`. The import stays inside the function so `registry` does not import torch unless `scratch` is requested.

- [ ] **Step 5: Run `make fmt && make check`** — expected: all pass; the hypothesis test builds a fresh tiny model per example, which is a few seconds in total.

- [ ] **Step 6: Commit**

```bash
git add src/newline_fixer/models/scratch.py src/newline_fixer/models/registry.py tests/test_scratch_fixer.py
git commit -m "feat: ScratchFixer behind the Fixer protocol; save, load and registry entry"
```

---

### Task 5: Training examples from the clean split

**Files:**
- Create: `src/newline_fixer/models/train_data.py`, `tests/test_train_data.py`

**Interfaces:**
- Consumes: `corrupt.corrupt`, `corrupt.CorruptConfig`, `text.split`, `text.derive_labels`, `data.records.CleanDoc`, vocabularies.
- Produces:
  - `@dataclass(frozen=True) Example(tokens: list[str], current: list[Gap], target: list[Gap])`
  - `examples_from_doc(doc: CleanDoc, rng: random.Random, budget: int, cfg: CorruptConfig | None = None) -> list[Example]` (corrupt once, cut into chunks of at most `budget` tokens on gap boundaries, drop chunks with fewer than two tokens)
  - `epoch_examples(docs: Sequence[CleanDoc], seed: int, epoch: int, budget: int) -> list[Example]` (per-document rng `random.Random(f"{seed}:{epoch}:{doc.id}")`, order shuffled by `random.Random(f"{seed}:{epoch}")`)
  - `class_counts(examples: Iterable[Example]) -> list[int]` (four counts over targets)
  - `batches(examples: Sequence[Example], words, chars, max_chars, batch_size, device) -> Iterator[tuple[Batch, Tensor]]` where the target tensor is `[B, N-1]` long with `-100` in padding.

- [ ] **Step 1: Write the failing tests**

```python
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
    exs = [Example(["a", "b", "c"], [Gap.SPACE, Gap.NL], [Gap.SPACE, Gap.PARA]), Example(["d", "e"], [Gap.NL], [Gap.JOIN])]
    assert class_counts(exs) == [1, 1, 0, 1]
    wv, cv = WordVocab.build("a b c d e".split(), 10), CharVocab.build("abcde", 1)
    out = list(batches(exs, wv, cv, max_chars=3, batch_size=2, device=torch.device("cpu")))
    assert len(out) == 1
    batch, target = out[0]
    assert batch.word.shape == (2, 3) and target.tolist() == [[1, 3], [0, -100]]


def test_corrupt_config_is_honoured() -> None:
    exs = examples_from_doc(DOCS[1], random.Random(1), budget=400, cfg=CorruptConfig(clean_fraction=1.0))
    assert all(e.current == e.target for e in exs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_data.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/models/train_data.py`**

```python
"""Training examples: corrupted windows of the clean training split (design 3.4)."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..corrupt import CorruptConfig, corrupt
from ..data.records import CleanDoc
from ..text import Gap, derive_labels, split
from .vocab import Batch, CharVocab, WordVocab, collate, encode_window

IGNORE = -100


@dataclass(frozen=True)
class Example:
    tokens: list[str]
    current: list[Gap]
    target: list[Gap]


def examples_from_doc(
    doc: CleanDoc, rng: random.Random, budget: int, cfg: CorruptConfig | None = None
) -> list[Example]:
    corrupted = corrupt(doc.clean, rng, cfg)
    tokens, current = split(corrupted.text)
    target = derive_labels(corrupted.text, doc.clean)
    out: list[Example] = []
    for start in range(0, len(tokens), budget):
        chunk = tokens[start : start + budget]
        if len(chunk) < 2:
            continue
        gaps = slice(start, start + len(chunk) - 1)
        out.append(Example(list(chunk), list(current[gaps]), list(target[gaps])))
    return out


def epoch_examples(docs: Sequence[CleanDoc], seed: int, epoch: int, budget: int) -> list[Example]:
    out: list[Example] = []
    for doc in docs:
        out.extend(examples_from_doc(doc, random.Random(f"{seed}:{epoch}:{doc.id}"), budget))
    random.Random(f"{seed}:{epoch}").shuffle(out)
    return out


def class_counts(examples: Iterable[Example]) -> list[int]:
    counts = [0, 0, 0, 0]
    for e in examples:
        for g in e.target:
            counts[int(g)] += 1
    return counts


def batches(
    examples: Sequence[Example],
    words: WordVocab,
    chars: CharVocab,
    max_chars: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[Batch, Tensor]]:
    for i in range(0, len(examples), batch_size):
        group = examples[i : i + batch_size]
        encoded = [encode_window(e.tokens, e.current, words, chars, max_chars) for e in group]
        batch = collate(encoded, device)
        n = batch.word.shape[1]
        target = torch.full((len(group), n - 1), IGNORE, dtype=torch.long)
        for j, e in enumerate(group):
            target[j, : len(e.target)] = torch.tensor([int(g) for g in e.target], dtype=torch.long)
        yield batch, target.to(device)
```

- [ ] **Step 4: Run `make fmt && make check`** — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/newline_fixer/models/train_data.py tests/test_train_data.py
git commit -m "feat: seeded per-epoch training examples from the clean split"
```

---

### Task 6: Trainer, training script, run records

**Files:**
- Create: `src/newline_fixer/models/trainer.py`, `scripts/train_scratch.py`, `tests/test_trainer.py`
- Modify: `scripts/results_table.py` (render training runs), `src/newline_fixer/eval/table.py` (add `render_training_table`)

**Interfaces:**
- Consumes: everything above plus `eval.runner.evaluate_set`, `data.records.read_jsonl`, `EvalItem`.
- Produces:
  - `@dataclass(frozen=True) TrainConfig(run_id: str, epochs: int = 8, batch_size: int = 32, lr: float = 2e-3, weight_decay: float = 0.01, patience: int = 2, seed: int = 1, class_weights: str = "none", grad_clip: float = 1.0, max_docs: int | None = None, char_min_count: int = 5)`; `class_weights` is `"none"` or `"inverse"`
  - `class_weight_tensor(counts: Sequence[int], mode: str, device) -> Tensor | None` (inverse: `total / (4 * count_c)` normalized to mean 1, zero-count classes get weight 1)
  - `train(cfg: ScratchConfig, tcfg: TrainConfig, train_docs: Sequence[CleanDoc], dev: dict[str, list[EvalItem]], run_dir: Path, device: torch.device) -> dict[str, object]` returning the run record (also written to `run_dir / "run.json"`); the best epoch by V1 macro-F1 is saved to `run_dir` via `ScratchFixer.save`
  - `render_training_table(records: Sequence[dict[str, object]]) -> str`
  - CLI `uv run python scripts/train_scratch.py --run-id <id> [--epochs 8 --batch 32 --lr 2e-3 --seed 1 --class-weights none|inverse --max-docs N --device cpu|mps]`, which reads `data/clean/train.jsonl`, `data/sets/V1.jsonl`, `data/sets/V3.jsonl`, writes weights to `experiments/runs/<run-id>/` and the record to `experiments/training/<run-id>.json`.

- [ ] **Step 1: Write the failing tests**

```python
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
    "V1": [EvalItem("v1", "t", "Title Line The quick\nbrown fox.", "Title Line\n\nThe quick brown fox.", 0.5, {})],
    "V3": [EvalItem("v3", "t", "Clean text\n\nstays clean.", "Clean text\n\nstays clean.", 0.0, {})],
}


def test_class_weight_tensor() -> None:
    assert class_weight_tensor([10, 70, 15, 5], "none", CPU) is None
    w = class_weight_tensor([10, 70, 15, 0], "inverse", CPU)
    assert w is not None and w.shape == (4,)
    assert w[1] < w[0] < w[2] and abs(float(w.mean()) - 1.0) < 1e-6


def test_train_tiny_end_to_end(tmp_path: Path) -> None:
    tcfg = TrainConfig(run_id="t", epochs=3, batch_size=4, lr=1e-2, patience=5, seed=1, char_min_count=1)
    rec = train(TINY, tcfg, DOCS, DEV, tmp_path, CPU)
    epochs = rec["epochs"]
    assert isinstance(epochs, list) and len(epochs) == 3
    first, last = epochs[0], epochs[-1]
    assert isinstance(first, dict) and isinstance(last, dict)
    assert last["train_loss"] < first["train_loss"]
    for k in ("run_id", "git_commit", "seed", "config", "train_config", "n_train_docs", "device", "best_epoch", "best", "weights_dir"):
        assert k in rec
    assert set(first) >= {"epoch", "train_loss", "seconds", "V1_macro_f1", "V3_damage"}
    assert (tmp_path / "model.pt").exists() and (tmp_path / "run.json").exists()
    fx = ScratchFixer.load(tmp_path, CPU)
    assert len(fx.predict(["a", "b", "c"], [Gap.SPACE, Gap.NL])) == 2


def test_early_stopping_stops_after_patience(tmp_path: Path) -> None:
    tcfg = TrainConfig(run_id="e", epochs=8, batch_size=4, lr=0.0, patience=1, seed=1, char_min_count=1)
    rec = train(TINY, tcfg, DOCS, DEV, tmp_path, CPU)
    epochs = rec["epochs"]
    assert isinstance(epochs, list) and len(epochs) == 2  # epoch 1 is best, epoch 2 no better, stop


def test_render_training_table_has_one_row_per_run() -> None:
    recs: list[dict[str, object]] = [
        {"run_id": "a", "git_commit": "abc123def456", "train_config": {"class_weights": "none"}, "best_epoch": 2,
         "best": {"V1_macro_f1": 0.7, "V3_damage": 0.001, "V2_macro_f1": None}, "epochs": [{}, {}], "n_params": 5_000_000, "device": "mps", "seconds": 120.0},
    ]
    md = render_training_table(recs)
    assert "| a |" in md and "0.700" in md and "0.0010" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trainer.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/models/trainer.py`**

```python
"""Training loop for the from-scratch model (design 4.4, 4.7)."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ..data.records import CleanDoc, EvalItem
from ..eval.runner import evaluate_set
from ..text import split
from .scratch import ScratchFixer
from .scratch_config import ScratchConfig
from .scratch_net import GapTagger, count_parameters
from .train_data import IGNORE, batches, class_counts, epoch_examples
from .vocab import CharVocab, WordVocab


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    epochs: int = 8
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.01
    patience: int = 2
    seed: int = 1
    class_weights: str = "none"
    grad_clip: float = 1.0
    max_docs: int | None = None
    char_min_count: int = 5


def class_weight_tensor(counts: Sequence[int], mode: str, device: torch.device) -> Tensor | None:
    if mode == "none":
        return None
    if mode != "inverse":
        raise ValueError(f"unknown class weighting {mode!r}; use 'none' or 'inverse'")
    total = sum(counts)
    raw = [total / (len(counts) * c) if c else 1.0 for c in counts]
    mean = sum(raw) / len(raw)
    return torch.tensor([r / mean for r in raw], dtype=torch.float32, device=device)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _dev_metrics(fixer: ScratchFixer, dev: dict[str, list[EvalItem]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, items in dev.items():
        r = evaluate_set(fixer, items)
        gap = r["gap"]
        assert isinstance(gap, dict)
        out[f"{name}_macro_f1"] = float(gap["macro_f1"])
        out[f"{name}_damage"] = float(gap["damage_rate"])
        out[f"{name}_wrong_join_per_1000"] = float(gap["wrong_join_per_1000"])
    return out


def train(
    cfg: ScratchConfig,
    tcfg: TrainConfig,
    train_docs: Sequence[CleanDoc],
    dev: dict[str, list[EvalItem]],
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    docs = list(train_docs)[: tcfg.max_docs] if tcfg.max_docs else list(train_docs)
    tokens = [t for d in docs for t in split(d.clean)[0]]
    words = WordVocab.build(tokens, cfg.vocab_size)
    chars = CharVocab.build(tokens, tcfg.char_min_count)
    torch.manual_seed(tcfg.seed)
    net = GapTagger(cfg, len(words), len(chars)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    first_epoch = epoch_examples(docs, tcfg.seed, 0, cfg.budget)
    weights = class_weight_tensor(class_counts(first_epoch), tcfg.class_weights, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)

    record: dict[str, object] = {
        "run_id": tcfg.run_id,
        "git_commit": _git_commit(),
        "started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": tcfg.seed,
        "config": asdict(cfg),
        "train_config": asdict(tcfg),
        "n_train_docs": len(docs),
        "n_examples_epoch0": len(first_epoch),
        "class_counts_epoch0": class_counts(first_epoch),
        "n_words": len(words),
        "n_chars": len(chars),
        "n_params": count_parameters(net),
        "device": device.type,
        "epochs": [],
    }
    epochs: list[dict[str, object]] = []
    best_f1, best_epoch, bad, start_all = -1.0, 0, 0, time.time()
    for epoch in range(tcfg.epochs):
        examples = first_epoch if epoch == 0 else epoch_examples(docs, tcfg.seed, epoch, cfg.budget)
        net.train()
        t0, total, steps = time.time(), 0.0, 0
        for batch, target in batches(examples, words, chars, cfg.max_chars, tcfg.batch_size, device):
            optimizer.zero_grad()
            logits = net(batch)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), tcfg.grad_clip)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        net.eval()
        fixer = ScratchFixer(cfg, words, chars, net, device)
        metrics = _dev_metrics(fixer, dev)
        row: dict[str, object] = {"epoch": epoch + 1, "train_loss": total / max(steps, 1), "seconds": time.time() - t0, **metrics}
        epochs.append(row)
        print(json.dumps(row))
        f1 = metrics.get("V1_macro_f1", -1.0)
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch + 1, 0
            fixer.save(run_dir)
            record["best"] = dict(metrics)
        else:
            bad += 1
            if bad >= tcfg.patience:
                break
    record["epochs"] = epochs
    record["best_epoch"] = best_epoch
    record["seconds"] = time.time() - start_all
    record["weights_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
```

`lr=0.0` in the early-stopping test makes every epoch identical, so epoch 1 is best and epoch 2 trips `patience=1`.

- [ ] **Step 4: Add `render_training_table` to `src/newline_fixer/eval/table.py`**

```python
def render_training_table(records: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| run | commit | class weights | best epoch / run | V1 macro-F1 | V2 macro-F1 | V3 damage | params | device | minutes |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in records:
        best = r.get("best", {})
        v2 = best.get("V2_macro_f1")
        lines.append(
            f"| {r['run_id']} | `{str(r.get('git_commit', ''))[:12]}` | {r.get('train_config', {}).get('class_weights', '')} | "
            f"{r.get('best_epoch', '')} / {len(r.get('epochs', []))} | {best.get('V1_macro_f1', 0.0):.3f} | "
            f"{'' if v2 is None else f'{v2:.3f}'} | {best.get('V3_damage', 0.0):.4f} | {int(r.get('n_params', 0)):,} | "
            f"{r.get('device', '')} | {float(r.get('seconds', 0.0)) / 60:.1f} |"
        )
    return "\n".join(lines) + "\n"
```

(add `from collections.abc import Sequence` to the imports of `table.py`).

- [ ] **Step 5: Write `scripts/train_scratch.py`**

```python
"""Train the from-scratch model. Usage:
  uv run python scripts/train_scratch.py --run-id scratch-v1 [--epochs 8 --batch 32 --lr 2e-3 --seed 1
      --class-weights none|inverse --max-docs N --device cpu|mps]
Writes weights to experiments/runs/<run-id>/ (git-ignored) and the record to
experiments/training/<run-id>.json (committed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newline_fixer.data.records import CleanDoc, EvalItem, read_jsonl
from newline_fixer.models.device import select_device
from newline_fixer.models.scratch_config import ScratchConfig
from newline_fixer.models.trainer import TrainConfig, train

CLEAN = Path("data/clean")
SETS = Path("data/sets")
RUNS = Path("experiments/runs")
RECORDS = Path("experiments/training")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--class-weights", default="none", choices=["none", "inverse"])
    p.add_argument("--max-docs", type=int)
    p.add_argument("--device")
    a = p.parse_args()
    tcfg = TrainConfig(
        run_id=a.run_id, epochs=a.epochs, batch_size=a.batch, lr=a.lr, seed=a.seed,
        class_weights=a.class_weights, max_docs=a.max_docs,
    )
    dev = {s: read_jsonl(SETS / f"{s}.jsonl", EvalItem) for s in ("V1", "V3")}
    docs = read_jsonl(CLEAN / "train.jsonl", CleanDoc)
    device = select_device(a.device)
    print(f"training {a.run_id} on {device} with {len(docs)} documents")
    record = train(ScratchConfig(), tcfg, docs, dev, RUNS / a.run_id, device)
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"{a.run_id}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch {record['best_epoch']}: {record['best']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Extend `scripts/results_table.py`** so `experiments/README.md` has a "Training runs" section after the result tables:

```python
    training = sorted(Path("experiments/training").glob("*.json"))
    if training:
        parts.append("\n## Training runs\n\n" + render_training_table([json.loads(p.read_text()) for p in training]))
```

(import `render_training_table` next to `render_table`; place this before the final `write_text`).

- [ ] **Step 7: Run `make fmt && make check`** — expected: all pass. The tiny end-to-end training test takes a few seconds on CPU.

- [ ] **Step 8: Smoke-run the script on a slice** (writes under git-ignored `experiments/runs/`; delete the record afterwards):

```bash
uv run python scripts/train_scratch.py --run-id smoke --epochs 1 --max-docs 50 --device cpu
rm experiments/training/smoke.json
```

Expected: one JSON line for the epoch with `V1_macro_f1` and `V3_damage`, then "best epoch 1".

- [ ] **Step 9: Commit**

```bash
git add src/newline_fixer/models/trainer.py src/newline_fixer/eval/table.py scripts/train_scratch.py scripts/results_table.py tests/test_trainer.py
git commit -m "feat: training loop with early stopping on V1 macro-F1, run records and training table"
```

---

### Task 7: Train the model, compare class weighting, record the decision

**Files:**
- Produce (committed): `experiments/training/scratch-v1.json`, `experiments/training/scratch-v1-inverse.json`, `experiments/results/m2-scratch.json`, `experiments/README.md`
- Create: `docs/decisions/0007-scratch-model-class-weighting.md`
- Modify: `README.md` (how to train and evaluate the scratch model), `data/README.md` (no change unless the training reveals a data problem)

**Interfaces:**
- Consumes: everything above; `scripts/evaluate.py --systems identity,rules,scratch`.
- Produces: the run that `experiments/runs/current` points at (a symlink or a copy of the chosen run directory; git-ignored).

- [ ] **Step 1: Train the unweighted run**

```bash
uv run python scripts/train_scratch.py --run-id scratch-v1 --seed 1 --class-weights none
```

Expected: up to eight epoch lines; each epoch about one to two minutes on MPS (around 300 steps at batch 32 plus the V1 and V3 evaluation). If MPS raises on `pack_padded_sequence` or the LSTM, re-run with `--device cpu` and record it; CPU is roughly three times slower. Note `V1_macro_f1` and `V3_damage` per epoch.

- [ ] **Step 2: Train the class-weighted run**

```bash
uv run python scripts/train_scratch.py --run-id scratch-v1-inverse --seed 1 --class-weights inverse
```

- [ ] **Step 3: Choose the run**

Pick by `best.V1_macro_f1`, subject to `best.V3_damage <= 0.0026` (the gate from decision 0006). If both pass the gate, the higher V1 macro-F1 wins; if only one passes, it wins; if neither passes, the one with lower V3 damage wins and the decision record says the gate is not yet met. Point the default weights at it:

```bash
rm -rf experiments/runs/current && cp -r experiments/runs/<chosen> experiments/runs/current
```

- [ ] **Step 4: Evaluate on the three dev sets and render**

```bash
uv run python scripts/evaluate.py --systems identity,rules,scratch --sets V1,V2,V3 --out experiments/results/m2-scratch.json
uv run python scripts/results_table.py
```

Read the table. Checks: scratch V3 damage against the gate 0.0026; scratch wrong-join per 1k on every set (B1 has 0.00; any wrong join is the worst error and must be discussed); scratch versus rules on V1 (0.635) and V2 (0.806) macro-F1 and on PARA F1; JOIN F1 on V1 (rules 0.682). Run the challenge example through the model and record whether it reproduces the expected output:

```bash
uv run python -c "
from newline_fixer.models.registry import get_fixer
from newline_fixer.windows import fix
from newline_fixer.example import EXAMPLE_INPUT, EXAMPLE_OUTPUT
from newline_fixer.text import normalize
out = fix(EXAMPLE_INPUT, get_fixer('scratch')).text
print(out == normalize(EXAMPLE_OUTPUT)); print(out)"
```

- [ ] **Step 5: Write `docs/decisions/0007-scratch-model-class-weighting.md`** (fill every number from the records and the table; the decision README forbids editing once accepted, so write it complete):

```markdown
# 0007. Class weighting for the from-scratch model; first trained model selected

Date: <date>. Status: accepted.

## Context

Design 4.4 fixes the architecture and optimizer of M1 and the risk table says class
weights are tried on V1 because JOIN and PARA are rare. Decision 0006 froze B1 at
macro-F1 0.635 (V1) and 0.806 (V2) with a clean-damage gate of 0.0026 on V3.

## Options

1. Unweighted cross-entropy (`scratch-v1`).
2. Inverse-frequency class weights normalized to mean one (`scratch-v1-inverse`).

## Decision

Option <n>: run `<run-id>` is the current from-scratch model. Numbers from
`experiments/training/*.json` and `experiments/results/m2-scratch.json`:

| run | best epoch | V1 macro-F1 | V1 JOIN F1 | V1 PARA F1 | V2 macro-F1 | V3 damage | wrong-join /1k (V1/V2/V3) |
|---|---|---|---|---|---|---|---|
| scratch-v1 | <n> | <n> | <n> | <n> | <n> | <n> | <n>/<n>/<n> |
| scratch-v1-inverse | <n> | <n> | <n> | <n> | <n> | <n> | <n>/<n>/<n> |
| rules (0006) | | 0.635 | 0.682 | 0.410 | 0.806 | 0.0026 | 0.00/0.00/0.00 |

<One paragraph: which run passes the gate, by how much the winner beats or trails B1 on
V1 and V2, where it loses (PARA, JOIN, wrong joins), and the example result.>

Training facts: <device>, <minutes per run>, <parameters>, <examples per epoch>, seed 1,
class counts in epoch 0 <counts>.

## Consequences

- `experiments/runs/current` and `NF_WEIGHTS` default to `<run-id>`; M3 serves it if it
  passes design 5.3 (V3 gate and latency), otherwise B1.
- <What the loss profile suggests for M5 and for any M1 follow-up, in one or two lines.>
- The weights are published in Task 8; this record is updated by a new record, never edited.
```

- [ ] **Step 6: README** — under Development add a "Train the from-scratch model" paragraph with the two commands from steps 1 and 4 and the sentence "Weights are written to `experiments/runs/<run-id>/` (git-ignored); set `NF_WEIGHTS` to a run directory to evaluate or serve it."

- [ ] **Step 7: `make check`, then commit** (records and results only; never `experiments/runs/`):

```bash
git status --short   # must list no file under experiments/runs/
git add experiments/training experiments/results/m2-scratch.json experiments/README.md docs/decisions/0007-scratch-model-class-weighting.md README.md
git commit -m "data: first from-scratch model trained; class-weighting decision 0007; M2 dev results"
```

---

### Task 8: Publish weights to the Hub and load them from there

**Files:**
- Create: `scripts/publish_weights.py`, `tests/test_weights_source.py`
- Modify: `src/newline_fixer/models/scratch.py` (`resolve_weights` learns `hf:` sources), `README.md` (one paragraph), `docs/decisions/0007-...` is NOT edited; the published revision goes in `README.md` and in `experiments/training/<run-id>.json` under a new key `hub` (allowed: the record is data, not a decision).

**Interfaces:**
- Produces: `parse_weights_source(source: str) -> tuple[str, str, str | None]` returning `("dir", path, None)` or `("hf", repo_id, revision)`; `resolve_weights` downloads `hf:` sources with `huggingface_hub.snapshot_download(repo_id, revision=revision)` (lazy import) and returns the local directory; CLI `uv run python scripts/publish_weights.py --run-id <id> --repo <user>/newline-fixer-scratch` which uploads the run directory with a generated `README.md` model card and prints the commit revision.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from newline_fixer.models.scratch import parse_weights_source, resolve_weights


def test_parse_weights_source() -> None:
    assert parse_weights_source("experiments/runs/current") == ("dir", "experiments/runs/current", None)
    assert parse_weights_source("hf:user/repo") == ("hf", "user/repo", None)
    assert parse_weights_source("hf:user/repo@abc123") == ("hf", "user/repo", "abc123")
    with pytest.raises(ValueError):
        parse_weights_source("hf:")


def test_resolve_dir_source_is_path(tmp_path: Path) -> None:
    assert resolve_weights(str(tmp_path)) == tmp_path
    assert resolve_weights(tmp_path) == tmp_path


@pytest.mark.skipif(not Path("experiments/runs/current/model.pt").exists(), reason="no trained weights")
def test_current_weights_load_and_fix_example() -> None:
    from newline_fixer.example import EXAMPLE_INPUT
    from newline_fixer.models.scratch import ScratchFixer
    from newline_fixer.text import content
    from newline_fixer.windows import fix

    fx = ScratchFixer.load("experiments/runs/current")
    assert content(fix(EXAMPLE_INPUT, fx).text) == content(EXAMPLE_INPUT)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_weights_source.py`
Expected: FAIL with `ImportError: cannot import name 'parse_weights_source'`

- [ ] **Step 3: Replace `resolve_weights` in `src/newline_fixer/models/scratch.py`**

```python
def parse_weights_source(source: str) -> tuple[str, str, str | None]:
    """'hf:repo[@revision]' is a Hub model repo; anything else is a local directory."""
    if not source.startswith("hf:"):
        return "dir", source, None
    spec = source[3:]
    if not spec:
        raise ValueError("hf: source needs a repo id, for example hf:user/newline-fixer-scratch@<revision>")
    repo, _, revision = spec.partition("@")
    return "hf", repo, revision or None


def resolve_weights(source: str | Path) -> Path:
    if isinstance(source, Path):
        return source
    kind, value, revision = parse_weights_source(source)
    if kind == "dir":
        return Path(value)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=value, revision=revision))
```

- [ ] **Step 4: Write `scripts/publish_weights.py`**

```python
"""Publish a trained run to the Hugging Face Hub (decision 0003). Usage:
  huggingface-cli login   # once
  uv run python scripts/publish_weights.py --run-id scratch-v1 --repo <user>/newline-fixer-scratch
Prints the commit revision to pin with NF_WEIGHTS=hf:<repo>@<revision>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("experiments/runs")
RECORDS = Path("experiments/training")


def model_card(record: dict[str, object], repo: str) -> str:
    best = record.get("best", {})
    assert isinstance(best, dict)
    return (
        "---\nlibrary_name: pytorch\nlicense: mit\ntags: [text-cleaning, newline-restoration]\n---\n\n"
        f"# newline-fixer from-scratch model ({record['run_id']})\n\n"
        "A character-aware BiLSTM that predicts the whitespace class (join, space, newline, paragraph) "
        "between consecutive tokens of English text. Trained with https://github.com/jalalhussein1982/newline-fixer "
        f"at commit `{str(record.get('git_commit', ''))[:12]}`.\n\n"
        f"- Dev macro-F1 V1 {best.get('V1_macro_f1', 0.0):.3f}, clean-text damage V3 {best.get('V3_damage', 0.0):.4f}\n"
        f"- Parameters {int(record.get('n_params', 0)):,}; seed {record.get('seed')}; best epoch {record.get('best_epoch')}\n\n"
        f"Load with `ScratchFixer.load('hf:{repo}@<revision>')`; files: config.json, words.json, chars.json, model.pt, run.json.\n"
    )


def main() -> None:
    from huggingface_hub import HfApi

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--repo", required=True)
    a = p.parse_args()
    run_dir = RUNS / a.run_id
    record_path = RECORDS / f"{a.run_id}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    (run_dir / "README.md").write_text(model_card(record, a.repo), encoding="utf-8")
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    info = api.upload_folder(
        folder_path=str(run_dir), repo_id=a.repo, repo_type="model",
        allow_patterns=["config.json", "words.json", "chars.json", "model.pt", "run.json", "README.md"],
        commit_message=f"{a.run_id} from {str(record.get('git_commit', ''))[:12]}",
    )
    record["hub"] = {"repo": a.repo, "revision": info.oid}
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"published https://huggingface.co/{a.repo} revision {info.oid}")
    print(f"pin with NF_WEIGHTS=hf:{a.repo}@{info.oid}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: `make fmt && make check`** — expected: all pass (the Hub test is skipped unless `experiments/runs/current` exists locally, where it runs against the real weights).

- [ ] **Step 6: Publish** (needs a Hugging Face login; the author runs it):

```bash
huggingface-cli login
uv run python scripts/publish_weights.py --run-id <chosen run> --repo <hf-user>/newline-fixer-scratch
```

Record the printed revision in `README.md` ("Weights" paragraph: repo, revision, the `NF_WEIGHTS=hf:...@...` line) and confirm a clean load:

```bash
NF_WEIGHTS=hf:<hf-user>/newline-fixer-scratch@<revision> uv run python -c "from newline_fixer.models.registry import get_fixer; print(get_fixer('scratch').name)"
```

If the login is not available when this task runs, complete steps 1 to 5, write the README paragraph with "pending" in place of the revision, and list the two commands under a "Pending" heading; the task is then complete except for the upload, which the author finishes.

- [ ] **Step 7: Commit**

```bash
git add src/newline_fixer/models/scratch.py scripts/publish_weights.py tests/test_weights_source.py README.md experiments/training
git commit -m "feat: load weights from the Hub by pinned revision; publish script with model card"
```

M2 is complete when this commit is on `main`, `experiments/README.md` shows the training runs and the scratch model on V1, V2, V3 next to identity and rules, decision 0007 is accepted with real numbers, and the chosen weights load from `experiments/runs/current` (and from the Hub once published).
