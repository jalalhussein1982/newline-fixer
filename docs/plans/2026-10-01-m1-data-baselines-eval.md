# M1: Data, Baselines and Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline, the two non-learned baselines and the evaluation harness, and produce the first results table on the development sets.

**Architecture:** A pure-Python library under `src/newline_fixer/` where every system implements one `Fixer` protocol over tokens and gap classes. Text is split into non-whitespace tokens and four-way gap labels, corrupted by a seeded generator, and evaluated with gap-level metrics. Scripts under `scripts/` drive data building and evaluation and write committed artifacts under `data/sets/` and `experiments/`.

**Tech Stack:** Python 3.12, uv, pytest, hypothesis, ruff, mypy. Data extras: `datasets`, `huggingface_hub`, `anthropic`. No PyTorch in this milestone.

**Spec:** `docs/02-design.md` (sections 2, 3, 4.1 to 4.3, 5, 7, 8). Requirements: `docs/01-requirements.md`.

## Global Constraints

- Python `>=3.12`. Package name `newline_fixer`, layout `src/newline_fixer/`.
- Gap classes are exactly `JOIN`, `SPACE`, `NL`, `PARA` with output strings `""`, `" "`, `"\n"`, `"\n\n"` (design 2.1).
- Every system preserves the non-whitespace character sequence (design 2.1). Reconstruction enforces it.
- Output is canonical whitespace: no leading or trailing whitespace, every gap one of the four strings (design 2.1).
- Every random process takes an explicit `random.Random` or seed. No global random state.
- Dev sets V1, V2, V3 are used for every choice. Test sets T0, T1, T2, T3 are not evaluated in this milestone except T0 as a unit test (design 5.1).
- `make check` passes at every commit. Commit messages follow `type: summary` with `docs`, `feat`, `test`, `chore`, `data`.
- Nothing under `data/` is committed except `data/README.md`, `data/split.json`, `data/sets/`, `data/realistic/`.
- Commits end with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

Inputs the spec implies but that no section spells out, each pinned by a test in the task that owns the code:

1. **Windows line endings** (`\r\n`) must count as one newline, not two or zero. Test in Task 2.
2. **Non-breaking spaces and tabs** inside a line must be treated as a space gap, not as content. Test in Task 2.
3. **Empty, whitespace-only and single-token input** must return `""` or the single token without error from `fix()`. Test in Task 5.
4. **One very long token** (a 2,000-character URL) must still be covered by windowing with the invariant intact. Test in Task 5.
5. **Non-ASCII text** (accents, CJK, emoji) must survive the corruptor and `fix()` with the invariant intact. Property tests in Tasks 4 and 5.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `Makefile`, `src/newline_fixer/__init__.py`, `src/newline_fixer/py.typed`, `tests/__init__.py`, `tests/test_package.py`, `data/README.md`
- Modify: `.gitignore`, `README.md`

**Interfaces:**
- Produces: `newline_fixer.__version__: str`; `make check`, `make test`, `make lint`, `make type`, `make fmt`.

- [ ] **Step 1: Write the smoke test**

`tests/__init__.py` is empty. `tests/test_package.py`:

```python
import newline_fixer


def test_version_is_a_string() -> None:
    assert isinstance(newline_fixer.__version__, str)
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "newline-fixer"
version = "0.1.0"
description = "A service that fixes newline placement in English text"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
data = ["datasets>=3.0", "huggingface_hub>=0.25", "anthropic>=0.40"]

[dependency-groups]
dev = ["pytest>=8", "hypothesis>=6", "ruff>=0.6", "mypy>=1.11"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/newline_fixer"]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests", "scripts"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]  # the formatter owns line layout; long strings are allowed

[tool.mypy]
strict = true
mypy_path = "src"
packages = ["newline_fixer"]
files = ["src", "tests", "scripts"]

[[tool.mypy.overrides]]
module = ["datasets", "datasets.*", "huggingface_hub", "huggingface_hub.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Write the package init and `py.typed`**

`src/newline_fixer/__init__.py`:

```python
"""newline-fixer: a service that fixes newline placement in English text."""

__version__ = "0.1.0"
```

`src/newline_fixer/py.typed` is an empty file.

- [ ] **Step 4: Write the Makefile** (recipes are indented with a tab, not spaces)

```make
.PHONY: check lint type test fmt sync

sync:
	uv sync --all-extras

check: lint type test

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

type:
	uv run mypy

test:
	uv run pytest

fmt:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts
```

Create an empty `scripts/__init__.py` so ruff and mypy have a target.

- [ ] **Step 5: Extend `.gitignore` and add `data/README.md`**

Replace the `data/*` lines in `.gitignore` with:

```
data/*
!data/README.md
!data/split.json
!data/sets/
!data/realistic/
```

`data/README.md`:

```markdown
# data

Only these are committed: this file, `split.json` (group ids per split), `sets/`
(evaluation sets as JSONL) and `realistic/` (raw and adjusted real passages with
reviewed targets). Everything else is built by `scripts/build_data.py` and published to
the Hugging Face Hub with a manifest of content hashes. See `docs/02-design.md` section 3.
```

- [ ] **Step 6: Install and run**

Run: `uv sync --all-extras && make check`
Expected: ruff, mypy and pytest all pass; one test collected.

- [ ] **Step 7: Update `README.md` layout table** by adding the rows `src/newline_fixer/ | library`, `scripts/ | data building and evaluation entry points`, `tests/ | pytest suite`, `data/ | see data/README.md`, and a "Development" section:

```markdown
## Development

```bash
uv sync --all-extras   # creates .venv with all dependencies
make check             # lint, type check, tests
```
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: project scaffold with uv, ruff, mypy, pytest"
```

---

### Task 2: Text core: gap classes, split, join, normalize

**Files:**
- Create: `src/newline_fixer/text.py`, `tests/test_text.py`

**Interfaces:**
- Produces:
  - `class Gap(IntEnum)`: `JOIN=0, SPACE=1, NL=2, PARA=3`
  - `GAP_STR: dict[Gap, str]`
  - `classify_ws(ws: str) -> Gap`
  - `split(text: str) -> tuple[list[str], list[Gap]]` (gaps has `len(tokens)-1` entries, or 0)
  - `join(tokens: Sequence[str], gaps: Sequence[Gap]) -> str`
  - `normalize(text: str) -> str`
  - `content(text: str) -> str` (all whitespace removed)

- [ ] **Step 1: Write the failing tests**

```python
from hypothesis import given
from hypothesis import strategies as st

from newline_fixer.text import GAP_STR, Gap, classify_ws, content, join, normalize, split


def test_classify_ws_basic() -> None:
    assert classify_ws("") is Gap.JOIN
    assert classify_ws(" ") is Gap.SPACE
    assert classify_ws("   ") is Gap.SPACE
    assert classify_ws("\n") is Gap.NL
    assert classify_ws("\n ") is Gap.NL
    assert classify_ws(" \n  ") is Gap.NL
    assert classify_ws("\n\n") is Gap.PARA
    assert classify_ws("\n \n") is Gap.PARA
    assert classify_ws("\n\n\n\n") is Gap.PARA


def test_classify_ws_windows_line_endings() -> None:
    assert classify_ws("\r\n") is Gap.NL
    assert classify_ws("\r\n\r\n") is Gap.PARA
    assert classify_ws("\r") is Gap.NL


def test_classify_ws_tabs_and_nbsp_are_space() -> None:
    assert classify_ws("\t") is Gap.SPACE
    assert classify_ws(" ") is Gap.SPACE
    assert classify_ws(" \t ") is Gap.SPACE


def test_split_example() -> None:
    tokens, gaps = split("3.2.3 Applications of Attention\n in our Model")
    assert tokens == ["3.2.3", "Applications", "of", "Attention", "in", "our", "Model"]
    assert gaps == [Gap.SPACE, Gap.SPACE, Gap.SPACE, Gap.NL, Gap.SPACE, Gap.SPACE]


def test_split_strips_leading_and_trailing_whitespace() -> None:
    assert split("  a b \n") == (["a", "b"], [Gap.SPACE])


def test_split_empty_and_single() -> None:
    assert split("") == ([], [])
    assert split("   \n ") == ([], [])
    assert split("word") == (["word"], [])


def test_join_rebuilds_canonical_text() -> None:
    assert join(["a", "b", "c", "d"], [Gap.JOIN, Gap.NL, Gap.PARA]) == "ab\nc\n\nd"
    assert join([], []) == ""
    assert join(["x"], []) == "x"


def test_join_rejects_wrong_gap_count() -> None:
    import pytest

    with pytest.raises(ValueError):
        join(["a", "b"], [])


def test_normalize_is_idempotent_and_canonical() -> None:
    raw = "  Hello \t world \n\n\n again\r\nend  "
    assert normalize(raw) == "Hello world\n\nagain\nend"
    assert normalize(normalize(raw)) == normalize(raw)


def test_content_removes_all_whitespace() -> None:
    assert content(" a\tb\nc d ") == "abcd"


@given(st.text())
def test_split_join_preserves_content(text: str) -> None:
    tokens, gaps = split(text)
    assert content(join(tokens, gaps)) == content(text)


@given(st.text())
def test_normalize_output_is_canonical(text: str) -> None:
    out = normalize(text)
    assert out == out.strip()
    tokens, gaps = split(out)
    assert join(tokens, gaps) == out
    for g in gaps:
        assert GAP_STR[g] in ("", " ", "\n", "\n\n")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_text.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'newline_fixer.text'`

- [ ] **Step 3: Implement `src/newline_fixer/text.py`**

```python
"""Tokens, gap classes, normalization and reconstruction (design section 2.1)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import IntEnum


class Gap(IntEnum):
    """What stands between two consecutive tokens."""

    JOIN = 0
    SPACE = 1
    NL = 2
    PARA = 3


GAP_STR: dict[Gap, str] = {Gap.JOIN: "", Gap.SPACE: " ", Gap.NL: "\n", Gap.PARA: "\n\n"}

_TOKEN_RE = re.compile(r"\S+")
_WS_RE = re.compile(r"\s+")


def classify_ws(ws: str) -> Gap:
    """Map a whitespace run to its gap class. The empty string is JOIN."""
    if ws == "":
        return Gap.JOIN
    newlines = ws.replace("\r\n", "\n").replace("\r", "\n").count("\n")
    if newlines >= 2:
        return Gap.PARA
    if newlines == 1:
        return Gap.NL
    return Gap.SPACE


def split(text: str) -> tuple[list[str], list[Gap]]:
    """Split text into non-whitespace tokens and the gap class between each pair."""
    tokens: list[str] = []
    gaps: list[Gap] = []
    end = 0
    for m in _TOKEN_RE.finditer(text):
        if tokens:
            gaps.append(classify_ws(text[end : m.start()]))
        tokens.append(m.group())
        end = m.end()
    return tokens, gaps


def join(tokens: Sequence[str], gaps: Sequence[Gap]) -> str:
    """Rebuild canonical text from tokens and gap classes."""
    expected = max(len(tokens) - 1, 0)
    if len(gaps) != expected:
        raise ValueError(f"{len(tokens)} tokens need {expected} gaps, got {len(gaps)}")
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i < len(gaps):
            parts.append(GAP_STR[gaps[i]])
    return "".join(parts)


def normalize(text: str) -> str:
    """Canonical whitespace form of text: split then join."""
    tokens, gaps = split(text)
    return join(tokens, gaps)


def content(text: str) -> str:
    """The text with every whitespace character removed."""
    return _WS_RE.sub("", text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make check`
Expected: all pass, ruff and mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/newline_fixer/text.py tests/test_text.py
git commit -m "feat: gap classes, tokenization, canonical reconstruction"
```

---

### Task 3: Label derivation, reachability, the challenge example

**Files:**
- Modify: `src/newline_fixer/text.py`
- Create: `src/newline_fixer/example.py`, `tests/conftest.py`, `tests/test_labels.py`

**Interfaces:**
- Produces:
  - `gap_after_char(text: str) -> list[Gap]` (one entry per non-whitespace character)
  - `derive_labels(corrupted: str, clean: str) -> list[Gap]` (one per gap of `corrupted`)
  - `unreachable_count(source: str, target: str) -> int`
  - `newline_fixer.example.EXAMPLE_INPUT`, `EXAMPLE_OUTPUT` (the challenge example without its trailing `[...]` marker)
  - pytest fixtures `example_input: str`, `example_output: str`

- [ ] **Step 1: Write the example constants and the fixtures**

`src/newline_fixer/example.py`:

```python
"""The challenge example, used as test set T0 and in unit tests."""

EXAMPLE_INPUT = (
    "3.2.3 Applications of Attention\n"
    " in our Model The Transformer uses multi-head attention in three different ways: "
    '• In "encoder-decoder attention" layers,\n'
    " the que\n"
    "ries come from the previous decoder layer."
)

EXAMPLE_OUTPUT = (
    "3.2.3 Applications of Attention in our Model\n"
    "\n"
    "The Transformer uses multi-head attention in three different ways:\n"
    '• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.'
)
```

`tests/conftest.py`:

```python
import pytest

from newline_fixer.example import EXAMPLE_INPUT, EXAMPLE_OUTPUT


@pytest.fixture
def example_input() -> str:
    return EXAMPLE_INPUT


@pytest.fixture
def example_output() -> str:
    return EXAMPLE_OUTPUT
```

- [ ] **Step 2: Write the failing tests** in `tests/test_labels.py`

```python
import pytest
from hypothesis import given
from hypothesis import strategies as st

from newline_fixer.text import (
    Gap,
    derive_labels,
    gap_after_char,
    join,
    normalize,
    split,
    unreachable_count,
)


def test_gap_after_char() -> None:
    assert gap_after_char("ab c\nd") == [Gap.JOIN, Gap.SPACE, Gap.NL, Gap.JOIN]
    assert gap_after_char("") == []
    assert gap_after_char("x") == [Gap.JOIN]


def test_derive_labels_on_example(example_input: str, example_output: str) -> None:
    tokens, current = split(example_input)
    labels = derive_labels(example_input, example_output)
    assert len(labels) == len(current)
    assert join(tokens, labels) == normalize(example_output)
    by_pair = {(tokens[i], tokens[i + 1]): labels[i] for i in range(len(labels))}
    assert by_pair[("Attention", "in")] is Gap.SPACE
    assert by_pair[("Model", "The")] is Gap.PARA
    assert by_pair[("ways:", "•")] is Gap.NL
    assert by_pair[("layers,", "the")] is Gap.SPACE
    assert by_pair[("que", "ries")] is Gap.JOIN


def test_derive_labels_rejects_content_mismatch() -> None:
    with pytest.raises(ValueError):
        derive_labels("a b", "a c")


def test_unreachable_count() -> None:
    assert unreachable_count("a b", "a\nb") == 0
    assert unreachable_count("paragraph.Second", "paragraph.\n\nSecond") == 1
    assert unreachable_count("a\nb", "ab") == 0


@given(st.text(min_size=1))
def test_derive_labels_roundtrip_from_clean(clean: str) -> None:
    canon = normalize(clean)
    tokens, _ = split(canon)
    assert join(tokens, derive_labels(canon, canon)) == canon
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_labels.py`
Expected: FAIL with `ImportError: cannot import name 'derive_labels'`

- [ ] **Step 4: Append to `src/newline_fixer/text.py`**

```python
def gap_after_char(text: str) -> list[Gap]:
    """For each non-whitespace character, the gap class that follows it.

    Characters inside a token are followed by JOIN; the last character of the last
    token is followed by JOIN as well.
    """
    tokens, gaps = split(text)
    out: list[Gap] = []
    for i, tok in enumerate(tokens):
        out.extend([Gap.JOIN] * (len(tok) - 1))
        out.append(gaps[i] if i < len(gaps) else Gap.JOIN)
    return out


def derive_labels(corrupted: str, clean: str) -> list[Gap]:
    """Target gap class for every gap of `corrupted`, read off `clean` (design 2.2)."""
    if content(corrupted) != content(clean):
        raise ValueError("corrupted and clean text differ in non-whitespace content")
    after = gap_after_char(clean)
    tokens, _ = split(corrupted)
    labels: list[Gap] = []
    n = 0
    for tok in tokens[:-1]:
        n += len(tok)
        labels.append(after[n - 1])
    return labels


def unreachable_count(source: str, target: str) -> int:
    """Count target breaks that fall where `source` has no gap (design 2.3)."""
    if content(source) != content(target):
        raise ValueError("source and target differ in non-whitespace content")
    src = gap_after_char(source)
    tgt = gap_after_char(target)
    return sum(1 for s, t in zip(src, tgt, strict=True) if s is Gap.JOIN and t is not Gap.JOIN)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `make check`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/newline_fixer/text.py src/newline_fixer/example.py tests/conftest.py tests/test_labels.py
git commit -m "feat: exact label derivation and reachability check; example fixture"
```

---

### Task 4: Seeded corruptor

**Files:**
- Create: `src/newline_fixer/corrupt.py`, `tests/test_corrupt.py`

**Interfaces:**
- Consumes: `text.split`, `text.join`, `text.content`, `text.Gap`, `text.GAP_STR`
- Produces:
  - `@dataclass(frozen=True) CorruptConfig(clean_fraction=0.1, remove_base=0.6, remove_scale=0.4, insert_min_chars=40, insert_max_chars=200, word_split_prob=0.4, double_prob=0.1)`
  - `@dataclass(frozen=True) Corrupted(text: str, severity: float)`
  - `corrupt(clean: str, rng: random.Random, cfg: CorruptConfig | None = None) -> Corrupted`

- [ ] **Step 1: Write the failing tests**

```python
import random

from hypothesis import given, settings
from hypothesis import strategies as st

from newline_fixer.corrupt import CorruptConfig, Corrupted, corrupt
from newline_fixer.text import Gap, content, derive_labels, normalize, split

CLEAN = (
    "2.1 Background\n\n"
    "Recurrent models align positions to steps in computation time. They generate "
    "hidden states as a function of the previous state and the input.\n\n"
    "Three advantages:\n"
    "• total computational complexity per layer\n"
    "• the amount of computation that can be parallelized\n"
    "• the path length between long-range dependencies\n\n"
    "The Transformer uses attention in three different ways."
)


def test_same_seed_same_output() -> None:
    a = corrupt(CLEAN, random.Random(7))
    b = corrupt(CLEAN, random.Random(7))
    assert a == b


def test_content_preserved_and_severity_in_range() -> None:
    for seed in range(50):
        out = corrupt(CLEAN, random.Random(seed))
        assert isinstance(out, Corrupted)
        assert content(out.text) == content(CLEAN)
        assert 0.0 <= out.severity <= 1.0


def test_clean_fraction_one_returns_normalized_input() -> None:
    out = corrupt("  a  b\n\n\nc ", random.Random(1), CorruptConfig(clean_fraction=1.0))
    assert out == Corrupted("a b\n\nc", 0.0)


def test_short_input_is_untouched() -> None:
    assert corrupt("word", random.Random(1)) == Corrupted("word", 0.0)
    assert corrupt("", random.Random(1)) == Corrupted("", 0.0)


def test_produces_every_kind_of_noise() -> None:
    cfg = CorruptConfig(clean_fraction=0.0)
    seen_join = seen_leading_space = seen_removed_break = seen_wrong_para = False
    for seed in range(200):
        out = corrupt(CLEAN, random.Random(seed), cfg)
        labels = derive_labels(out.text, CLEAN)
        _, current = split(out.text)
        seen_join |= Gap.JOIN in labels
        seen_leading_space |= "\n " in out.text
        seen_removed_break |= any(
            c is Gap.SPACE and t in (Gap.NL, Gap.PARA) for c, t in zip(current, labels, strict=True)
        )
        seen_wrong_para |= any(
            c is Gap.PARA and t is not Gap.PARA for c, t in zip(current, labels, strict=True)
        )
    assert seen_join and seen_leading_space and seen_removed_break and seen_wrong_para


def test_severity_scales_noise() -> None:
    cfg = CorruptConfig(clean_fraction=0.0)
    changes: list[tuple[float, int]] = []
    for seed in range(300):
        out = corrupt(CLEAN, random.Random(seed), cfg)
        labels = derive_labels(out.text, CLEAN)
        _, current = split(out.text)
        n_changed = sum(1 for c, t in zip(current, labels, strict=True) if c != t)
        changes.append((out.severity, n_changed))
    low = [n for s, n in changes if s < 0.3]
    high = [n for s, n in changes if s > 0.7]
    assert sum(high) / len(high) > sum(low) / len(low)


@settings(max_examples=300)
@given(st.text(min_size=2), st.integers(min_value=0, max_value=10_000))
def test_invariant_on_arbitrary_unicode(text: str, seed: int) -> None:
    out = corrupt(text, random.Random(seed), CorruptConfig(clean_fraction=0.0))
    assert content(out.text) == content(text)
    assert out.text == normalize(out.text) or "\n " in out.text or " \n" in out.text or "\n\n" in out.text
```

The last assertion documents that corrupted text is not canonical: that is the point.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_corrupt.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/corrupt.py`**

```python
"""Seeded corruption of clean text (design section 3.3)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .text import GAP_STR, Gap, content, join, split


@dataclass(frozen=True)
class CorruptConfig:
    clean_fraction: float = 0.1
    remove_base: float = 0.6
    remove_scale: float = 0.4
    insert_min_chars: int = 40
    insert_max_chars: int = 200
    word_split_prob: float = 0.4
    double_prob: float = 0.1


@dataclass(frozen=True)
class Corrupted:
    text: str
    severity: float


def corrupt(clean: str, rng: random.Random, cfg: CorruptConfig | None = None) -> Corrupted:
    """Corrupt `clean`. Severity 0 means the canonical form of the input, unchanged."""
    cfg = cfg or CorruptConfig()
    tokens, gaps = split(clean)
    base = join(tokens, gaps)
    if len(tokens) < 2 or rng.random() < cfg.clean_fraction:
        return Corrupted(base, 0.0)
    severity = 1.0 - rng.random()  # in (0, 1]

    p_remove = cfg.remove_base + cfg.remove_scale * severity
    ws = [GAP_STR[g] for g in gaps]
    for i, g in enumerate(gaps):
        if g in (Gap.NL, Gap.PARA) and rng.random() < p_remove:
            ws[i] = " "

    chars = list(_interleave(tokens, ws))
    span = rng.uniform(cfg.insert_min_chars, cfg.insert_max_chars)
    n_insert = round(len(chars) * severity / span)
    positions = sorted((rng.randrange(len(chars)) for _ in range(n_insert)), reverse=True)
    for p in positions:
        _insert_newline(chars, p, rng, cfg)

    text = "".join(chars)
    if content(text) != content(clean):
        raise AssertionError("corruptor changed non-whitespace content")
    return Corrupted(text, severity)


def _interleave(tokens: Sequence[str], ws: Sequence[str]) -> str:
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i < len(ws):
            parts.append(ws[i])
    return "".join(parts)


def _insert_newline(chars: list[str], p: int, rng: random.Random, cfg: CorruptConfig) -> None:
    nl = "\n\n" if rng.random() < cfg.double_prob else "\n"
    if chars[p].isspace():
        _replace_run(chars, p, nl, rng)
    elif p > 0 and chars[p - 1].isspace():
        _replace_run(chars, p - 1, nl, rng)
    elif p > 0 and rng.random() < cfg.word_split_prob:
        chars[p:p] = list(nl)


def _replace_run(chars: list[str], p: int, nl: str, rng: random.Random) -> None:
    a = p
    while a > 0 and chars[a - 1].isspace():
        a -= 1
    b = p
    while b + 1 < len(chars) and chars[b + 1].isspace():
        b += 1
    chars[a : b + 1] = list(rng.choice([nl, nl + " ", " " + nl]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make check`
Expected: all pass. If `test_produces_every_kind_of_noise` fails on `seen_wrong_para`, confirm `double_prob` is applied in `_insert_newline` and that `CLEAN` contains enough text; do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add src/newline_fixer/corrupt.py tests/test_corrupt.py
git commit -m "feat: seeded corruptor producing removed breaks, mid-word and leading-space newlines"
```

---

### Task 5: Fixer protocol, identity fixer, windowing, `fix()`

**Files:**
- Create: `src/newline_fixer/models/__init__.py`, `src/newline_fixer/models/base.py`, `src/newline_fixer/models/identity.py`, `src/newline_fixer/windows.py`, `tests/test_windows.py`

**Interfaces:**
- Produces:
  - `class Fixer(Protocol)` with `name: str`, `budget: int`, `token_cost(token: str) -> int`, `gap_cost(gap: Gap) -> int`, `overhead() -> int`, `predict(tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]`
  - `class IdentityFixer` (name `"identity"`, budget 256, unit token cost)
  - `make_windows(n: int, token_costs: Sequence[int], gap_costs: Sequence[int], overhead: int, budget: int) -> list[tuple[int, int]]`
  - `assign_gaps(windows: Sequence[tuple[int, int]], n_gaps: int) -> list[int]`
  - `windows_for(fixer: Fixer, tokens: Sequence[str], current: Sequence[Gap]) -> list[tuple[int, int]]`
  - `predict_all(fixer: Fixer, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]`
  - `@dataclass(frozen=True) FixResult(text: str, tokens: int, gaps: int, changed: int, windows: int)`
  - `fix(text: str, fixer: Fixer) -> FixResult`

Per-token cost is capped by each fixer so a single token never exceeds half the budget; `make_windows` raises if it does. This is how design 4.1's "truncated for the model input only" is enforced: the fixer truncates what it feeds the model, and the cost it reports is the truncated cost.

- [ ] **Step 1: Write the failing tests**

```python
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from newline_fixer.models.identity import IdentityFixer
from newline_fixer.text import Gap, content, normalize, split
from newline_fixer.windows import FixResult, assign_gaps, fix, make_windows, predict_all


def unit(n: int) -> list[int]:
    return [1] * n


def test_make_windows_single_window_when_it_fits() -> None:
    assert make_windows(10, unit(10), [0] * 9, 0, 256) == [(0, 10)]


def test_make_windows_overlap_by_half() -> None:
    assert make_windows(10, unit(10), [0] * 9, 0, 4) == [(0, 4), (2, 6), (4, 8), (6, 10)]


def test_make_windows_counts_gap_and_overhead_costs() -> None:
    # budget 6, overhead 2 leaves 4; tokens cost 1, gaps cost 1 -> 2 tokens + 1 gap = 3, 3 tokens = 5 > 4
    assert make_windows(5, unit(5), [1] * 4, 2, 6)[0] == (0, 2)


def test_make_windows_rejects_oversized_token() -> None:
    with pytest.raises(ValueError):
        make_windows(3, [1, 300, 1], [0, 0], 0, 256)


def test_make_windows_empty() -> None:
    assert make_windows(0, [], [], 0, 256) == []


def test_assign_gaps_covers_every_gap_once() -> None:
    windows = [(0, 4), (2, 6), (4, 8), (6, 10)]
    owner = assign_gaps(windows, 9)
    assert len(owner) == 9
    assert all(0 <= o < len(windows) for o in owner)
    assert owner[0] == 0 and owner[8] == 3
    assert owner[4] in (1, 2)


def test_assign_gaps_raises_when_uncovered() -> None:
    with pytest.raises(ValueError):
        assign_gaps([(0, 2), (3, 5)], 4)


class Flip:
    """Test fixer: turns every SPACE into NL, counts calls."""

    name = "flip"
    budget = 4

    def __init__(self) -> None:
        self.calls = 0

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        self.calls += 1
        return [Gap.NL if g is Gap.SPACE else g for g in current]


def test_predict_all_merges_windows() -> None:
    tokens = [f"t{i}" for i in range(10)]
    current = [Gap.SPACE] * 9
    flip = Flip()
    assert predict_all(flip, tokens, current) == [Gap.NL] * 9
    assert flip.calls == 4


def test_fix_identity_normalizes_only() -> None:
    res = fix("  a \t b\n\n\nc ", IdentityFixer())
    assert res == FixResult(text="a b\n\nc", tokens=3, gaps=2, changed=0, windows=1)


def test_fix_reports_changes() -> None:
    res = fix("a b c", Flip())
    assert res.text == "a\nb\nc"
    assert res.changed == 2


def test_fix_empty_whitespace_and_single_token() -> None:
    assert fix("", IdentityFixer()).text == ""
    assert fix("   \n\t ", IdentityFixer()).text == ""
    assert fix("  word  ", IdentityFixer()) == FixResult("word", 1, 0, 0, 0)


def test_fix_long_token_is_covered() -> None:
    url = "https://example.com/" + "x" * 2000
    text = " ".join(["a"] * 300 + [url] + ["b"] * 300)
    res = fix(text, Flip())
    assert content(res.text) == content(text)
    assert res.gaps == 600 and res.changed == 600


@given(st.text())
def test_fix_identity_equals_normalize(text: str) -> None:
    assert fix(text, IdentityFixer()).text == normalize(text)


@given(st.text())
def test_fix_preserves_content_with_flip(text: str) -> None:
    out = fix(text, Flip()).text
    assert content(out) == content(text)
    tokens, _ = split(text)
    assert len(tokens) < 2 or "\n" in out or " " not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_windows.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/models/base.py`** (`models/__init__.py` is empty)

```python
"""The one interface every system implements (design section 4.1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..text import Gap


class Fixer(Protocol):
    """Predicts a gap class for every gap in a window of tokens.

    Costs describe how many model input units a span occupies, so windowing can respect
    the model's budget. A fixer must cap `token_cost` so that no single token costs more
    than half of `budget`; what it feeds the model for such a token is truncated, the
    reconstruction always uses the original token.
    """

    name: str
    budget: int

    def token_cost(self, token: str) -> int: ...

    def gap_cost(self, gap: Gap) -> int: ...

    def overhead(self) -> int: ...

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]: ...
```

- [ ] **Step 4: Implement `src/newline_fixer/models/identity.py`**

```python
"""B0: keep the current gap classes; output is the normalized input."""

from __future__ import annotations

from collections.abc import Sequence

from ..text import Gap


class IdentityFixer:
    name = "identity"
    budget = 256

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        return list(current)
```

- [ ] **Step 5: Implement `src/newline_fixer/windows.py`**

```python
"""Windowing, merging and the single `fix` entry point (design section 4.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .models.base import Fixer
from .text import Gap, join, split


def make_windows(
    n: int,
    token_costs: Sequence[int],
    gap_costs: Sequence[int],
    overhead: int,
    budget: int,
) -> list[tuple[int, int]]:
    """Greedy windows [start, end) over n tokens, each within budget, overlapping by half.

    Raises ValueError if a single token costs more than half the budget after overhead,
    because such a token could leave a gap uncovered.
    """
    if n == 0:
        return []
    if len(token_costs) != n or len(gap_costs) != max(n - 1, 0):
        raise ValueError("cost lists do not match token count")
    room = budget - overhead
    limit = room // 2
    for i, c in enumerate(token_costs):
        if c > limit:
            raise ValueError(f"token {i} costs {c}, above half the budget {limit}")
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = start + 1
        cost = token_costs[start]
        while end < n:
            extra = gap_costs[end - 1] + token_costs[end]
            if cost + extra > room:
                break
            cost += extra
            end += 1
        windows.append((start, end))
        if end >= n:
            return windows
        start += max(1, (end - start) // 2)


def assign_gaps(windows: Sequence[tuple[int, int]], n_gaps: int) -> list[int]:
    """Owner window per gap: the window containing both tokens whose center is nearest."""
    owner = [-1] * n_gaps
    best = [float("inf")] * n_gaps
    for w, (s, e) in enumerate(windows):
        center = (s + e - 1) / 2
        for g in range(s, min(e - 1, n_gaps)):
            d = abs(g + 0.5 - center)
            if d < best[g]:
                best[g] = d
                owner[g] = w
    if any(o < 0 for o in owner):
        raise ValueError("windowing left a gap without a prediction")
    return owner


def windows_for(fixer: Fixer, tokens: Sequence[str], current: Sequence[Gap]) -> list[tuple[int, int]]:
    token_costs = [fixer.token_cost(t) for t in tokens]
    gap_costs = [fixer.gap_cost(g) for g in current]
    return make_windows(len(tokens), token_costs, gap_costs, fixer.overhead(), fixer.budget)


def predict_all(fixer: Fixer, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
    if len(tokens) < 2:
        return []
    return _merge(fixer, tokens, current, windows_for(fixer, tokens, current))


def _merge(
    fixer: Fixer,
    tokens: Sequence[str],
    current: Sequence[Gap],
    windows: Sequence[tuple[int, int]],
) -> list[Gap]:
    n = len(tokens)
    owner = assign_gaps(windows, n - 1)
    out = list(current)
    for w, (s, e) in enumerate(windows):
        pred = fixer.predict(tokens[s:e], current[s : e - 1])
        if len(pred) != e - s - 1:
            raise ValueError(f"{fixer.name} returned {len(pred)} gaps for {e - s} tokens")
        for g in range(s, e - 1):
            if owner[g] == w:
                out[g] = pred[g - s]
    return out


@dataclass(frozen=True)
class FixResult:
    text: str
    tokens: int
    gaps: int
    changed: int
    windows: int


def fix(text: str, fixer: Fixer) -> FixResult:
    """Tokenize, predict every gap with the fixer, rebuild canonical text."""
    tokens, current = split(text)
    if len(tokens) < 2:
        return FixResult(join(tokens, []), len(tokens), 0, 0, 0)
    windows = windows_for(fixer, tokens, current)
    pred = _merge(fixer, tokens, current, windows)
    changed = sum(1 for a, b in zip(current, pred, strict=True) if a != b)
    return FixResult(join(tokens, pred), len(tokens), len(current), changed, len(windows))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `make check`
Expected: all pass. mypy must accept `Flip` and `IdentityFixer` as `Fixer` structurally.

- [ ] **Step 7: Commit**

```bash
git add src/newline_fixer/models src/newline_fixer/windows.py tests/test_windows.py
git commit -m "feat: fixer protocol, identity baseline, budgeted windowing with coverage check"
```

---

### Task 6: Lexicon and the rules fixer (B1)

**Files:**
- Create: `src/newline_fixer/lexicon.py`, `src/newline_fixer/rules.py`, `src/newline_fixer/resources/__init__.py`, `src/newline_fixer/resources/lexicon.txt` (placeholder of a few hundred common words until Task 11 replaces it), `tests/test_rules.py`

**Interfaces:**
- Produces:
  - `class Lexicon` with `__init__(words: Iterable[str])`, `known(word: str) -> bool`, `__len__`, `classmethod from_counts(counts: Mapping[str, int], min_count: int = 3) -> Lexicon`, `classmethod from_file(path: Path) -> Lexicon`, `classmethod bundled() -> Lexicon`, `write(path: Path) -> None`
  - `class RulesFixer(lexicon: Lexicon)` (name `"rules"`, budget 256) implementing `Fixer`
  - `rules_predict(tokens, current, lexicon) -> list[Gap]`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from newline_fixer.lexicon import Lexicon
from newline_fixer.rules import RulesFixer, rules_predict
from newline_fixer.text import Gap, normalize, split
from newline_fixer.windows import fix

WORDS = ["the", "model", "queries", "attention", "come", "from", "a", "use", "usea"]


def lex() -> Lexicon:
    return Lexicon(WORDS)


def gaps_for(text: str) -> list[Gap]:
    tokens, current = split(text)
    return rules_predict(tokens, current, lex())


def test_lexicon_basics() -> None:
    lx = lex()
    assert lx.known("The") and lx.known("queries") and not lx.known("que")
    assert len(lx) == len(set(WORDS))
    assert Lexicon.from_counts({"a": 5, "b": 2}, min_count=3).known("a")
    assert not Lexicon.from_counts({"a": 5, "b": 2}, min_count=3).known("b")


def test_lexicon_file_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "lex.txt"
    lex().write(p)
    assert Lexicon.from_file(p).known("attention")


def test_rule1_joins_only_real_fragments() -> None:
    assert gaps_for("que\nries") == [Gap.JOIN]
    assert gaps_for("the\nmodel") == [Gap.SPACE]
    assert gaps_for("use\na") == [Gap.SPACE]  # both known, keep the boundary


def test_rule2_lowercase_continuation_is_space() -> None:
    assert gaps_for("layers,\n the") == [Gap.SPACE]
    assert gaps_for("word\n\n, more") == [Gap.SPACE, Gap.SPACE]


def test_rule3_list_marker_after_colon_or_terminal() -> None:
    assert gaps_for("ways: • In") == [Gap.NL, Gap.SPACE]
    assert gaps_for("done. - next") == [Gap.NL, Gap.SPACE]
    assert gaps_for("x and - y") == [Gap.SPACE] * 3


def test_rule4_numbered_heading_then_capital_is_para() -> None:
    assert gaps_for("3.2.3 Applications of Attention in our Model The Transformer")[-2] is Gap.PARA
    assert gaps_for("3.2.3 Applications of Attention")[0] is Gap.SPACE
    assert gaps_for("3.2.3 Applications of Attention")[2] is Gap.SPACE


def test_rule4_title_case_heading_only_when_break_present() -> None:
    assert gaps_for("Applications Of Attention\nThe Transformer")[2] is Gap.PARA
    assert gaps_for("Applications Of Attention The Transformer")[2] is Gap.SPACE


def test_rules_solve_the_example(example_input: str, example_output: str) -> None:
    out = fix(example_input, RulesFixer(lex()))
    assert out.text == normalize(example_output)


def test_bundled_lexicon_loads() -> None:
    assert len(Lexicon.bundled()) > 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rules.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/lexicon.py`**

```python
"""Word list used by the rules baseline (design section 4.3)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path


class Lexicon:
    def __init__(self, words: Iterable[str]) -> None:
        self._words = {w.strip().lower() for w in words if w.strip()}

    def known(self, word: str) -> bool:
        return word.lower() in self._words

    def __len__(self) -> int:
        return len(self._words)

    @classmethod
    def from_counts(cls, counts: Mapping[str, int], min_count: int = 3) -> Lexicon:
        return cls(w for w, c in counts.items() if c >= min_count and w.isalpha())

    @classmethod
    def from_file(cls, path: Path) -> Lexicon:
        return cls(path.read_text(encoding="utf-8").splitlines())

    @classmethod
    def bundled(cls) -> Lexicon:
        text = resources.files("newline_fixer.resources").joinpath("lexicon.txt").read_text("utf-8")
        return cls(text.splitlines())

    def write(self, path: Path) -> None:
        path.write_text("\n".join(sorted(self._words)) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Create the placeholder resource**

`src/newline_fixer/resources/__init__.py` is empty. Generate `lexicon.txt` with the 300 or so most common English words plus the words of the challenge example; the simplest way:

```bash
uv run python - <<'PY'
from pathlib import Path
words = """the of and to in a is that for it as was with be by on not he i this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no she my its about out up what some them than into may say then do could time only now very any made other new just these should way like such people over our me also first how after into three different ways uses attention transformer model applications encoder decoder layers queries come previous layer keys values output multi head""".split()
Path("src/newline_fixer/resources/lexicon.txt").write_text("\n".join(sorted(set(words))) + "\n")
PY
```

Then append the 1,000 most common English words from any public word-frequency list you have locally, or leave the list as is; Task 11 replaces this file with the lexicon built from the training split.

- [ ] **Step 5: Implement `src/newline_fixer/rules.py`**

```python
"""B1: the rule-based baseline (design section 4.3)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .lexicon import Lexicon
from .text import Gap

LIST_MARKER = re.compile(r"^(?:[•\-\*–]|\d{1,3}[.)])$")
SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?$")
TERMINAL = ".!?:;"
CLOSING = ")]}.,;:"
STOP = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with",
    "our", "at", "by", "from", "as", "vs", "into", "over", "under", "per",
}
MAX_HEADING_TOKENS = 8


def _heading_like(line: Sequence[str], require_number: bool) -> bool:
    if not line or len(line) > MAX_HEADING_TOKENS:
        return False
    last = line[-1]
    if last[-1] in TERMINAL or last.lower() in STOP:
        return False
    if SECTION_NUMBER.match(line[0]):
        return len(line) >= 2
    if require_number:
        return False
    capitalized = [t for t in line if t[0].isalpha() and t[0].isupper()]
    acceptable = all(t.lower() in STOP or not t[0].isalpha() or t[0].isupper() for t in line)
    return acceptable and len(capitalized) >= 1


def rules_predict(tokens: Sequence[str], current: Sequence[Gap], lexicon: Lexicon) -> list[Gap]:
    out: list[Gap] = []
    line_start = 0
    for i, cur in enumerate(current):
        left, right = tokens[i], tokens[i + 1]
        has_break = cur in (Gap.NL, Gap.PARA)
        line = tokens[line_start : i + 1]
        if (
            has_break
            and left.isalpha()
            and right.isalpha()
            and lexicon.known(left + right)
            and not (lexicon.known(left) and lexicon.known(right))
        ):
            g = Gap.JOIN
        elif has_break and (right[0].islower() or right[0] in CLOSING):
            g = Gap.SPACE
        elif LIST_MARKER.match(right) and left[-1] in TERMINAL:
            g = Gap.NL
        elif right[0].isupper() and _heading_like(line, require_number=not has_break):
            g = Gap.PARA
        else:
            g = cur
        out.append(g)
        if g in (Gap.NL, Gap.PARA):
            line_start = i + 1
    return out


class RulesFixer:
    name = "rules"
    budget = 256

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon or Lexicon.bundled()

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        return rules_predict(tokens, current, self.lexicon)
```

Note `line_start` is reset when the *predicted* gap is a break, so rule 4 sees the line as the rules have rebuilt it. That is what lets the example's heading be recognized after rule 2 has turned `Attention\n in` into a space.

- [ ] **Step 6: Run tests to verify they pass**

Run: `make check`
Expected: all pass. If `test_rules_solve_the_example` fails, print `fix(example_input, RulesFixer(lex())).text` and compare gap by gap with `derive_labels`; the expected trace is in design 4.3 and in the test `test_rule4_numbered_heading_then_capital_is_para`.

- [ ] **Step 7: Commit**

```bash
git add src/newline_fixer/lexicon.py src/newline_fixer/rules.py src/newline_fixer/resources tests/test_rules.py
git commit -m "feat: lexicon and rules baseline; the challenge example passes through B1"
```

---

### Task 7: Metrics

**Files:**
- Create: `src/newline_fixer/eval/__init__.py`, `src/newline_fixer/eval/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `@dataclass ClassCounts(tp=0, fp=0, fn=0, support=0)` with properties `precision`, `recall`, `f1` (0.0 when undefined)
  - `class GapMetrics` with `update(pred: Sequence[Gap], ref: Sequence[Gap], current: Sequence[Gap]) -> None`, `per_class: dict[Gap, ClassCounts]`, `n_gaps: int`, `n_changed: int`, `n_wrong_join: int`, `n_items: int`, `n_items_untouched: int`, `macro_f1() -> float`, `macro_classes() -> list[Gap]`, `break_f1() -> float`, `wrong_join_per_1000() -> float`, `damage_rate() -> float`, `to_dict() -> dict[str, object]`
  - `paragraph_match(output: str, reference: str) -> tuple[int, int]` (matched, total)

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from newline_fixer.eval.metrics import ClassCounts, GapMetrics, paragraph_match
from newline_fixer.text import Gap

J, S, N, P = Gap.JOIN, Gap.SPACE, Gap.NL, Gap.PARA


def test_class_counts_properties() -> None:
    c = ClassCounts(tp=2, fp=1, fn=1, support=3)
    assert c.precision == pytest.approx(2 / 3)
    assert c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3)
    assert ClassCounts().f1 == 0.0


def test_gap_metrics_counts() -> None:
    m = GapMetrics()
    m.update(pred=[S, N, J, P], ref=[S, S, J, N], current=[S, S, N, N])
    assert m.n_gaps == 4
    assert m.per_class[S] == ClassCounts(tp=1, fp=0, fn=1, support=2)
    assert m.per_class[N] == ClassCounts(tp=0, fp=1, fn=1, support=1)
    assert m.per_class[J] == ClassCounts(tp=1, fp=0, fn=0, support=1)
    assert m.per_class[P] == ClassCounts(tp=0, fp=1, fn=0, support=0)
    assert m.n_changed == 3
    assert m.n_wrong_join == 0
    assert m.macro_classes() == [J, S, N]
    assert m.n_items == 1 and m.n_items_untouched == 0


def test_wrong_join_and_untouched() -> None:
    m = GapMetrics()
    m.update(pred=[J, S], ref=[S, S], current=[S, S])
    m.update(pred=[S], ref=[S], current=[S])
    assert m.n_wrong_join == 1
    assert m.wrong_join_per_1000() == pytest.approx(1000 / 3)
    assert m.n_items == 2 and m.n_items_untouched == 1
    assert m.damage_rate() == pytest.approx(1 / 3)


def test_break_f1() -> None:
    m = GapMetrics()
    m.update(pred=[N, S, P, S], ref=[N, N, S, S], current=[S, S, S, S])
    # breaks: pred {0,2}, ref {0,1}: tp 1, fp 1, fn 1 -> f1 0.5
    assert m.break_f1() == pytest.approx(0.5)


def test_update_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        GapMetrics().update(pred=[S], ref=[S, S], current=[S, S])


def test_paragraph_match() -> None:
    ref = "Title\n\nFirst para.\n\nSecond para."
    assert paragraph_match("Title\n\nFirst para.\n\nSecond para.", ref) == (3, 3)
    assert paragraph_match("Title First para.\n\nSecond para.", ref) == (1, 3)
    assert paragraph_match("", ref) == (0, 3)


def test_to_dict_has_expected_keys() -> None:
    m = GapMetrics()
    m.update(pred=[S], ref=[S], current=[S])
    d = m.to_dict()
    assert set(d) >= {"n_gaps", "macro_f1", "macro_classes", "break_f1", "wrong_join_per_1000", "damage_rate", "per_class"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/eval/metrics.py`** (`eval/__init__.py` is empty)

```python
"""Gap-level and paragraph metrics (design section 5.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..text import Gap

BREAKS = (Gap.NL, Gap.PARA)


@dataclass
class ClassCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    support: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class GapMetrics:
    per_class: dict[Gap, ClassCounts] = field(default_factory=lambda: {g: ClassCounts() for g in Gap})
    n_gaps: int = 0
    n_changed: int = 0
    n_wrong_join: int = 0
    n_items: int = 0
    n_items_untouched: int = 0
    break_tp: int = 0
    break_fp: int = 0
    break_fn: int = 0

    def update(self, pred: Sequence[Gap], ref: Sequence[Gap], current: Sequence[Gap]) -> None:
        if not (len(pred) == len(ref) == len(current)):
            raise ValueError("pred, ref and current must have the same length")
        changed = 0
        for p, r, c in zip(pred, ref, current, strict=True):
            self.n_gaps += 1
            self.per_class[r].support += 1
            if p == r:
                self.per_class[p].tp += 1
            else:
                self.per_class[p].fp += 1
                self.per_class[r].fn += 1
                if p is Gap.JOIN:
                    self.n_wrong_join += 1
            if p != c:
                changed += 1
            pb, rb = p in BREAKS, r in BREAKS
            if pb and rb:
                self.break_tp += 1
            elif pb:
                self.break_fp += 1
            elif rb:
                self.break_fn += 1
        self.n_changed += changed
        self.n_items += 1
        if changed == 0:
            self.n_items_untouched += 1

    def macro_classes(self) -> list[Gap]:
        return [g for g in Gap if self.per_class[g].support > 0]

    def macro_f1(self) -> float:
        classes = self.macro_classes()
        return sum(self.per_class[g].f1 for g in classes) / len(classes) if classes else 0.0

    def break_f1(self) -> float:
        tp, fp, fn = self.break_tp, self.break_fp, self.break_fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    def wrong_join_per_1000(self) -> float:
        return 1000 * self.n_wrong_join / self.n_gaps if self.n_gaps else 0.0

    def damage_rate(self) -> float:
        return self.n_changed / self.n_gaps if self.n_gaps else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "n_items": self.n_items,
            "n_gaps": self.n_gaps,
            "macro_f1": self.macro_f1(),
            "macro_classes": [g.name for g in self.macro_classes()],
            "break_f1": self.break_f1(),
            "wrong_join_per_1000": self.wrong_join_per_1000(),
            "damage_rate": self.damage_rate(),
            "items_untouched_rate": self.n_items_untouched / self.n_items if self.n_items else 0.0,
            "per_class": {
                g.name: {
                    "support": c.support,
                    "precision": c.precision,
                    "recall": c.recall,
                    "f1": c.f1,
                }
                for g, c in self.per_class.items()
            },
        }


def paragraph_match(output: str, reference: str) -> tuple[int, int]:
    """(matched, total): reference paragraphs that occur verbatim among output paragraphs."""
    ref_paras = [p for p in reference.split("\n\n") if p]
    out_paras = {p for p in output.split("\n\n") if p}
    return sum(1 for p in ref_paras if p in out_paras), len(ref_paras)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/newline_fixer/eval tests/test_metrics.py
git commit -m "feat: gap-level metrics with per-class support, wrong-join rate and paragraph match"
```

---

### Task 8: Dataset records, filters, deduplication, grouped split, passages

**Files:**
- Create: `src/newline_fixer/data/__init__.py`, `src/newline_fixer/data/records.py`, `src/newline_fixer/data/filters.py`, `src/newline_fixer/data/splits.py`, `src/newline_fixer/data/passages.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) CleanDoc(id: str, source: str, source_ref: str, group: str, sha256: str, clean: str)` with `classmethod make(id, source, source_ref, group, text) -> CleanDoc` (normalizes and hashes)
  - `@dataclass(frozen=True) EvalItem(id: str, source: str, input: str, target: str, severity: float, meta: dict[str, object])`
  - `read_jsonl(path: Path, cls: type[T]) -> list[T]`, `write_jsonl(path: Path, rows: Iterable[T]) -> int`
  - `is_hard_wrapped(text: str) -> bool`, `long_enough(text: str, min_chars: int = 200) -> bool`, `has_structure(text: str) -> bool`
  - `dedupe(docs: Iterable[CleanDoc]) -> list[CleanDoc]`
  - `assign_splits(groups: Iterable[str], seed: int, val: float = 0.05, test: float = 0.05) -> dict[str, str]`
  - `cut_passages(text: str, min_chars: int = 300, max_chars: int = 800) -> list[str]`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from newline_fixer.data.filters import has_structure, is_hard_wrapped, long_enough
from newline_fixer.data.passages import cut_passages
from newline_fixer.data.records import CleanDoc, EvalItem, read_jsonl, write_jsonl
from newline_fixer.data.splits import assign_splits, dedupe

HARD_WRAPPED = (
    "It was the best of times, it was the worst of times, it was the age of\n"
    "wisdom, it was the age of foolishness, it was the epoch of belief, it was\n"
    "the epoch of incredulity, it was the season of Light, it was the season of\n"
    "Darkness, it was the spring of hope, it was the winter of despair.\n"
)
PARAGRAPHED = (
    "History\n\n"
    "The town was founded in 1820. It grew quickly after the railway arrived.\n\n"
    "Economy\n\n"
    "Farming remains the main activity. Tourism is growing.\n"
)


def test_clean_doc_make_normalizes_and_hashes() -> None:
    d = CleanDoc.make("x", "test", "ref", "g", "  a  b\n\n\nc ")
    assert d.clean == "a b\n\nc"
    assert len(d.sha256) == 64
    assert d == CleanDoc.make("x", "test", "ref", "g", "a b\n\nc")


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [CleanDoc.make("1", "s", "r", "g", "hello world"), CleanDoc.make("2", "s", "r2", "g2", "x\n\ny")]
    p = tmp_path / "docs.jsonl"
    assert write_jsonl(p, rows) == 2
    assert read_jsonl(p, CleanDoc) == rows
    items = [EvalItem("a", "s", "in put", "in\nput", 0.5, {"k": 1})]
    q = tmp_path / "items.jsonl"
    write_jsonl(q, items)
    assert read_jsonl(q, EvalItem) == items


def test_hard_wrap_filter() -> None:
    assert is_hard_wrapped(HARD_WRAPPED)
    assert not is_hard_wrapped(PARAGRAPHED)
    assert not is_hard_wrapped("one line only")


def test_length_and_structure_filters() -> None:
    assert not long_enough("short")
    assert long_enough("x" * 200)
    assert has_structure(PARAGRAPHED)
    assert not has_structure("a single paragraph with no breaks at all " * 5)


def test_dedupe_by_hash_and_prefix() -> None:
    a = CleanDoc.make("1", "s", "r", "g", "same text " * 30)
    b = CleanDoc.make("2", "s", "r", "g", "same text " * 30)
    c = CleanDoc.make("3", "s", "r", "g", "same text " * 30 + "but longer tail")
    d = CleanDoc.make("4", "s", "r", "g", "different " * 30)
    out = dedupe([a, b, c, d])
    assert [x.id for x in out] == ["1", "4"]


def test_assign_splits_is_deterministic_and_grouped() -> None:
    groups = [f"g{i}" for i in range(1000)]
    s1 = assign_splits(groups, seed=1)
    s2 = assign_splits(groups, seed=1)
    assert s1 == s2
    counts = {k: sum(1 for v in s1.values() if v == k) for k in ("train", "val", "test")}
    assert 40 <= counts["val"] <= 60 and 40 <= counts["test"] <= 60
    assert counts["train"] == 1000 - counts["val"] - counts["test"]
    assert assign_splits(groups, seed=2) != s1


def test_cut_passages_respects_bounds_and_boundaries() -> None:
    paras = [f"Paragraph {i}. " + "word " * 40 for i in range(12)]
    text = "\n\n".join(p.strip() for p in paras)
    out = cut_passages(text, min_chars=300, max_chars=800)
    assert out
    for p in out:
        assert 300 <= len(p) <= 800
        assert p == p.strip()
        assert all(part in text for part in p.split("\n\n"))
    assert cut_passages("too short") == []


def test_cut_passages_is_deterministic() -> None:
    text = "\n\n".join("Sentence number %d is here." % i + " filler" * 20 for i in range(20))
    assert cut_passages(text) == cut_passages(text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/data/records.py`** (`data/__init__.py` is empty)

```python
"""Dataset record types and JSONL IO (design section 3.2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TypeVar

from ..text import normalize


@dataclass(frozen=True)
class CleanDoc:
    id: str
    source: str
    source_ref: str
    group: str
    sha256: str
    clean: str

    @classmethod
    def make(cls, id: str, source: str, source_ref: str, group: str, text: str) -> CleanDoc:
        clean = normalize(text)
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        return cls(id=id, source=source, source_ref=source_ref, group=group, sha256=digest, clean=clean)


@dataclass(frozen=True)
class EvalItem:
    id: str
    source: str
    input: str
    target: str
    severity: float = 0.0
    meta: dict[str, object] = field(default_factory=dict)


T = TypeVar("T", CleanDoc, EvalItem)


def write_jsonl(path: Path, rows: Iterable[T]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path, cls: type[T]) -> list[T]:
    names = {f.name for f in fields(cls)}
    out: list[T] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                out.append(cls(**{k: v for k, v in raw.items() if k in names}))
    return out
```

- [ ] **Step 4: Implement `src/newline_fixer/data/filters.py`**

```python
"""Document filters (design section 3.1)."""

from __future__ import annotations

TERMINAL = ".!?:;\"')»”"


def long_enough(text: str, min_chars: int = 200) -> bool:
    return len(text) >= min_chars


def has_structure(text: str) -> bool:
    """At least one paragraph break or line break survives normalization."""
    return "\n" in text


def is_hard_wrapped(text: str, threshold: float = 0.3) -> bool:
    """True when many lines end mid-sentence and the next line starts lowercase."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return False
    suspicious = 0
    for a, b in zip(lines, lines[1:], strict=False):
        if a[-1] not in TERMINAL and b[0].islower():
            suspicious += 1
    return suspicious / (len(lines) - 1) > threshold
```

- [ ] **Step 5: Implement `src/newline_fixer/data/splits.py`**

```python
"""Deduplication and grouped splitting (design section 3.2)."""

from __future__ import annotations

import random
from collections.abc import Iterable

from .records import CleanDoc

PREFIX_CHARS = 200


def dedupe(docs: Iterable[CleanDoc]) -> list[CleanDoc]:
    seen_hash: set[str] = set()
    seen_prefix: set[str] = set()
    out: list[CleanDoc] = []
    for d in docs:
        prefix = d.clean[:PREFIX_CHARS]
        if d.sha256 in seen_hash or prefix in seen_prefix:
            continue
        seen_hash.add(d.sha256)
        seen_prefix.add(prefix)
        out.append(d)
    return out


def assign_splits(
    groups: Iterable[str], seed: int, val: float = 0.05, test: float = 0.05
) -> dict[str, str]:
    ids = sorted(set(groups))
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * val)
    n_test = round(len(ids) * test)
    out: dict[str, str] = {}
    for i, g in enumerate(ids):
        if i < n_val:
            out[g] = "val"
        elif i < n_val + n_test:
            out[g] = "test"
        else:
            out[g] = "train"
    return out
```

- [ ] **Step 6: Implement `src/newline_fixer/data/passages.py`**

```python
"""Cut documents into passages on paragraph boundaries (design section 5.1)."""

from __future__ import annotations


def cut_passages(text: str, min_chars: int = 300, max_chars: int = 800) -> list[str]:
    """Greedy, deterministic: accumulate paragraphs until adding one would exceed max."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        if len(p) > max_chars:
            if min_chars <= size:
                out.append("\n\n".join(buf))
            buf, size = [], 0
            continue
        extra = len(p) + (2 if buf else 0)
        if size + extra > max_chars:
            if size >= min_chars:
                out.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += extra
    if size >= min_chars:
        out.append("\n\n".join(buf))
    return out
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `make check`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/newline_fixer/data tests/test_data.py
git commit -m "feat: dataset records, filters, dedupe, grouped split, passage cutting"
```

---

### Task 9: Wikipedia loader with pinned revision

**Files:**
- Create: `src/newline_fixer/data/wikipedia.py`, `tests/test_wikipedia.py`

**Interfaces:**
- Consumes: `CleanDoc.make`, `filters`
- Produces:
  - `clean_wikipedia_text(text: str, max_chars: int = 4000) -> str | None`
  - `resolve_revision(repo: str = "wikimedia/wikipedia") -> str`
  - `iter_wikipedia(n_docs: int, seed: int, revision: str, config: str = "20231101.en") -> Iterator[CleanDoc]`

The network loader is exercised manually in Task 11; unit tests cover only the pure text cleaning. The `datasets` and `huggingface_hub` APIs are imported inside functions so the core package stays dependency-free.

- [ ] **Step 1: Write the failing tests**

```python
from newline_fixer.data.wikipedia import clean_wikipedia_text

ARTICLE = (
    "Anarchism is a political philosophy. It questions the legitimacy of hierarchy and "
    "proposes voluntary association in its place.\n\n"
    "It is skeptical of authority. Its advocates have differed on tactics, on economics "
    "and on how far the critique of hierarchy should extend.\n\n"
    "Etymology\n\n"
    "The word comes from Greek. It entered English in the sixteenth century and was used "
    "as a term of abuse long before anyone claimed it.\n\n"
    "See also\n\n"
    "Libertarianism\nMutualism\n\n"
    "References\n\n"
    "Some citation.\n"
)


def test_clean_wikipedia_drops_trailing_sections_and_keeps_body() -> None:
    out = clean_wikipedia_text(ARTICLE)
    assert out is not None
    assert out.startswith("Anarchism is a political philosophy.\n\nIt is skeptical")
    assert "Etymology\n\nThe word comes from Greek." in out
    assert "See also" not in out and "References" not in out


def test_clean_wikipedia_truncates_on_paragraph_boundary() -> None:
    long = "\n\n".join(f"Paragraph {i} " + "text " * 100 for i in range(30))
    out = clean_wikipedia_text(long, max_chars=2000)
    assert out is not None and len(out) <= 2000
    assert out.endswith(out.split("\n\n")[-1])
    assert all(p.startswith("Paragraph") for p in out.split("\n\n"))


def test_clean_wikipedia_rejects_short() -> None:
    assert clean_wikipedia_text("Too short.") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wikipedia.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/data/wikipedia.py`**

```python
"""Wikipedia articles from the `wikimedia/wikipedia` dump (design section 3.1)."""

from __future__ import annotations

from collections.abc import Iterator

from ..text import normalize
from .filters import has_structure, is_hard_wrapped, long_enough
from .records import CleanDoc

TRAILING_SECTIONS = {
    "see also", "references", "external links", "notes", "further reading",
    "bibliography", "sources", "footnotes", "gallery",
}


def clean_wikipedia_text(text: str, max_chars: int = 4000) -> str | None:
    """Drop trailing reference sections, truncate on a paragraph boundary, filter."""
    paras = [p.strip() for p in normalize(text).split("\n\n") if p.strip()]
    kept: list[str] = []
    size = 0
    for p in paras:
        if p.lower() in TRAILING_SECTIONS:
            break
        extra = len(p) + (2 if kept else 0)
        if size + extra > max_chars:
            break
        kept.append(p)
        size += extra
    out = "\n\n".join(kept)
    if not (long_enough(out) and has_structure(out)) or is_hard_wrapped(out):
        return None
    return out


def resolve_revision(repo: str = "wikimedia/wikipedia") -> str:
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo).sha
    if not sha:
        raise RuntimeError(f"could not resolve revision for {repo}")
    return sha


def iter_wikipedia(
    n_docs: int, seed: int, revision: str, config: str = "20231101.en"
) -> Iterator[CleanDoc]:
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True, revision=revision
    ).shuffle(seed=seed, buffer_size=10_000)
    produced = 0
    for row in ds:
        cleaned = clean_wikipedia_text(row["text"])
        if cleaned is None:
            continue
        article_id = str(row["id"])
        yield CleanDoc.make(f"wiki-{article_id}", "wikipedia", article_id, f"wiki-{article_id}", cleaned)
        produced += 1
        if produced >= n_docs:
            return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/newline_fixer/data/wikipedia.py tests/test_wikipedia.py
git commit -m "feat: wikipedia loader with trailing-section removal and pinned revision"
```

---

### Task 10: Generated documents through the Claude API

**Files:**
- Create: `src/newline_fixer/data/generated.py`, `tests/test_generated.py`

**Interfaces:**
- Produces:
  - `REGISTERS: list[str]`, `TOPICS: list[str]`, `MARKERS: list[str]`
  - `build_brief(register: str, topic: str, marker: str) -> str`
  - `validate_generated(text: str, marker: str) -> str | None` (normalized text or None)
  - `generate_docs(n: int, seed: int, out_dir: Path, model: str, client: object | None = None) -> list[CleanDoc]` (caches each document as `out_dir/NNNNN.txt` and skips existing files; the client is any object with `messages.create(...)` so tests can stub it)

Before writing the call, load the `claude-api` skill and confirm the current model id for the cheapest capable model and the `messages.create` signature; the code below assumes `anthropic.Anthropic().messages.create(model=..., max_tokens=..., messages=[{"role": "user", "content": ...}])` returning `.content[0].text`. Default model: `claude-haiku-4-5-20251001`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
from types import SimpleNamespace

from newline_fixer.data.generated import build_brief, generate_docs, validate_generated

GOOD = (
    "2.1 Scope\n\n"
    "This manual covers the installation of the pump in a domestic garden setting. Read it "
    "fully before starting, and keep it near the pump for later reference.\n\n"
    "Tools required:\n"
    "- a torque wrench\n"
    "- two M8 bolts\n"
    "- thread sealant\n\n"
    "Keep the work area dry. The pump must not run without water for more than ten seconds, "
    "because the seals depend on water for cooling."
)


def test_brief_mentions_register_topic_marker() -> None:
    b = build_brief("user manual", "a garden pump", "-")
    assert "user manual" in b and "garden pump" in b and '"-"' in b


def test_validate_accepts_structured_plain_text() -> None:
    assert validate_generated(GOOD, "-") == GOOD


def test_validate_rejects_markdown_or_missing_structure() -> None:
    assert validate_generated("# Title\n\nbody\n\n- item", "-") is None
    assert validate_generated("just one paragraph of text", "-") is None


def test_validate_accepts_any_known_marker() -> None:
    # Models sometimes substitute a marker; any marker from MARKERS is acceptable.
    assert validate_generated(GOOD.replace("- ", "* "), "-") is not None


class FakeMessages:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(text=GOOD)])


def test_generate_docs_caches_and_validates(tmp_path: Path) -> None:
    fake = SimpleNamespace(messages=FakeMessages())
    docs = generate_docs(3, seed=1, out_dir=tmp_path, model="test", client=fake)
    assert len(docs) == 3
    assert fake.messages.calls == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["00000.txt", "00001.txt", "00002.txt"]
    again = generate_docs(3, seed=1, out_dir=tmp_path, model="test", client=fake)
    assert fake.messages.calls == 3
    assert again == docs
    assert docs[0].source == "generated" and docs[0].group == "gen-00000"
```

`validate_generated(text, marker)` accepts a document that has a paragraph break, no Markdown heading, emphasis or code fence, and at least two consecutive lines starting with the same marker from `MARKERS`. The `marker` argument is the one that was requested; it is kept in the signature so a later version can be stricter, but any known marker is accepted because models sometimes substitute one.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generated.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/data/generated.py`**

```python
"""LLM-generated structured documents (design section 3.1, decision 0005)."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from ..text import normalize
from .filters import has_structure, long_enough
from .records import CleanDoc

REGISTERS = [
    "a section of a research paper", "a user manual", "a technical report",
    "a business email", "meeting notes", "product documentation", "a policy memo",
    "a tutorial", "a news article", "an internal wiki page", "a grant proposal",
    "lecture notes", "a changelog with explanations", "a legal summary",
]
TOPICS = [
    "a garden pump", "attention mechanisms", "a city budget", "bread baking", "bicycle repair",
    "a database migration", "coral reefs", "a new hiring process", "solar panel installation",
    "a chess opening", "a kitchen renovation", "a vaccination campaign", "a railway timetable",
    "a mobile app release", "soil chemistry", "a library catalogue", "a marathon training plan",
    "wind turbine maintenance", "a school curriculum change", "a satellite launch",
]
MARKERS = ["•", "-", "*", "1."]

_MARKDOWN = re.compile(r"(^#+\s|\*\*|__|^```)", re.MULTILINE)


def build_brief(register: str, topic: str, marker: str) -> str:
    return (
        f"Write one self-contained English document of 250 to 600 words in the form of "
        f"{register}. Topic: {topic}.\n"
        "Rules: plain text only, no Markdown syntax (no #, no **, no backticks). Include at "
        "least one heading on its own line (either numbered like \"2.1 Scope\" or a short "
        f"title-case line), at least one list of three or more items on separate lines each "
        f'starting with "{marker}" (for "1." continue 2., 3.), and at least two ordinary '
        "paragraphs. Separate paragraphs and headings with one blank line. Put list items on "
        "consecutive lines with no blank lines between them. Never wrap lines inside a "
        "paragraph. Output only the document."
    )


def _list_lines(text: str) -> int:
    best = 0
    for marker in MARKERS:
        run = 0
        for line in text.split("\n"):
            starts = line.startswith(marker + " ") or (
                marker == "1." and re.match(r"^\d+\.\s", line) is not None
            )
            run = run + 1 if starts else 0
            best = max(best, run)
    return best


def validate_generated(text: str, marker: str) -> str | None:
    """Return the normalized document if it is usable; `marker` is the requested one."""
    del marker  # any marker from MARKERS is accepted, see the test
    norm = normalize(text)
    if _MARKDOWN.search(norm):
        return None
    if not (long_enough(norm) and has_structure(norm) and "\n\n" in norm):
        return None
    if _list_lines(norm) < 2:
        return None
    return norm


def generate_docs(
    n: int, seed: int, out_dir: Path, model: str, client: Any | None = None
) -> list[CleanDoc]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    choices: list[tuple[str, str, str]] = [
        (rng.choice(REGISTERS), rng.choice(TOPICS), rng.choice(MARKERS)) for _ in range(n)
    ]
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    docs: list[CleanDoc] = []
    for i, (register, topic, marker) in enumerate(choices):
        path = out_dir / f"{i:05d}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = _ask(client, model, build_brief(register, topic, marker), marker)
            path.write_text(text, encoding="utf-8")
        docs.append(CleanDoc.make(f"gen-{i:05d}", "generated", f"{model}|{register}|{topic}", f"gen-{i:05d}", text))
    return docs


def _ask(client: Any, model: str, brief: str, marker: str, attempts: int = 3) -> str:
    for _ in range(attempts):
        msg = client.messages.create(
            model=model, max_tokens=1200, messages=[{"role": "user", "content": brief}]
        )
        text = validate_generated(msg.content[0].text, marker)
        if text is not None:
            return text
    raise RuntimeError("generation failed validation three times")
```

The prompt text, model id and date are recorded by `scripts/build_data.py generate` in Task 11.

- [ ] **Step 4: Run tests to verify they pass**

Run: `make check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/newline_fixer/data/generated.py tests/test_generated.py
git commit -m "feat: generated structured documents with validation and on-disk cache"
```

---

### Task 11: `build_data` script, evaluation sets, lexicon resource, manifest, publish

**Files:**
- Create: `scripts/build_data.py`, `src/newline_fixer/data/build.py`, `src/newline_fixer/data/manifest.py`, `tests/test_build.py`
- Replace: `src/newline_fixer/resources/lexicon.txt`
- Produce (committed): `data/split.json`, `data/sets/V1.jsonl`, `data/sets/V3.jsonl`, `data/sets/T1.jsonl`, `data/sets/T3.jsonl`, `data/sets/T0.jsonl`, `data/sets/meta.json` (seed, corruptor parameters, commit)
- Produce (not committed, published): `data/clean/{train,val,test}.jsonl`, `data/manifest.json`

**Interfaces:**
- Produces:
  - `assemble(docs: Iterable[CleanDoc], seed: int) -> dict[str, list[CleanDoc]]` (filters, dedupes, splits; keys `train`, `val`, `test`)
  - `make_corrupted_set(docs: Sequence[CleanDoc], seed: int, limit: int | None) -> list[EvalItem]`
  - `make_clean_set(docs: Sequence[CleanDoc], seed: int, n_passages: int) -> list[EvalItem]`
  - `build_lexicon(docs: Iterable[CleanDoc], min_count: int = 3) -> Lexicon`
  - `write_manifest(path: Path, entries: dict[str, object]) -> None`, `file_sha256(path: Path) -> str`
  - CLI: `uv run python scripts/build_data.py {wikipedia,generate,assemble,sets,lexicon,publish} ...`

- [ ] **Step 1: Write the failing tests** for the pure functions

Documents in these tests must differ within their first 200 characters, or `dedupe` collapses them; `body(i)` puts the index into every repeated word for that reason.

```python
from pathlib import Path

from newline_fixer.data.build import assemble, build_lexicon, make_clean_set, make_corrupted_set
from newline_fixer.data.manifest import file_sha256, write_manifest
from newline_fixer.data.records import CleanDoc
from newline_fixer.text import content, derive_labels


def doc(i: int, text: str, group: str | None = None) -> CleanDoc:
    return CleanDoc.make(f"d{i}", "test", str(i), group or f"g{i}", text)


def body(i: int) -> str:
    return "\n\n".join(f"Heading {i}-{k}\n\nParagraph {k}. " + f"words {i} " * 30 for k in range(4))


def test_assemble_filters_dedupes_and_splits_by_group() -> None:
    docs = [doc(i, body(i), group=f"g{i % 50}") for i in range(200)]
    docs.append(doc(999, "short"))
    docs.append(doc(998, body(1)))
    parts = assemble(docs, seed=3)
    assert sum(len(v) for v in parts.values()) == 200
    group_side = {}
    for side, ds in parts.items():
        for d in ds:
            assert group_side.setdefault(d.group, side) == side


def test_corrupted_set_items_are_consistent() -> None:
    docs = [doc(i, body(i)) for i in range(20)]
    items = make_corrupted_set(docs, seed=5, limit=10)
    assert len(items) == 10
    for it in items:
        assert content(it.input) == content(it.target)
        derive_labels(it.input, it.target)
        assert 0.0 <= it.severity <= 1.0
    assert items == make_corrupted_set(docs, seed=5, limit=10)


def test_clean_set_items_are_identity_pairs() -> None:
    docs = [doc(i, body(i)) for i in range(20)]
    items = make_clean_set(docs, seed=5, n_passages=15)
    assert len(items) == 15
    for it in items:
        assert it.input == it.target and it.severity == 0.0
        assert 300 <= len(it.input) <= 800


def test_build_lexicon_counts_alpha_tokens() -> None:
    lx = build_lexicon([doc(1, "alpha alpha alpha beta beta 42 42 42")], min_count=3)
    assert lx.known("alpha") and not lx.known("beta") and not lx.known("42")


def test_manifest(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert file_sha256(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    write_manifest(tmp_path / "manifest.json", {"files": {"a.txt": file_sha256(f)}})
    assert (tmp_path / "manifest.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_build.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/data/build.py`**

```python
"""Assemble, split, and derive evaluation sets (design sections 3.2, 3.3, 5.1)."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence

from ..corrupt import corrupt
from ..lexicon import Lexicon
from ..text import split
from .filters import has_structure, is_hard_wrapped, long_enough
from .passages import cut_passages
from .records import CleanDoc, EvalItem
from .splits import assign_splits, dedupe


def assemble(docs: Iterable[CleanDoc], seed: int) -> dict[str, list[CleanDoc]]:
    kept = [
        d for d in docs
        if long_enough(d.clean) and has_structure(d.clean) and not is_hard_wrapped(d.clean)
    ]
    kept = dedupe(kept)
    side = assign_splits((d.group for d in kept), seed=seed)
    out: dict[str, list[CleanDoc]] = {"train": [], "val": [], "test": []}
    for d in kept:
        out[side[d.group]].append(d)
    return out


def make_corrupted_set(docs: Sequence[CleanDoc], seed: int, limit: int | None) -> list[EvalItem]:
    rng = random.Random(seed)
    chosen = list(docs) if limit is None else rng.sample(list(docs), min(limit, len(docs)))
    items: list[EvalItem] = []
    for d in chosen:
        c = corrupt(d.clean, random.Random(f"{seed}:{d.id}"))
        items.append(EvalItem(d.id, d.source, c.text, d.clean, c.severity, {"group": d.group}))
    return items


def make_clean_set(docs: Sequence[CleanDoc], seed: int, n_passages: int) -> list[EvalItem]:
    rng = random.Random(seed)
    pool: list[tuple[CleanDoc, int, str]] = []
    for d in docs:
        for k, p in enumerate(cut_passages(d.clean)):
            pool.append((d, k, p))
    chosen = rng.sample(pool, min(n_passages, len(pool)))
    return [
        EvalItem(f"{d.id}#{k}", d.source, p, p, 0.0, {"group": d.group}) for d, k, p in chosen
    ]


def build_lexicon(docs: Iterable[CleanDoc], min_count: int = 3) -> Lexicon:
    counts: Counter[str] = Counter()
    for d in docs:
        tokens, _ = split(d.clean)
        counts.update(t.lower() for t in tokens if t.isalpha())
    return Lexicon.from_counts(counts, min_count=min_count)
```

`random.Random(f"{seed}:{d.id}")` seeds from a string, which Python hashes deterministically for `random.Random` (it uses the string's bytes, not `hash()`), so items are reproducible per document.

- [ ] **Step 4: Implement `src/newline_fixer/data/manifest.py`**

```python
"""Dataset manifest with content hashes (design section 3.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, entries: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `make check`
Expected: all pass.

- [ ] **Step 6: Write `scripts/build_data.py`**

```python
"""Build the datasets. Usage:

  uv run python scripts/build_data.py wikipedia --n 20000 --seed 1
  uv run python scripts/build_data.py generate --n 2000 --seed 1 [--model ID]
  uv run python scripts/build_data.py assemble --seed 1
  uv run python scripts/build_data.py sets --seed 1
  uv run python scripts/build_data.py lexicon
  uv run python scripts/build_data.py publish --repo USER/newline-fixer-data

Outputs under data/ (see data/README.md for what is committed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from newline_fixer.corrupt import CorruptConfig
from newline_fixer.data.build import assemble, build_lexicon, make_clean_set, make_corrupted_set
from newline_fixer.data.generated import generate_docs
from newline_fixer.data.manifest import file_sha256, write_manifest
from newline_fixer.data.records import CleanDoc, EvalItem, read_jsonl, write_jsonl
from newline_fixer.data.wikipedia import iter_wikipedia, resolve_revision
from newline_fixer.example import EXAMPLE_INPUT, EXAMPLE_OUTPUT

DATA = Path("data")
RAW = DATA / "raw"
CLEAN = DATA / "clean"
SETS = DATA / "sets"
LEXICON = Path("src/newline_fixer/resources/lexicon.txt")
DATASET_VERSION = "1"


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def cmd_wikipedia(args: argparse.Namespace) -> None:
    revision = args.revision or resolve_revision()
    docs = list(iter_wikipedia(args.n, args.seed, revision))
    n = write_jsonl(RAW / "wikipedia.jsonl", docs)
    (RAW / "wikipedia.meta.json").write_text(json.dumps({"revision": revision, "n": n, "seed": args.seed}))
    print(f"wikipedia: {n} docs, revision {revision}")


def cmd_generate(args: argparse.Namespace) -> None:
    docs = generate_docs(args.n, args.seed, RAW / "generated", args.model)
    n = write_jsonl(RAW / "generated.jsonl", docs)
    (RAW / "generated.meta.json").write_text(
        json.dumps({"model": args.model, "n": n, "seed": args.seed, "date": dt.date.today().isoformat()})
    )
    print(f"generated: {n} docs")


def cmd_assemble(args: argparse.Namespace) -> None:
    docs = read_jsonl(RAW / "wikipedia.jsonl", CleanDoc) + read_jsonl(RAW / "generated.jsonl", CleanDoc)
    parts = assemble(docs, seed=args.seed)
    for side, ds in parts.items():
        write_jsonl(CLEAN / f"{side}.jsonl", ds)
        print(f"{side}: {len(ds)} docs")
    split = {side: sorted({d.group for d in ds}) for side, ds in parts.items()}
    (DATA / "split.json").write_text(json.dumps({"seed": args.seed, "groups": split}, indent=0) + "\n")


def cmd_sets(args: argparse.Namespace) -> None:
    val = read_jsonl(CLEAN / "val.jsonl", CleanDoc)
    test = read_jsonl(CLEAN / "test.jsonl", CleanDoc)
    write_jsonl(SETS / "V1.jsonl", make_corrupted_set(val, args.seed, limit=500))
    write_jsonl(SETS / "T1.jsonl", make_corrupted_set(test, args.seed + 1, limit=500))
    write_jsonl(SETS / "V3.jsonl", make_clean_set(val, args.seed, n_passages=100))
    write_jsonl(SETS / "T3.jsonl", make_clean_set(test, args.seed + 1, n_passages=200))
    write_jsonl(SETS / "T0.jsonl", [EvalItem("readme-example", "challenge", EXAMPLE_INPUT, EXAMPLE_OUTPUT, 1.0, {})])
    (SETS / "meta.json").write_text(
        json.dumps({"seed": args.seed, "corruptor": asdict(CorruptConfig()), "git_commit": git_commit()}, indent=2) + "\n"
    )
    print("sets written:", sorted(p.name for p in SETS.iterdir()))


def cmd_lexicon(args: argparse.Namespace) -> None:
    train = read_jsonl(CLEAN / "train.jsonl", CleanDoc)
    lx = build_lexicon(train)
    lx.write(LEXICON)
    print(f"lexicon: {len(lx)} words -> {LEXICON}")


def cmd_publish(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    files = [*CLEAN.glob("*.jsonl"), *SETS.glob("*.jsonl"), *RAW.glob("*.meta.json"), DATA / "split.json"]
    write_manifest(
        DATA / "manifest.json",
        {
            "dataset_version": DATASET_VERSION,
            "git_commit": git_commit(),
            "built": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "files": {str(p.relative_to(DATA)): file_sha256(p) for p in files},
        },
    )
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(DATA), repo_id=args.repo, repo_type="dataset",
        allow_patterns=["clean/*.jsonl", "sets/*.jsonl", "raw/*.meta.json", "raw/generated/*.txt", "split.json", "manifest.json", "README.md"],
        commit_message=f"dataset v{DATASET_VERSION} from {git_commit()[:12]}",
    )
    print(f"published to https://huggingface.co/datasets/{args.repo}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wikipedia"); w.add_argument("--n", type=int, default=20000); w.add_argument("--seed", type=int, default=1); w.add_argument("--revision")
    g = sub.add_parser("generate"); g.add_argument("--n", type=int, default=2000); g.add_argument("--seed", type=int, default=1); g.add_argument("--model", default="claude-haiku-4-5-20251001")
    a = sub.add_parser("assemble"); a.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("sets"); s.add_argument("--seed", type=int, default=1)
    sub.add_parser("lexicon")
    pub = sub.add_parser("publish"); pub.add_argument("--repo", required=True)
    args = p.parse_args()
    {"wikipedia": cmd_wikipedia, "generate": cmd_generate, "assemble": cmd_assemble,
     "sets": cmd_sets, "lexicon": cmd_lexicon, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    main()
```

Reformat the one-line `add_parser` chains with `make fmt`; ruff will split them.

- [ ] **Step 7: Run the pipeline**

```bash
export ANTHROPIC_API_KEY=...   # for generate
uv run python scripts/build_data.py wikipedia --n 20000 --seed 1
uv run python scripts/build_data.py generate --n 2000 --seed 1
uv run python scripts/build_data.py assemble --seed 1
uv run python scripts/build_data.py sets --seed 1
uv run python scripts/build_data.py lexicon
make check
```

Expected: `data/clean/train.jsonl` with roughly 19,000 documents, `val` and `test` with roughly 1,000 each; five files under `data/sets/`; the lexicon resource replaced with tens of thousands of words; all tests still pass with the new lexicon. If Wikipedia streaming is slow, start with `--n 5000` and record the number in the manifest; the design's quantities are targets, not requirements.

Inspect ten random items from `V1.jsonl` by eye and confirm they resemble the challenge example. Inspect ten `generated` documents and confirm they have headings, lists and paragraphs.

- [ ] **Step 8: Publish**

```bash
huggingface-cli login   # once
uv run python scripts/build_data.py publish --repo <your-hf-user>/newline-fixer-data
```

Record the dataset URL in `data/README.md`.

- [ ] **Step 9: Commit**

```bash
git add scripts/build_data.py src/newline_fixer/data/build.py src/newline_fixer/data/manifest.py tests/test_build.py \
        src/newline_fixer/resources/lexicon.txt data/split.json data/sets data/README.md
git commit -m "data: build pipeline, dev and test sets v1, lexicon from the training split"
```

`data/sets/meta.json` is included by `data/sets`.

---

### Task 12: Realistic sets V2 and T2

**Files:**
- Create: `src/newline_fixer/data/realistic.py`, `scripts/make_realistic_set.py`, `tests/test_realistic.py`, `data/realistic/sources.json`, `data/realistic/review.json`
- Produce (committed): `data/realistic/<doc>/<nn>.raw.txt`, `.input.txt`, `.target.txt`; `data/sets/V2.jsonl`, `data/sets/T2.jsonl`

**Interfaces:**
- Produces:
  - `join_hyphenation(raw: str) -> tuple[str, int]` (adjusted text, number of joins)
  - `cut_raw_passages(raw: str, rng: random.Random, per_doc: int = 8, min_chars: int = 300, max_chars: int = 800) -> list[str]`
  - `propose_target(input_text: str, model: str, client: object | None = None) -> str` (raises if content differs after three attempts)
  - `validate_pair(input_text: str, target: str) -> dict[str, int]` with keys `unreachable`, `gaps`
  - `load_reviewed(root: Path, role: str) -> list[EvalItem]` (only passages listed as reviewed in `review.json`)
  - CLI: `uv run python scripts/make_realistic_set.py {extract,cut,propose,build}`

Requires `pdftotext` from poppler (`brew install poppler`).

- [ ] **Step 1: Choose and record the sources** in `data/realistic/sources.json`

```json
{
  "dev": [
    {"doc": "word2vec", "title": "Efficient Estimation of Word Representations in Vector Space", "url": "https://arxiv.org/pdf/1301.3781"},
    {"doc": "fasttext", "title": "Bag of Tricks for Efficient Text Classification", "url": "https://arxiv.org/pdf/1607.01759"},
    {"doc": "nist-800-63", "title": "NIST SP 800-63-3 Digital Identity Guidelines", "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63-3.pdf"},
    {"doc": "python-tutorial", "title": "The Python Tutorial (PDF build)", "url": "https://docs.python.org/3/archives/python-3.12-docs-pdf-a4.zip"}
  ],
  "test": [
    {"doc": "attention", "title": "Attention Is All You Need", "url": "https://arxiv.org/pdf/1706.03762"},
    {"doc": "bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "url": "https://arxiv.org/pdf/1810.04805"},
    {"doc": "resnet", "title": "Deep Residual Learning for Image Recognition", "url": "https://arxiv.org/pdf/1512.03385"},
    {"doc": "adam", "title": "Adam: A Method for Stochastic Optimization", "url": "https://arxiv.org/pdf/1412.6980"},
    {"doc": "nist-ai-rmf", "title": "NIST AI Risk Management Framework 1.0", "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"},
    {"doc": "gnu-make", "title": "GNU Make Manual", "url": "https://www.gnu.org/software/make/manual/make.pdf"}
  ]
}
```

Substitutes are fine; keep the dev and test lists disjoint and record any change here.

- [ ] **Step 2: Write the failing tests**

```python
import random
from pathlib import Path
from types import SimpleNamespace

from newline_fixer.data.realistic import (
    cut_raw_passages,
    join_hyphenation,
    load_reviewed,
    propose_target,
    validate_pair,
)

RAW = (
    "3.2.3 Applications of Attention in our Model\n"
    "The Transformer uses multi-head attention in three different ways:\n"
    "• In \"encoder-decoder attention\" layers, the que-\n"
    "ries come from the previous decoder layer, and the memory keys and val-\n"
    "ues come from the output of the encoder.\n"
)


def test_join_hyphenation() -> None:
    out, n = join_hyphenation(RAW)
    assert n == 2
    assert "que\nries" in out and "val\nues" in out
    assert "state-of-the-art" in join_hyphenation("a state-of-the-art\nresult")[0]


def test_cut_raw_passages_bounds() -> None:
    raw = "\f".join(("Para %d. " % i + "text " * 60 + "\n\n") * 3 for i in range(10))
    out = cut_raw_passages(raw, random.Random(0), per_doc=4)
    assert 1 <= len(out) <= 4
    for p in out:
        assert 300 <= len(p) <= 800
        assert "\f" not in p


def test_validate_pair_counts() -> None:
    assert validate_pair("a b\nc", "a\nb c") == {"unreachable": 0, "gaps": 2}
    assert validate_pair("a.B", "a.\n\nB")["unreachable"] == 1


def test_propose_target_rejects_content_change() -> None:
    bad = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: SimpleNamespace(content=[SimpleNamespace(text="changed words")])))
    import pytest

    with pytest.raises(RuntimeError):
        propose_target("some words", "m", client=bad)
    good = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: SimpleNamespace(content=[SimpleNamespace(text="some\nwords")])))
    assert propose_target("some words", "m", client=good) == "some\nwords"


def test_load_reviewed_only_returns_reviewed(tmp_path: Path) -> None:
    import json

    d = tmp_path / "doc1"
    d.mkdir()
    for nn, ok in (("00", True), ("01", False)):
        (d / f"{nn}.raw.txt").write_text("a b")
        (d / f"{nn}.input.txt").write_text("a b")
        (d / f"{nn}.target.txt").write_text("a\nb")
    (tmp_path / "sources.json").write_text(json.dumps({"dev": [{"doc": "doc1", "title": "t", "url": "u"}], "test": []}))
    (tmp_path / "review.json").write_text(json.dumps({"doc1/00": {"reviewer": "JH", "date": "2026-10-02", "note": ""}}))
    items = load_reviewed(tmp_path, "dev")
    assert [i.id for i in items] == ["doc1/00"]
    assert items[0].meta["unreachable"] == 0
    assert load_reviewed(tmp_path, "test") == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_realistic.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `src/newline_fixer/data/realistic.py`**

```python
"""Realistic passages from real PDF extractions (design section 5.1)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from ..text import content, normalize, split, unreachable_count
from .records import EvalItem

_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n[ \t]*(?=[a-z])")


def join_hyphenation(raw: str) -> tuple[str, int]:
    """`que-\\nries` -> `que\\nries`. Keeps the break so the service must still join it."""
    out, n = _HYPHEN_BREAK.subn("\n", raw)
    return out, n


def cut_raw_passages(
    raw: str, rng: random.Random, per_doc: int = 8, min_chars: int = 300, max_chars: int = 800
) -> list[str]:
    """Cut on page breaks and blank lines, then take passages within bounds at random."""
    chunks: list[str] = []
    for page in raw.split("\f"):
        buf: list[str] = []
        for block in re.split(r"\n[ \t]*\n", page):
            block = block.strip("\n")
            if not block.strip():
                continue
            candidate = "\n\n".join([*buf, block])
            if len(candidate) > max_chars:
                if len("\n\n".join(buf)) >= min_chars:
                    chunks.append("\n\n".join(buf))
                buf = [block] if len(block) <= max_chars else []
            else:
                buf.append(block)
        if len("\n\n".join(buf)) >= min_chars:
            chunks.append("\n\n".join(buf))
    chunks = [c.strip("\n") for c in chunks if min_chars <= len(c.strip("\n")) <= max_chars]
    return rng.sample(chunks, min(per_doc, len(chunks)))


PROPOSE_PROMPT = (
    "The text below was extracted from a PDF and its line breaks are wrong. Rewrite it with "
    "correct newlines: one blank line between paragraphs and after headings, a single newline "
    "before each list item, no line breaks inside sentences, and split words joined. Change "
    "ONLY whitespace. Every non-whitespace character must stay exactly as it is, in the same "
    "order. Output only the corrected text.\n\n"
)


def propose_target(input_text: str, model: str, client: Any | None = None, attempts: int = 3) -> str:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    for _ in range(attempts):
        msg = client.messages.create(
            model=model, max_tokens=2000, messages=[{"role": "user", "content": PROPOSE_PROMPT + input_text}]
        )
        text = normalize(msg.content[0].text)
        if content(text) == content(input_text):
            return text
    raise RuntimeError("proposal changed non-whitespace content three times")


def validate_pair(input_text: str, target: str) -> dict[str, int]:
    _, gaps = split(input_text)
    return {"unreachable": unreachable_count(input_text, target), "gaps": len(gaps)}


def load_reviewed(root: Path, role: str) -> list[EvalItem]:
    sources = json.loads((root / "sources.json").read_text())
    review = json.loads((root / "review.json").read_text()) if (root / "review.json").exists() else {}
    items: list[EvalItem] = []
    for src in sources[role]:
        doc_dir = root / src["doc"]
        if not doc_dir.exists():
            continue
        for inp in sorted(doc_dir.glob("*.input.txt")):
            nn = inp.name.split(".")[0]
            key = f"{src['doc']}/{nn}"
            if key not in review:
                continue
            raw = (doc_dir / f"{nn}.raw.txt").read_text()
            input_text = normalize(inp.read_text())
            target = normalize((doc_dir / f"{nn}.target.txt").read_text())
            if content(input_text) != content(target):
                raise ValueError(f"{key}: target changes non-whitespace content")
            stats = validate_pair(input_text, target)
            _, adjustments = join_hyphenation(raw)
            items.append(
                EvalItem(
                    key, "realistic", input_text, target, 1.0,
                    {"doc": src["doc"], "adjustments": adjustments, **stats, **review[key]},
                )
            )
    return items
```

- [ ] **Step 5: Write `scripts/make_realistic_set.py`**

```python
"""Build the realistic sets. Usage:

  uv run python scripts/make_realistic_set.py extract --doc attention --pdf ~/Downloads/1706.03762.pdf
  uv run python scripts/make_realistic_set.py cut --doc attention --seed 1
  uv run python scripts/make_realistic_set.py propose --doc attention [--model ID]
  # review: edit data/realistic/<doc>/<nn>.target.txt, then add "<doc>/<nn>" to review.json
  uv run python scripts/make_realistic_set.py build
"""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

from newline_fixer.data.realistic import cut_raw_passages, join_hyphenation, load_reviewed, propose_target
from newline_fixer.data.records import write_jsonl

ROOT = Path("data/realistic")
SETS = Path("data/sets")


def cmd_extract(args: argparse.Namespace) -> None:
    out = ROOT / args.doc / "full.raw.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftotext", "-enc", "UTF-8", args.pdf, str(out)], check=True)
    print(f"extracted {out} ({out.stat().st_size} bytes)")


def cmd_cut(args: argparse.Namespace) -> None:
    doc_dir = ROOT / args.doc
    raw = (doc_dir / "full.raw.txt").read_text()
    for i, passage in enumerate(cut_raw_passages(raw, random.Random(args.seed))):
        (doc_dir / f"{i:02d}.raw.txt").write_text(passage)
        adjusted, _ = join_hyphenation(passage)
        (doc_dir / f"{i:02d}.input.txt").write_text(adjusted)
    (doc_dir / "full.raw.txt").unlink()  # the full document is not committed
    print(f"cut {args.doc}: {len(list(doc_dir.glob('*.input.txt')))} passages")


def cmd_propose(args: argparse.Namespace) -> None:
    for inp in sorted((ROOT / args.doc).glob("*.input.txt")):
        target = inp.with_name(inp.name.replace(".input.", ".target."))
        if target.exists():
            continue
        target.write_text(propose_target(inp.read_text(), args.model))
        print(f"proposed {target}")


def cmd_build(args: argparse.Namespace) -> None:
    for role, name in (("dev", "V2"), ("test", "T2")):
        items = load_reviewed(ROOT, role)
        write_jsonl(SETS / f"{name}.jsonl", items)
        unreachable = sum(int(i.meta["unreachable"]) for i in items)  # type: ignore[call-overload]
        print(f"{name}: {len(items)} reviewed passages, {unreachable} unreachable boundaries")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--doc", required=True); e.add_argument("--pdf", required=True)
    c = sub.add_parser("cut"); c.add_argument("--doc", required=True); c.add_argument("--seed", type=int, default=1)
    pr = sub.add_parser("propose"); pr.add_argument("--doc", required=True); pr.add_argument("--model", default="claude-sonnet-5-5")
    sub.add_parser("build")
    args = p.parse_args()
    {"extract": cmd_extract, "cut": cmd_cut, "propose": cmd_propose, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
```

Confirm the Sonnet model id with the `claude-api` skill before running `propose`.

- [ ] **Step 6: Run tests, then build the sets**

Run: `make check`. Then for each of the ten documents: download the PDF, `extract`, `cut`, `propose`. Then review: open each `.target.txt` next to its `.input.txt`, correct it, and add the key to `data/realistic/review.json` with your initials and the date. Only reviewed passages enter the sets. Then `build`.

Expected: V2 with 25 to 35 passages, T2 with 35 to 50. The build prints unreachable counts; record them in `data/README.md`.

- [ ] **Step 7: Commit**

```bash
git add src/newline_fixer/data/realistic.py scripts/make_realistic_set.py tests/test_realistic.py data/realistic data/sets/V2.jsonl data/sets/T2.jsonl data/README.md
git commit -m "data: realistic dev and test sets from real PDF extractions, hand-reviewed"
```

---

### Task 13: Evaluation runner, results table, first results, B1 frozen

**Files:**
- Create: `src/newline_fixer/models/registry.py`, `src/newline_fixer/eval/runner.py`, `src/newline_fixer/eval/table.py`, `scripts/evaluate.py`, `scripts/results_table.py`, `tests/test_runner.py`, `docs/decisions/0006-rules-baseline-frozen.md`
- Produce (committed): `experiments/results/m1-baselines.json`, `experiments/README.md`

**Interfaces:**
- Produces:
  - `get_fixer(name: str) -> Fixer` for names `identity`, `rules`; `FIXER_NAMES: list[str]`
  - `evaluate_set(fixer: Fixer, items: Sequence[EvalItem]) -> dict[str, object]` with keys `gap` (GapMetrics dict), `paragraph_match_rate`, `string_changed_vs_raw`, `string_changed_vs_normalized`, `by_severity` (band -> GapMetrics dict, only when severities vary)
  - `run(fixers: Sequence[str], sets: Sequence[str], sets_dir: Path) -> dict[str, object]`
  - `render_table(results: dict[str, object]) -> str` (Markdown)

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from newline_fixer.data.records import EvalItem, write_jsonl
from newline_fixer.eval.runner import evaluate_set, run
from newline_fixer.eval.table import render_table
from newline_fixer.models.registry import FIXER_NAMES, get_fixer

ITEMS = [
    EvalItem("1", "t", "a b\nc", "a b c", 0.2, {}),
    EvalItem("2", "t", "x\n\ny", "x\n\ny", 0.0, {}),
    EvalItem("3", "t", "Heading Body text here.", "Heading\n\nBody text here.", 0.9, {}),
]


def test_registry() -> None:
    assert set(FIXER_NAMES) >= {"identity", "rules"}
    assert get_fixer("identity").name == "identity"


def test_evaluate_set_identity() -> None:
    res = evaluate_set(get_fixer("identity"), ITEMS)
    gap = res["gap"]
    assert isinstance(gap, dict) and gap["n_items"] == 3 and gap["n_gaps"] == 6
    assert res["string_changed_vs_normalized"] == 0.0
    assert res["string_changed_vs_raw"] == 0.0
    assert set(res["by_severity"]) == {"0", "(0,0.33]", "(0.33,0.66]", "(0.66,1]"}  # type: ignore[arg-type]
    assert 0.0 <= float(res["paragraph_match_rate"]) <= 1.0


def test_run_reads_sets_and_writes_structure(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "V9.jsonl", ITEMS)
    out = run(["identity", "rules"], ["V9"], tmp_path)
    assert set(out["systems"]) == {"identity", "rules"}  # type: ignore[arg-type]
    assert "V9" in out["systems"]["identity"]  # type: ignore[index]
    md = render_table(out)
    assert "| V9" in md or "V9" in md
    assert "macro_f1" in md.lower() or "macro-F1" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/newline_fixer/models/registry.py`**

```python
"""Name -> fixer. Learned models register here in later milestones."""

from __future__ import annotations

from collections.abc import Callable

from .base import Fixer
from .identity import IdentityFixer


def _rules() -> Fixer:
    from ..rules import RulesFixer

    return RulesFixer()


_REGISTRY: dict[str, Callable[[], Fixer]] = {"identity": IdentityFixer, "rules": _rules}
FIXER_NAMES = sorted(_REGISTRY)


def get_fixer(name: str) -> Fixer:
    try:
        return _REGISTRY[name]()
    except KeyError as e:
        raise ValueError(f"unknown fixer {name!r}; known: {FIXER_NAMES}") from e
```

- [ ] **Step 4: Implement `src/newline_fixer/eval/runner.py`**

```python
"""Run systems over evaluation sets (design section 5)."""

from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ..data.records import EvalItem, read_jsonl
from ..models.base import Fixer
from ..models.registry import get_fixer
from ..text import derive_labels, normalize, split
from ..windows import fix
from .metrics import GapMetrics, paragraph_match

BANDS = (("0", 0.0, 0.0), ("(0,0.33]", 0.0, 0.33), ("(0.33,0.66]", 0.33, 0.66), ("(0.66,1]", 0.66, 1.0))


def _band(severity: float) -> str:
    if severity == 0.0:
        return "0"
    for name, lo, hi in BANDS[1:]:
        if lo < severity <= hi:
            return name
    return BANDS[-1][0]


def evaluate_set(fixer: Fixer, items: Sequence[EvalItem]) -> dict[str, object]:
    overall = GapMetrics()
    by_band = {name: GapMetrics() for name, _, _ in BANDS}
    matched = total = 0
    changed_raw = changed_norm = 0
    for it in items:
        tokens, current = split(it.input)
        ref = derive_labels(it.input, it.target)
        result = fix(it.input, fixer)
        _, pred = split(result.text)
        if len(pred) != len(ref):
            raise RuntimeError(f"{fixer.name} changed the token count on item {it.id}")
        overall.update(pred, ref, current)
        by_band[_band(it.severity)].update(pred, ref, current)
        m, t = paragraph_match(result.text, normalize(it.target))
        matched += m
        total += t
        changed_raw += result.text != it.input
        changed_norm += result.text != normalize(it.input)
    n = len(items)
    return {
        "gap": overall.to_dict(),
        "paragraph_match_rate": matched / total if total else 0.0,
        "string_changed_vs_raw": changed_raw / n if n else 0.0,
        "string_changed_vs_normalized": changed_norm / n if n else 0.0,
        "by_severity": {k: v.to_dict() for k, v in by_band.items()},
    }


def run(fixers: Sequence[str], sets: Sequence[str], sets_dir: Path) -> dict[str, object]:
    systems: dict[str, dict[str, object]] = {}
    for name in fixers:
        fixer = get_fixer(name)
        systems[name] = {s: evaluate_set(fixer, read_jsonl(sets_dir / f"{s}.jsonl", EvalItem)) for s in sets}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "sets": list(sets),
        "systems": systems,
    }
```

- [ ] **Step 5: Implement `src/newline_fixer/eval/table.py`**

```python
"""Render evaluation results as Markdown."""

from __future__ import annotations

from typing import Any


def render_table(results: dict[str, Any]) -> str:
    systems: dict[str, dict[str, Any]] = results["systems"]
    sets: list[str] = results["sets"]
    lines = [
        f"Results at commit `{results.get('git_commit', '')[:12]}`, {results.get('run_at', '')}.",
        "",
        "| set | system | gaps | macro-F1 | classes | break-F1 | JOIN F1 | PARA F1 | wrong-join /1k | damage | str≠raw | str≠norm | para match |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sets:
        for name, per_set in systems.items():
            r = per_set[s]
            g = r["gap"]
            pc = g["per_class"]
            lines.append(
                f"| {s} | {name} | {g['n_gaps']} | {g['macro_f1']:.3f} | {','.join(g['macro_classes'])} | "
                f"{g['break_f1']:.3f} | {pc['JOIN']['f1']:.3f} | {pc['PARA']['f1']:.3f} | "
                f"{g['wrong_join_per_1000']:.2f} | {g['damage_rate']:.4f} | "
                f"{r['string_changed_vs_raw']:.3f} | {r['string_changed_vs_normalized']:.3f} | "
                f"{r['paragraph_match_rate']:.3f} |"
            )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 6: Write the scripts**

`scripts/evaluate.py`:

```python
"""Evaluate systems on sets. Usage:
  uv run python scripts/evaluate.py --systems identity,rules --sets V1,V2,V3 --out experiments/results/m1-baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newline_fixer.eval.runner import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", required=True)
    p.add_argument("--sets", required=True)
    p.add_argument("--sets-dir", default="data/sets")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    results = run(a.systems.split(","), a.sets.split(","), Path(a.sets_dir))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

`scripts/results_table.py`:

```python
"""Render experiments/results/*.json into experiments/README.md."""

from __future__ import annotations

import json
from pathlib import Path

from newline_fixer.eval.table import render_table


def main() -> None:
    parts = ["# Experiments\n", "Rendered by `scripts/results_table.py`; do not edit by hand.\n"]
    for path in sorted(Path("experiments/results").glob("*.json")):
        parts.append(f"\n## {path.stem}\n\n" + render_table(json.loads(path.read_text())))
    Path("experiments/README.md").write_text("\n".join(parts))
    print("wrote experiments/README.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run tests, then the first evaluation**

```bash
make check
uv run python scripts/evaluate.py --systems identity,rules --sets V1,V2,V3 --out experiments/results/m1-baselines.json
uv run python scripts/results_table.py
```

Read the table. Checks that must hold before continuing: identity has `damage 0.0000` and `str≠norm 0.000` on V3; rules has a lower wrong-join rate than one per thousand on V3 or the lexicon threshold is reconsidered; rules beats identity on V1 and V2 macro-F1. If rules is worse than identity anywhere, inspect twenty disagreements by printing `it.input`, the rules output and `it.target`, adjust the rule thresholds (`MAX_HEADING_TOKENS`, `STOP`, `min_count`), re-run, and record what was changed in the decision record below.

- [ ] **Step 8: Write `docs/decisions/0006-rules-baseline-frozen.md`**

```markdown
# 0006. The rules baseline is frozen

Date: <date>. Status: accepted.

## Context

Design 4.3 defines B1 and says it is checked on V1 and V2, then frozen so later
improvements go into the models and the comparison stays honest.

## Options

1. Keep tuning B1 alongside the models.
2. Freeze B1 at the first version that beats identity on V1 and V2 and keeps wrong joins
   under one per thousand gaps on V3.

## Decision

Option 2. B1 is frozen at commit `<commit>`. Numbers on the dev sets at that commit:

| set | macro-F1 identity | macro-F1 rules | wrong-join /1k rules | damage rules (V3) |
|---|---|---|---|---|
| V1 | <n> | <n> | <n> | |
| V2 | <n> | <n> | <n> | |
| V3 | | | <n> | <n> |

Changes made while checking: <list, or "none">.

## Consequences

- Any later change to `rules.py` needs a new decision record and re-runs every table.
- The clean-damage threshold for the serving rule (design 5.3) is set from the rules
  number on V3: <value>.
```

Fill in every placeholder from the table before committing.

- [ ] **Step 9: Commit**

```bash
git add src/newline_fixer/models/registry.py src/newline_fixer/eval scripts/evaluate.py scripts/results_table.py tests/test_runner.py experiments docs/decisions/0006-rules-baseline-frozen.md
git commit -m "feat: evaluation runner and results table; first baseline results; B1 frozen"
```

---

### Task 14: CI workflow and documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`, `data/README.md`

- [ ] **Step 1: Write the workflow**

```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv python install 3.12
      - run: uv sync --all-extras
      - run: make check
```

Check the current action versions on the GitHub marketplace before committing; replace `@v4` and `@v3` with the current majors.

- [ ] **Step 2: Update `README.md`**

Add under "Development":

```markdown
Rebuild data: see the docstring of `scripts/build_data.py`. Evaluate:

```bash
uv run python scripts/evaluate.py --systems identity,rules --sets V1,V2,V3 --out experiments/results/dev.json
uv run python scripts/results_table.py
```

Results tables live in `experiments/README.md`.
```

- [ ] **Step 3: Update `data/README.md`** with the dataset URL, the counts per split, the set sizes, and the unreachable counts from Task 12.

- [ ] **Step 4: Run `make check` and commit**

```bash
git add .github README.md data/README.md
git commit -m "chore: CI workflow; document data and evaluation commands"
```

M1 is complete when this commit is on `main`, `experiments/README.md` shows identity and rules on V1, V2 and V3, and decision record 0006 is filled in.
