# Design

Status: v1, 2026-10-01. Builds on `01-requirements.md`. Says how the service is built.
Decisions with real alternatives are recorded in `decisions/`.

## 1. Overview

The service repairs whitespace. It splits the input into non-whitespace tokens and the gaps
between them, predicts one of four classes per gap, and re-joins the tokens with the
predicted gaps. Four systems implement the same interface and are measured on the same
test sets:

| Id | System | Learned | Purpose |
|---|---|---|---|
| B0 | Identity | no | floor; exposes class imbalance |
| B1 | Rules | no | credible non-ML baseline the models must beat |
| M1 | From-scratch BiLSTM with character features | yes | a model fully owned and explainable |
| M2 | Fine-tuned small cased pretrained encoder | yes | the expected quality leader |

The HTTP service can serve any of them, selected by an environment variable.

## 2. Problem formulation

### 2.1 Tokens and gaps

- A **token** is a maximal run of non-whitespace characters (`\S+` in Python, Unicode aware).
- A **gap** is the whitespace between two consecutive tokens. Leading and trailing
  whitespace of the input is discarded; output has none.
- Gap classes, in both input and output:

  | Class | Output string | Meaning |
  |---|---|---|
  | `JOIN` | `""` | the two tokens are one word that was split |
  | `SPACE` | `" "` | same line |
  | `NL` | `"\n"` | line break (list item, heading line, address line) |
  | `PARA` | `"\n\n"` | paragraph break |

- **Normalization** maps any whitespace run to a class: a run containing two or more
  newlines is `PARA`, exactly one newline is `NL`, otherwise `SPACE`. Tabs and other
  Unicode whitespace count as whitespace. Spaces around a newline are ignored, so
  `"\n  "` is `NL`. Clean reference text is normalized before use, so the target is
  always expressible in the four classes.
- In the input, a gap always contains whitespace, so its current class is one of
  `SPACE`, `NL`, `PARA`. The current class is an input feature for every model.
- **Invariant.** The non-whitespace character sequence of the output equals that of the
  input. Reconstruction enforces it; a test checks it for every system.

### 2.2 Label derivation

Given clean text C and corrupted text X with equal non-whitespace character sequences:

1. For C, compute `after[i]`, the normalized gap class following non-whitespace character
   `i`, with `JOIN` when the next character is also non-whitespace.
2. Walk the tokens of X. The gap after the token that ends at cumulative non-whitespace
   character count `n` gets label `after[n-1]`.

This is exact, needs no alignment heuristics, and fails loudly if the invariant is broken.

### 2.3 Known limitation

A break that was deleted without leaving any whitespace (`ways:•In`) is not a gap and
cannot be repaired. The requirements put that out of scope. Hyphenated line breaks are
also out of scope; the realistic test set joins them before labelling (section 5.1).

## 3. Data

### 3.1 Clean sources

| Source | What it contributes | Notes |
|---|---|---|
| Wikitext-103 raw | paragraphs, section headings | heading markup `= Title =` converted to a plain heading line |
| Wikipedia (Hugging Face dump, English) | paragraphs, section titles, some lists | sampled, not the whole dump |
| LLM-generated documents | numbered headings, bullet lists with `•`, `-`, `*`, `1.`, mixed registers: papers, manuals, reports, emails, notes | written to a brief; a few thousand documents; cost in the low dollars |

Hard-wrapped documents are excluded by a heuristic: if most lines end without terminal
punctuation and the next line starts lowercase, the document is dropped. Documents shorter
than 200 characters are dropped.

### 3.2 Normalization, deduplication, split

- Normalize whitespace to the four classes. Strip each line.
- Deduplicate on a hash of the normalized text and on the first 200 characters.
- Split by document, 90/5/5 into train, validation, test, with a fixed seed, before any
  corruption. The split file is committed so it can be audited.
- Store as JSONL: `{id, source, clean}`. Corrupted variants are produced on the fly or
  materialized once with a recorded seed: `{id, source, clean, corrupted, severity}`.

### 3.3 Corruptor

Seeded, parametrized, per document:

1. One document in ten is left uncorrupted (`severity = 0`).
2. Otherwise sample `severity` uniformly in (0, 1].
3. **Remove true breaks.** Each `NL` or `PARA` gap becomes `SPACE` with probability
   `0.6 + 0.4 * severity`.
4. **Insert spurious newlines.** Expected one insertion per `L` characters, with `L`
   drawn from [40, 200] and scaled by severity. Each insertion picks a uniform character
   position:
   - inside a word: insert `"\n"` with probability 0.4, splitting the word; otherwise skip;
   - at or next to whitespace: replace the run with one of `"\n"`, `"\n "`, `" \n"` chosen
     uniformly, which produces the leading-space pattern seen in the challenge;
   - with probability 0.1 the inserted newline is `"\n\n"`, so wrong paragraph breaks also
     occur.
5. Record the parameters and seed with the dataset version.

The corruptor's shape follows the assumption in the requirements. The realistic test set
exists to check that assumption.

### 3.4 Training examples

Documents are cut into windows of at most 256 tokens on gap boundaries. Each example is a
list of tokens, their current gap classes, and the target gap classes.

## 4. Models

### 4.1 Common interface

```
class Fixer(Protocol):
    name: str
    def predict(self, tokens: list[str], current: list[GapClass]) -> list[GapClass]: ...
```

`fix(text) -> str` is one function for all systems: tokenize, window, predict, merge
window predictions, reconstruct. Windows are 256 tokens with stride 128. A gap inside two
windows takes the prediction from the window whose center is nearer. The first and last
window are included unchanged, so no gap is left unpredicted.

### 4.2 B0, identity

Returns the current classes.

### 4.3 B1, rules

Applied per gap, first match wins:

1. Current `NL` or `PARA`, previous character and next character are both lowercase
   letters: `JOIN`.
2. Current `NL` or `PARA`, next token starts with a lowercase letter or with closing
   punctuation: `SPACE`.
3. Next token is a list marker (`•`, `-`, `*`, `–`, or `\d+[.)]`) and the previous token
   ends with `:` or terminal punctuation: `NL`.
4. The line ending at this gap looks like a heading: starts with a section number such
   as `3.2.3`, or is at most eight tokens, title-cased, with no terminal punctuation: `PARA`.
5. Otherwise keep the current class.

The rule set is frozen once written; improvements go into the models, not the baseline.

### 4.4 M1, from scratch

- Word embedding (vocabulary of the 30k most frequent lowercased training tokens plus
  unknown), dimension 128.
- Character CNN per token: character embedding 32, 64 filters of width 3, max-pooled.
  Captures case, digits, punctuation and partial words such as `que`.
- Current gap class embedding, dimension 8, attached to the token before the gap.
- Two-layer bidirectional LSTM, hidden 192 per direction, dropout 0.2.
- Classifier on the concatenation of the hidden states on both sides of the gap and the
  gap embedding: one hidden layer of 256, then four logits.
- About five million parameters. Cross-entropy loss. AdamW, learning rate 2e-3, batch 32
  windows, up to eight epochs, early stopping on validation macro-F1.
- Trains on the M1 Mac (MPS when available, CPU otherwise).

### 4.5 M2, fine-tuned pretrained encoder

- `AutoModelForTokenClassification` from Hugging Face Transformers.
- Candidates: `microsoft/deberta-v3-xsmall` and `distilbert-base-cased`. Uncased models
  are excluded because capitalization of the next token is the main paragraph cue.
- Selection: train each candidate for one epoch on a 20k-window subset, measure validation
  macro-F1 and CPU latency per 256-token window, choose by F1 subject to latency being
  within three times that of M1. Recorded as a decision record with the numbers.
- Current whitespace is exposed by inserting marker words `[NL]` and `[PP]` between tokens
  whose current gap is `NL` or `PARA`. The markers are added as special tokens; the
  embedding matrix is resized. Markers carry no label.
- Labels sit on the first subword of each real token and predict the gap after it. Other
  subwords and markers get the ignore index.
- Learning rate 3e-5 to 5e-5, batch 16, up to three epochs, mixed precision on a free
  Colab GPU, checkpoints saved to Drive. Early stopping on validation macro-F1.
- Window size is validated against the 512 subword limit; if a 256-token window exceeds
  it, the window is shortened for that model.

### 4.6 Experiment tracking

Each training run writes `experiments/<run-id>.json` with git commit, dataset version,
seed, config, training curve summary and validation metrics. A script renders
`experiments/README.md` as a table from those files.

## 5. Evaluation

### 5.1 Test sets

| Id | Set | Size | Built from |
|---|---|---|---|
| T1 | synthetic held-out | about 500 documents | the test split, corrupted at recorded severities, reported overall and by severity band |
| T2 | realistic | 50 to 80 passages of 300 to 800 characters | `pdftotext` output of about ten real PDFs (papers, a manual, a report), including the Transformer paper; targets proposed by an LLM and reviewed by hand |
| T3 | clean | about 200 passages | untouched clean documents from the test split and the T2 targets |
| T0 | the challenge example | 1 | the README; a unit test |

For T2, line-end hyphenations in the raw extraction (`que-\nries`) are joined to
`que\nries` before labelling so the content invariant holds. The report states this.

### 5.2 Metrics

Gap level, for every system on every set:

- precision, recall, F1 per class, and macro-F1;
- headline F1 for newline-versus-none (`NL` or `PARA` against `JOIN` or `SPACE`);
- on T3, false-edit rate: fraction of gaps changed, and fraction of passages left
  untouched;
- paragraph exact-match rate: fraction of reference paragraphs reproduced exactly.

Service level, for M1 and M2 and B1:

- model size on disk, resident memory after warm-up;
- p50 and p95 latency at 500, 2,000 and 10,000 characters, single request, on the named
  Mac and inside the container;
- throughput in characters per second at batch 8.

All numbers land in one table in `report.md`, with the dataset version and commit.

### 5.3 Decision rule

M2 is the default served model if it beats B1 on T2 macro-F1 and does not exceed the
latency target below. If it does not, the best system by that rule is served and the
report says why.

Latency target: p50 under 300 ms for a 2,000-character input on the M1 Mac CPU. This is a
target to verify, not a guarantee.

## 6. Service

### 6.1 API

- `POST /v1/fix`, body `{"text": "<string>"}`.
  Response `{"text": "<fixed>", "stats": {"tokens": n, "gaps": n, "changed": n,
  "model": "<name>", "latency_ms": x}}`.
  Empty text returns empty text. Missing or non-string `text` gives 422. Text longer
  than the configured limit (default 100,000 characters) gives 413.
- `GET /healthz`: `{"status": "ok", "model": "<name>", "ready": true}`; 503 until the
  model is loaded.
- `GET /metrics`: Prometheus text format: request count by endpoint and status, error
  count, latency histogram, input-length histogram, gaps-changed histogram, model info.
- `GET /`: a minimal HTML and JavaScript page with a text area, a button and the result.
  This is the UI for the Hugging Face Space; the Space runs the same image.

### 6.2 Configuration

Environment variables with the prefix `NF_`: `NF_MODEL` (`rules`, `scratch`,
`finetuned`), `NF_MODEL_REVISION` (pinned Hub revision), `NF_MAX_CHARS`,
`NF_LOG_LEVEL`. Defaults serve the model chosen by the decision rule.

### 6.3 Logging

One structured JSON line per request: timestamp, request id, endpoint, status, input
length, gaps changed, latency, model. No request text is logged.

### 6.4 Packaging and weights

- `uv` with `pyproject.toml` and `uv.lock`. CPU-only PyTorch wheels.
- Multi-stage `Dockerfile`: builder installs dependencies and downloads weights by pinned
  revision; runtime image copies the virtual environment and the weights, runs as a
  non-root user, exposes port 8000, has a `HEALTHCHECK`.
- Weights are published to a Hugging Face model repository under the author's account.
  The repository holds the model card, the config and the evaluation table. The git
  repository holds the training code and the pinned revision, not the weights.
- ONNX export of M2 is a stretch item, pursued only if the latency target is missed.

## 7. Testing

- Unit: tokenization and reconstruction, normalization, label derivation, the corruptor
  (seeded, invariant-preserving, severity 0 is identity), every rule in B1, window
  merging.
- Property-based (Hypothesis): for random Unicode text, `fix(text)` preserves the
  non-whitespace character sequence, for every system; `fix(fix(text)) == fix(text)` is
  measured, not required.
- Model: a tiny untrained M1 and the rules model run through the full `fix` path; tests
  that need published weights are marked and skipped when weights are absent.
- API: the test client covers the example, empty input, clean input, a long input that
  needs windows, the size limit, health before and after load, and the metrics endpoint.
- Container: a script builds the image, starts it, waits for health, posts the example,
  checks the output and stops it.
- `make check` runs lint, type check and tests. A GitHub Actions workflow runs the same,
  for the case where the repository is pushed to a remote.

## 8. Repository layout

```
newline-fixer/
  docs/                      requirements, design, plan, decisions/
  src/newline_fixer/
    text.py                  tokens, gaps, normalization, reconstruction, label derivation
    corrupt.py               corruptor
    rules.py                 B1
    windows.py               windowing and merge
    data/                    source loaders, filters, split, build
    models/                  fixer protocol, identity, scratch (M1), finetuned (M2)
    eval/                    metrics, evaluation runner, benchmark
    service/                 FastAPI app, schemas, metrics, logging, static UI
  scripts/                   build_data, train_scratch, train_finetune, evaluate, bench,
                             make_realistic_set, publish_weights
  notebooks/                 the Colab notebook for M2, kept in sync with scripts
  tests/
  experiments/               run results and the rendered table
  data/                      ignored except the split file, T0, T2 and T3
  Dockerfile  Makefile  pyproject.toml  uv.lock  report.md
```

## 9. Process and milestones

- Library code is written test-first. Each plan task is one commit or a short series.
- Each real choice gets a decision record on the day it is made.
- Short-lived branches per milestone, merged into `main`; the bundle carries all of them.

| Milestone | Leaves the repo in this state |
|---|---|
| M0 | requirements, design, plan, decision records |
| M1 | data pipeline, corruptor, T1 and T3 built, B0 and B1, evaluation harness, first results table |
| M2 | from-scratch model trained and evaluated, T2 built and reviewed |
| M3 | fine-tuned model selected, trained, evaluated, weights published |
| M4 | service, Docker, tests, benchmark numbers |
| M5 | report, Space deployed, bundle produced |
| M6 (optional) | ONNX, more data, hyphen handling, only while a measured number improves |

Every milestone is submittable. The bundle is sent after M5.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Models learn the corruptor, not the task | T2 is real text; the decision rule uses T2 |
| Class imbalance hides weak `JOIN` and `PARA` | per-class metrics are primary; class weights tried on validation |
| Window seams introduce errors | merge by centrality; a test feeds a long input and checks seams |
| Colab session limits | checkpoints to Drive; the notebook resumes |
| 8 GB memory during data building | streaming readers; no full corpus in memory |
| Hub unavailable at build | pinned revision; a documented local-weights path for the build |
| Uncased next-token cue lost by subword splitting | cased models only; current-gap markers |
