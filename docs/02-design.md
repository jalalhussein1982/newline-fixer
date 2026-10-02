# Design

Status: v2, 2026-10-01. Builds on `01-requirements.md`. Says how the service is built.
Decisions with real alternatives are recorded in `decisions/`.

Changes from v1, after an external design review: realistic data split into development
and test by source document; the destructive first rule of B1 replaced by a lexicon rule;
windowing defined per model budget with a coverage check; B0 defined as normalization,
with string-level change measured separately; T2 keeps raw and adjusted inputs and
counts unreachable boundaries; Wikitext dropped as a source and datasets published with
hashes; the selection rule gains a clean-text damage condition; milestones reordered so
a submittable state exists before the second model. Details are in the sections below
and in decision records 0004 and 0005.

Changes since v2, recorded 2026-10-02 after M4 so the document matches what was built
(the sections below keep the v2 text; decision records and `report.md` carry the
numbers): the from-scratch model of 4.4 trained on a Colab T4, not on the M1 Mac, after
MPS measured 12 minutes per epoch (decision 0007); the `cost()` of 4.1 is implemented as
`token_cost`, `gap_cost` and `overhead` on the `Fixer` protocol, additive, with the same
budget rule; T3 of 5.1 is built from 200 untouched clean passages of the test split only,
the T2 targets are the references of T2 and are not duplicated into T3; 6.2 gains
`NF_WEIGHTS` (a local run directory or `hf:repo@revision`) and `NF_DEVICE` (default
`cpu`); the M2 fine-tuned encoder of 4.5 was not started before M4 (section 9 allows
sending after M4 with that stated) and was built in M5. Added after M5: the encoder
of 4.5 was trained at learning rate 5e-5 (the top of the 4.5 range) for both candidates, and
selection latency was measured on the host CPU.

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
- **Output contract.** The output is always in canonical whitespace form: no leading or
  trailing whitespace, and every gap is exactly one of the four strings above. Input
  that is already canonical and needs no change comes back byte-identical. Input with
  tabs, double spaces or spaces around newlines comes back normalized even when no gap
  class changes. The API documents this.
- **What the invariant does not guarantee.** `the model` and `themodel` have the same
  non-whitespace characters. Preserving characters does not preserve words. Wrong `JOIN`
  predictions are therefore the most damaging error class, and they are measured on
  their own.

### 2.2 Label derivation

Given clean text C and corrupted text X with equal non-whitespace character sequences:

1. For C, compute `after[i]`, the normalized gap class following non-whitespace character
   `i`, with `JOIN` when the next character is also non-whitespace.
2. Walk the tokens of X. The gap after the token that ends at cumulative non-whitespace
   character count `n` gets label `after[n-1]`.

This is exact, needs no alignment heuristics, and fails loudly if the invariant is broken.

### 2.3 Known limitation

A break that was deleted without leaving any whitespace (`ways:•In`,
`paragraph.Second`) is not a gap and cannot be repaired. The requirements put that out
of scope. Hyphenated line breaks are also out of scope; the realistic sets join them
before labelling (section 5.1).

Dataset validation checks reachability explicitly: for every input and target pair, every
target break must coincide with an input gap. Unreachable boundaries are counted and
reported per dataset. The synthetic pipeline must report zero by construction; the
realistic sets report their count rather than silently dropping those boundaries.

## 3. Data

### 3.1 Clean sources

| Source | What it contributes | Pinning |
|---|---|---|
| Wikipedia, English, `wikimedia/wikipedia` on the Hugging Face Hub | paragraphs, section titles, some lists | dataset revision pinned; article ids stored with every document |
| LLM-generated documents | numbered headings, bullet lists with `•`, `-`, `*`, `1.`, mixed registers: papers, manuals, reports, emails, notes | the generated documents themselves are kept and published; the prompt, model id and date are recorded |

Wikitext-103 was considered and rejected: its "raw" variant is still tokenized, with
spaces before punctuation and `@-@` markers, so its whitespace is not clean reference
formatting. See decision record 0005. A Markdown documentation corpus is a possible third
source, deferred until the first results show whether list coverage is sufficient.

Hard-wrapped documents are excluded by a heuristic: if most lines end without terminal
punctuation and the next line starts lowercase, the document is dropped. Documents shorter
than 200 characters are dropped.

### 3.2 Normalization, deduplication, split

- Normalize whitespace to the four classes. Strip each line.
- Deduplicate on a hash of the normalized text and on the first 200 characters. Group
  Wikipedia documents by article id so two sections or revisions of one article never
  land on different sides of the split.
- Split by group, 90/5/5 into train, validation, test, with a fixed seed, before any
  corruption. The split file, listing group ids per side, is committed.
- Store as JSONL: `{id, source, source_ref, sha256, clean}`. Corrupted variants are
  materialized once with a recorded seed: `{id, source, clean, corrupted, severity}`.
- The built dataset is published to a Hugging Face dataset repository with a manifest of
  content hashes and a `DATASET_VERSION`. Re-running the pipeline is documented, but the
  published artifact is the reference; the generated documents in particular cannot be
  regenerated identically.

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
window predictions, reconstruct.

Windowing is defined per model. Each `Fixer` exposes `cost(tokens, current) -> int`, the
number of model input units a span would occupy, including marker and special tokens,
and `budget`, the maximum it accepts. For B0, B1 and M1 the cost is the token count and
the budget is 256. For M2 the cost is the subword count after markers are inserted, and
the budget is the encoder limit of 512. Windows are built greedily: extend from a start
token until the next token would exceed the budget, then start the next window at the
midpoint of the one just built, so adjacent windows overlap by about half. Every gap is
assigned to the window whose center is nearest. A test asserts complete coverage: every
gap receives exactly one prediction.

A single token whose cost alone exceeds the budget is truncated for the model input
only, keeping its first units; reconstruction always uses the original token. Tokens
longer than 500 characters trigger this path and are counted in the request stats.

### 4.2 B0, normalization only

Returns the current classes. Because reconstruction emits canonical whitespace, B0 is
not byte-identity: it collapses double spaces and tabs and strips lines. Gap-level
metrics treat it as the do-nothing floor. String-level change on clean input is reported
separately (section 5.2) so normalization is never mistaken for a model decision.

### 4.3 B1, rules

Applied per gap, first match wins. `known(w)` means `w` lowercased is in a lexicon built
from the training split, tokens with frequency at least 3.

1. Current `NL` or `PARA`, both neighbouring tokens are alphabetic, `known(left + right)`
   and at least one of `known(left)`, `known(right)` is false: `JOIN`. This joins
   `que` + `ries` and leaves `the` + `model` alone.
2. Current `NL` or `PARA`, next token starts with a lowercase letter or with closing
   punctuation: `SPACE`.
3. Next token is a list marker (`•`, `-`, `*`, `–`, or `\d+[.)]`) and the previous token
   ends with `:` or terminal punctuation: `NL`.
4. The line ending at this gap looks like a heading: starts with a section number such
   as `3.2.3`, or is at most eight tokens, title-cased, with no terminal punctuation: `PARA`.
5. Otherwise keep the current class.

The rules and their thresholds are checked on the validation split and the realistic
development set, then frozen with a decision record. After that, improvements go into
the models, not the baseline, so the comparison stays honest.

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
- Windows follow the per-model budget rule in section 4.1, with cost measured in
  subwords after marker insertion.

### 4.6 Ablation: pretraining in isolation

M1 against M2 compares two complete approaches; architecture and capacity differ as well
as pretraining. To isolate the value of the pretrained weights, the chosen M2 encoder is
also trained from random initialization with the same code, data and schedule. This is
one extra run and is part of the fine-tuned-model milestone.

### 4.7 Experiment tracking

Each training run writes `experiments/<run-id>.json` with git commit, dataset version,
seed, config, training curve summary and validation metrics. A script renders
`experiments/README.md` as a table from those files.

## 5. Evaluation

### 5.1 Test sets

Development sets are used for every choice: hyperparameters, the B1 thresholds, the M2
candidate, the clean-damage threshold, the served model. Test sets are evaluated once,
at the end, and reported.

| Id | Role | Set | Size | Built from |
|---|---|---|---|---|
| V1 | dev | synthetic validation | the validation split | corrupted at recorded severities |
| V2 | dev | realistic development | about 4 source documents, 25 to 35 passages | real PDF extractions, as T2 |
| V3 | dev | clean development | about 100 passages | untouched clean documents from the validation split |
| T1 | test | synthetic held-out | about 500 documents | the test split, corrupted at recorded severities, reported overall and by severity band |
| T2 | test | realistic | about 6 source documents, 35 to 50 passages | `pdftotext` output of real PDFs (papers, a manual, a report), including the Transformer paper; targets proposed by an LLM and reviewed by hand |
| T3 | test | clean | about 200 passages | untouched clean documents from the test split and the T2 targets |
| T0 | test | the challenge example | 1 | the README; a unit test |

Realistic passages are split between V2 and T2 by source document, never by passage, so
no document contributes to both. With about ten documents in total, the realistic
result is evidence of limited breadth, and the report says so.

For every realistic passage the repository keeps the raw extraction, the adjusted input,
the target and a note of what was adjusted. The only adjustment is joining line-end
hyphenations (`que-\nries` becomes `que\nries`) so the content invariant holds. The
report gives the number of passages adjusted, the number excluded and why, and the
number of unreachable target boundaries (section 2.3).

### 5.2 Metrics

Gap level, for every system on every set, with the support of each class printed next
to its scores:

- precision, recall, F1 per class, and macro-F1 over the classes with non-zero support
  in the reference. On clean sets `JOIN` has no support and is excluded; the table says
  which classes entered each macro-F1.
- headline F1 for newline-versus-none (`NL` or `PARA` against `JOIN` or `SPACE`);
- wrong-join rate: predicted `JOIN` where the reference is not `JOIN`, per thousand
  gaps. Reported everywhere because it is the error that changes words.
- on clean sets, gap-level damage: fraction of gaps whose class changed, and fraction of
  passages with no changed gap; and string-level change: fraction of passages whose
  output differs from the raw input, and from the normalized input. The two string-level
  numbers separate normalization from model decisions.
- paragraph exact-match: split reference and output on `PARA`; a reference paragraph
  matches if an identical string occurs among the output paragraphs; the rate is matched
  reference paragraphs over all reference paragraphs.

Service level, for M1 and M2 and B1:

- model size on disk, resident memory after warm-up;
- p50 and p95 latency at 500, 2,000 and 10,000 characters, single request, on the named
  Mac and inside the container;
- throughput in characters per second at batch 8.

All numbers land in one table in `report.md`, with the dataset version and commit.

### 5.3 Decision rule

The served model is chosen on development data only:

1. Candidates are the systems whose clean-damage rate on V3 is at most a threshold set
   on V3 once B1 exists, expected to be around one changed gap per thousand, and whose
   p50 latency for a 2,000-character input on the M1 Mac CPU is under 300 ms.
2. Among candidates, the one with the highest macro-F1 on V2 is served, with wrong-join
   rate as the tie-breaker.
3. If no learned model qualifies, B1 is served and the report says why.

Test sets T1, T2 and T3 are then evaluated once for all systems and reported. The
latency figure is a target to verify, not a guarantee.

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

Time budget: no calendar limit was set. The core deliverable is one trained model with
credible evaluation and a working service, reached at milestone M4. The second model and
everything after it are extensions, pursued only while the repository stays submittable.

| Milestone | Leaves the repo in this state | Submittable |
|---|---|---|
| M0 | requirements, design, plan, decision records | no, checkpoint |
| M1 | data pipeline, corruptor, dev and test sets built, B0 and B1 frozen, evaluation harness, first results table | no, checkpoint |
| M2 | from-scratch model trained and evaluated, realistic sets built and reviewed | no, checkpoint |
| M3 | service, Docker, tests, benchmark numbers, serving M1 or B1 by the decision rule | yes |
| M4 | report, bundle produced | yes, core deliverable |
| M5 | fine-tuned model selected, trained, evaluated, ablation run, weights published, report updated | yes |
| M6 | Space deployed; then ONNX, more data, hyphen handling, only while a measured number improves | yes |

The bundle is sent after M5 at the earliest, unless the fine-tuned model proves
infeasible, in which case it is sent after M4 with that finding in the report.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Models learn the corruptor, not the task | V2 and T2 are real text; the decision rule uses V2, the report uses T2 |
| Realistic evidence is thin | about ten source documents; the report states the limit and keeps dev and test documents disjoint |
| Class imbalance hides weak `JOIN` and `PARA` | per-class metrics with support are primary; class weights tried on V1 |
| Wrong joins change words | wrong-join rate reported everywhere; clean-damage threshold gates serving |
| Window seams introduce errors | merge by centrality; a coverage test and a long-input seam test |
| Colab session limits | checkpoints to Drive; the notebook resumes |
| 8 GB memory during data building | streaming readers; no full corpus in memory |
| Hub unavailable at build | pinned revision; a documented local-weights path for the build |
| Uncased next-token cue lost by subword splitting | cased models only; current-gap markers |
