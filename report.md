# newline-fixer: report

## 1. Abstract

This service repairs whitespace in English text. It splits the input into non-whitespace tokens, predicts for each gap between two tokens one of four classes (JOIN, SPACE, NL, PARA), and re-joins the tokens with the predicted gaps. It never changes a non-whitespace character. Three systems are built and measured on the same sets: an identity baseline (B0), a rule baseline (B1), and a from-scratch BiLSTM with character features (scratch). The service serves the rules baseline B1 by default, because the decision rule picked it (decision 0008). The learned model wins by a wide margin on synthetic corruptions and loses to the rules on real passages, so it ships behind a flag (`NF_MODEL=scratch`). The demo page is at `/` of the running service. A Hugging Face Space is not deployed: that is the optional deliverable D6, planned as milestone M6.

## 2. How to run

Docker, serving the rules baseline (the default):

```bash
docker build -t newline-fixer:local .
docker run --rm -p 8000:8000 newline-fixer:local
```

Docker, serving the from-scratch model (the image fetches its weights at build time by a pinned Hub revision):

```bash
docker run --rm -p 8000:8000 -e NF_MODEL=scratch newline-fixer:local
```

Local:

```bash
uv sync --all-extras
make serve                       # http://localhost:8000, demo page at /
```

The challenge example, through the API:

```bash
curl -s localhost:8000/v1/fix -H 'content-type: application/json' \
  -d '{"text": "3.2.3 Applications of Attention\n in our Model The Transformer uses multi-head attention in three different ways: • In \"encoder-decoder attention\" layers,\n the que\nries come from the previous decoder layer."}'
```

The response with the default model (`latency_ms` varies from run to run):

```json
{"text":"3.2.3 Applications of Attention in our Model\n\nThe Transformer uses multi-head attention in three different ways:\n• In \"encoder-decoder attention\" layers, the queries come from the previous decoder layer.","stats":{"tokens":30,"gaps":29,"changed":5,"model":"rules","latency_ms":1.08}}
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /v1/fix` | body `{"text": "..."}`; returns `{"text", "stats"}`; 422 for a missing or non-string `text`, 413 above `NF_MAX_CHARS` |
| `GET /healthz` | 503 until the model is loaded, then `{"status": "ok", "model", "ready": true}` |
| `GET /metrics` | Prometheus text: requests by endpoint and status, errors, latency, input length and gaps changed histograms |
| `GET /` | the demo page: a text area, a button and the result |

Configuration, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `NF_MODEL` | `rules` | `identity`, `rules` or `scratch` (decision 0008 sets the default) |
| `NF_MODEL_REVISION` | the published scratch-v1 revision | Hub revision of the scratch weights |
| `NF_WEIGHTS` | unset | a local run directory or `hf:repo@revision`; overrides `NF_MODEL_REVISION` |
| `NF_MAX_CHARS` | `100000` | inputs longer than this get 413 |
| `NF_LOG_LEVEL` | `INFO` | level of the JSON request log on stdout |
| `NF_DEVICE` | `cpu` | torch device for the scratch model |

Inside the image the scratch revision is fixed at build time (`--build-arg NF_MODEL_REVISION=<rev>`), so `NF_MODEL_REVISION` has no effect on a running container. One JSON line per request goes to stdout; request text is never logged. The [README](README.md) has the offline build.

Tests. `make check` runs lint, type check and the pytest suite. CI ([workflow](.github/workflows/ci.yml)) has two jobs: `check` runs `uv sync --all-extras` and `make check`; `container` runs `scripts/container_check.py`, which builds the image, starts it, waits for health, posts the challenge example and fails unless the output matches. The suite covers the classes the requirements name: API contract and error codes (`tests/test_service_api.py`), content preservation for any text (Hypothesis tests in `tests/test_service_api.py`, `tests/test_scratch_fixer.py` and `tests/test_text.py`), clean input unchanged, a long input that needs windows, the size limit (413), readiness (503 before load), metrics and request logging (`tests/test_service_observability.py`), configuration, and the container check.

## 3. The problem as formulated

[Decision 0001](docs/decisions/0001-gap-classification-formulation.md) fixes the formulation. A token is a maximal run of non-whitespace characters. A gap is the whitespace between two tokens. Each gap becomes one of four classes: JOIN (the empty string), SPACE, NL (one newline) or PARA (two newlines). A whitespace run is normalized to a class by counting newlines: two or more is PARA, one is NL, none is SPACE. Leading and trailing whitespace is dropped. The content invariant is that the non-whitespace characters of the output equal those of the input; reconstruction enforces it and tests check it for every system.

Out of scope, by design (design 2.3): a break that was deleted without leaving whitespace (`ways:•In`), hyphenated line breaks, and re-wrapping to a column width. The realistic sets join hyphenations before labelling.

The formulation makes the five operations in the challenge example measurable: three removed newlines (one inside a word, which is a JOIN), one inserted line break before a bullet (NL) and one inserted paragraph break after a heading (PARA).

## 4. Data

Two clean sources, chosen in [decision 0005](docs/decisions/0005-clean-text-sources.md). Wikipedia: 5,000 documents from `wikimedia/wikipedia` at pinned revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` (reduced from a 20,000 target). Generated structured documents: 2,000 documents with numbered headings, bullet lists and mixed registers, written in a Claude session under the pipeline's prompt rules, not through the API, so the script cannot regenerate them. Wikitext-103 was rejected because its "raw" variant is tokenized. After filtering and deduplication the split is by document: train 6,299, validation 350, test 350.

The training and synthetic evaluation inputs come from a seeded corruptor. It replaces true newlines by spaces with a probability that grows with severity, and inserts newlines at arbitrary character positions, including inside words. One document in ten is left uncorrupted.

<!-- dataset-url -->
The built dataset is to be published to the Hugging Face Hub with a content-hash manifest. This has not happened yet: the Hub URL is pending. Until it is published, the generated documents are not in the repository, and the training data cannot be rebuilt from it alone. The committed evaluation sets under `data/sets/` and the realistic passages under `data/realistic/` are enough to re-run the evaluation.

Evaluation sets. Development sets drive every choice; test sets were evaluated once.

| Id | Role | Set | Items | Gaps |
|---|---|---|---:|---:|
| V1 | dev | synthetic, validation split | 350 | 123610 |
| V2 | dev | realistic, 4 source documents | 29 | 2401 |
| V3 | dev | clean, validation split | 100 | 9187 |
| T0 | test | the challenge example | 1 | 29 |
| T1 | test | synthetic, test split | 350 | 120009 |
| T2 | test | realistic, 6 source documents | 40 | 3645 |
| T3 | test | clean, test split | 200 | 18392 |

Realistic sets. They come from ten real PDFs: word2vec, fasttext, nist-800-63 and the GNU Bash manual (V2); attention, bert, resnet, adam, nist-ai-rmf and gnu-make (T2). Text was extracted with `pdftotext` and cut into passages, 8 per document at most, and every passage is split between V2 and T2 by document. Targets were proposed in a Claude session (whitespace only) and then reviewed: the four dev documents and `attention` by the author; the other five by Claude Fable 5.1 against the review questions, signed off by the author. Hyphenations are joined before labelling; the number of passages that needed this adjustment is 0 for both sets. Eleven passages were excluded as having no sensible newline target: tables and diagram labels, equation debris, and one two-column table extracted with its columns interleaved (3 from V2, 8 from T2). The unreachable boundaries are 0 for V2 and 0 for T2. Ten documents is limited breadth: the realistic result is evidence, not an estimate for all text.

## 5. Systems

**B0, identity.** Returns the current gap classes. Reconstruction emits canonical whitespace, so B0 is not byte identity; string-level change is measured separately.

**B1, rules.** Applied per gap, first match wins: a lexicon rule joins a split word when the joined form is a known word and a part is not (`que` + `ries`); an NL or PARA before a lowercase word or closing punctuation becomes SPACE; a list marker after terminal punctuation gets NL; a heading-like line (a section number, or a short title-case line) gets PARA; otherwise the gap is kept. [Decision 0006](docs/decisions/0006-rules-baseline-frozen.md) froze B1 at V1 0.635 and V2 0.806 macro-F1. It sets the clean-damage gate at B1's own V3 value, 0.0026, which a served model must not exceed.

**Scratch, a model written from scratch.** A 30,000-word embedding, a character CNN per token (so that fragments such as `que` and capitalization are visible), an embedding of the current gap class, a two-layer BiLSTM, and a classifier over the states on both sides of the gap: 5,551,692 parameters (design 4.4). Training examples are re-corrupted every epoch. It was trained on a Colab T4 at about 52 s per epoch; the M1 Mac measured 12.1 minutes per epoch on MPS, so training moved. [Decision 0007](docs/decisions/0007-scratch-model-class-weighting.md): unweighted cross-entropy beat inverse-frequency class weights on every metric. The weights are on the Hub at the pinned revision `6c311e757d17e89c80b7b86908043637a4f56e28`, never in git ([decision 0003](docs/decisions/0003-weights-on-hub-not-in-git.md)).

<!-- rendered by scripts/report_tables.py at a43bf4d -->
| run | device | epochs | minutes | best epoch | V1 macro-F1 | V3 damage | params | commit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| scratch-v1-inverse | cuda | 5 | 5.2 | 3 | 0.741 | 0.0447 | 5,551,692 | `dd2a39b955d0` |
| scratch-v1 | cuda | 8 | 8.6 | 8 | 0.922 | 0.0016 | 5,551,692 | `dd2a39b955d0` |

**The fine-tuned pretrained encoder (design 4.5) is planned work, not done.** The report has no result for it.

## 6. Evaluation method

Metrics (design 5.2), per system and set:

- precision, recall and F1 per gap class, and macro-F1 over the classes with support in the reference (JOIN has no support on clean sets);
- break-F1: newline (NL or PARA) against none (JOIN or SPACE);
- wrong-join rate per thousand gaps: JOIN predicted where the reference is not JOIN. It is reported everywhere because it is the one error that changes words;
- clean damage: the fraction of gaps whose class changed on clean text;
- string-level change: the fraction of passages whose output differs from the raw or the normalized input;
- paragraph match: the fraction of reference paragraphs that occur unchanged among the output paragraphs.

The tables below show macro-F1, break-F1, PARA F1, wrong-join rate, clean damage and paragraph match. The JSON records under `experiments/results/` hold the precision, recall and F1 of all four classes; `experiments/README.md` adds JOIN F1 and the string-level numbers.

The decision rule (design 5.3), verbatim:

1. Candidates are the systems whose clean-damage rate on V3 is at most a threshold set on V3 once B1 exists, expected to be around one changed gap per thousand, and whose p50 latency for a 2,000-character input on the M1 Mac CPU is under 300 ms.
2. Among candidates, the one with the highest macro-F1 on V2 is served, with wrong-join rate as the tie-breaker.
3. If no learned model qualifies, B1 is served and the report says why.

The discipline: every choice (B1's thresholds, the class weighting, the served model) used dev sets only. Test sets T0 to T3 were evaluated once for all systems, at commit `4955e06adaa2`, after decision 0008.

## 7. Results

**The verdict on requirement Q2.** The requirement is "The learned model must be shown to add value over both, or the report must say that it does not." The learned model adds value over both baselines on synthetic corruptions: V1 macro-F1 0.922 against 0.635 for the rules and 0.418 for identity, and T1 0.921 against 0.627 and 0.426. It also damages less clean text than the rules: V3 0.0016 against 0.0026, T3 0.0023 against 0.0102. It does not add value on the real passages that decide the serving choice. On V2 it scores 0.733, below the rules (0.806) and below identity (0.753), with 9.16 wrong joins per thousand gaps where the rules make none. On T2 its macro-F1 is higher than both (0.536 against 0.498 and 0.487), but its break-F1 is lower than the rules' (0.669 against 0.760) and it makes 6.58 wrong joins per thousand against 0.00. On the challenge example (T0) the rules score 1.000 and the model 0.620. This report therefore says that the learned model is not shown to add value on real text, and the rules are served.

Dev sets:

<!-- rendered by scripts/report_tables.py at a43bf4d -->
| set | system | gaps | macro-F1 | break-F1 | PARA F1 | wrong-join /1k | clean damage | paragraph match |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | identity | 123610 | 0.418 | 0.346 | 0.424 | 0.00 | 0.0000 | 0.161 |
| V1 | rules | 123610 | 0.635 | 0.513 | 0.410 | 0.00 | 0.0186 | 0.145 |
| V1 | scratch | 123610 | 0.922 | 0.885 | 0.861 | 0.11 | 0.0365 | 0.656 |
| V2 | identity | 2401 | 0.753 | 0.714 | 0.864 | 0.00 | 0.0000 | 0.555 |
| V2 | rules | 2401 | 0.806 | 0.869 | 0.776 | 0.00 | 0.0604 | 0.526 |
| V2 | scratch | 2401 | 0.733 | 0.737 | 0.785 | 9.16 | 0.0804 | 0.453 |
| V3 | identity | 9187 | 1.000 | 1.000 | 1.000 | 0.00 | 0.0000 | 1.000 |
| V3 | rules | 9187 | 0.940 | 0.988 | 0.933 | 0.00 | 0.0026 | 0.921 |
| V3 | scratch | 9187 | 0.975 | 0.971 | 0.988 | 0.00 | 0.0016 | 0.959 |

Test sets (evaluated once; the record was rendered from a dirty tree, see section 10):

<!-- rendered by scripts/report_tables.py at a43bf4d -->
| set | system | gaps | macro-F1 | break-F1 | PARA F1 | wrong-join /1k | clean damage | paragraph match |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | identity | 29 | 0.231 | 0.000 | 0.000 | 0.00 | 0.0000 | 0.000 |
| T0 | rules | 29 | 1.000 | 1.000 | 1.000 | 0.00 | 0.1724 | 1.000 |
| T0 | scratch | 29 | 0.620 | 0.800 | 0.500 | 0.00 | 0.2069 | 0.000 |
| T1 | identity | 120009 | 0.426 | 0.356 | 0.455 | 0.00 | 0.0000 | 0.187 |
| T1 | rules | 120009 | 0.627 | 0.508 | 0.436 | 0.00 | 0.0188 | 0.174 |
| T1 | scratch | 120009 | 0.921 | 0.884 | 0.861 | 0.13 | 0.0382 | 0.667 |
| T2 | identity | 3645 | 0.487 | 0.587 | 0.708 | 0.00 | 0.0000 | 0.368 |
| T2 | rules | 3645 | 0.498 | 0.760 | 0.667 | 0.00 | 0.0601 | 0.382 |
| T2 | scratch | 3645 | 0.536 | 0.669 | 0.651 | 6.58 | 0.0829 | 0.375 |
| T3 | identity | 18392 | 1.000 | 1.000 | 1.000 | 0.00 | 0.0000 | 1.000 |
| T3 | rules | 18392 | 0.811 | 0.978 | 0.759 | 0.00 | 0.0102 | 0.893 |
| T3 | scratch | 18392 | 0.977 | 0.968 | 0.982 | 0.11 | 0.0023 | 0.929 |

T1 by severity band, rules and scratch:

<!-- rendered by scripts/report_tables.py at a43bf4d -->
| severity band | system | items | gaps | macro-F1 | wrong-join /1k | damage |
|---|---|---:|---:|---:|---:|---:|
| 0 | rules | 53 | 18575 | 0.806 | 0.00 | 0.0095 |
| 0 | scratch | 53 | 18575 | 0.987 | 0.00 | 0.0012 |
| (0,0.33] | rules | 92 | 30364 | 0.656 | 0.00 | 0.0080 |
| (0,0.33] | scratch | 92 | 30364 | 0.922 | 0.03 | 0.0241 |
| (0.33,0.66] | rules | 104 | 38495 | 0.607 | 0.00 | 0.0207 |
| (0.33,0.66] | scratch | 104 | 38495 | 0.917 | 0.13 | 0.0439 |
| (0.66,1] | rules | 101 | 32575 | 0.502 | 0.00 | 0.0320 |
| (0.66,1] | scratch | 101 | 32575 | 0.895 | 0.31 | 0.0658 |

Where the model wins. JOIN: on V1 its JOIN F1 is 0.974 against 0.682 for the rules (`experiments/README.md`); the rules join a split word only when a lexicon lookup succeeds. PARA structure on clean text: scratch has paragraph match 0.959 on V3 and 0.929 on T3 against 0.921 and 0.893 for the rules, and fewer damaged gaps. The severity table shows a gain in every band: macro-F1 0.987 against 0.806 at severity 0, and 0.895 against 0.502 at the highest band. Its own cost grows with severity: wrong joins go from 0.00 to 0.31 per thousand and damage from 0.0012 to 0.0658.

Where it loses. On V2 and T2 the model has a lower break-F1 than the rules (0.737 against 0.869 on V2, 0.669 against 0.760 on T2), and it glues words together. Decision 0007 measured NL recall of 0.37 on V2. Paragraph match is lower than the rules' on V2 (0.453 against 0.526) and on T2 (0.375 against 0.382).

Why. The design's risk table predicted that models learn the corruptor, not the task. The synthetic corruptions are uniform random breaks at the same rates in training and in V1 and T1; the PDF extractions differ from them. The gap between the V1 and V2 scores (0.922 and 0.733) is the size of that difference for this model. The explanation was not tested beyond this comparison.

## 8. Service numbers

<!-- rendered by scripts/report_tables.py at a43bf4d -->
| label | system | p50 / p95 ms @500 | @2,000 | @10,000 | chars/s (batch 8) | RSS MB | disk MB | commit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| container-rules | rules | 1.4 / 4.3 | 1.9 / 3.1 | 4.2 / 5.0 | 1,502,660 | - | - | `3ef9677a888b` |
| container-scratch | scratch | 54.7 / 61.8 | 304.1 / 318.5 | 1977.4 / 2056.9 | 23,563 | - | - | `3ef9677a888b` |
| m1-mac-cpu | identity | 0.1 / 0.1 | 0.3 / 0.3 | 1.5 / 1.5 | 7,204,882 | 24 | 0.0 | `c181dae5d6cb` |
| m1-mac-cpu | rules | 0.1 / 0.1 | 0.5 / 0.5 | 2.7 / 2.7 | 4,412,221 | 32 | 0.2 | `c181dae5d6cb` |
| m1-mac-cpu | scratch | 7.5 / 7.6 | 40.6 / 41.9 | 268.1 / 274.1 | 71,266 | 274 | 22.6 | `c181dae5d6cb` |

Latency is one request at a time; throughput is eight concurrent requests of 2,000 characters. Rows labelled `m1-mac-cpu` are in-process measurements on an Apple M1 (8 GB) on CPU. Rows labelled `container-*` go through HTTP against the image running in Docker Desktop's Linux VM on the same Mac, so they carry no size or memory figures.

The design states the latency rule on the host CPU, so the host p50 at 2,000 characters is the figure that enters the decision rule: rules 0.5 ms, scratch 40.6 ms, limit 300 ms. Both pass. The container figure for scratch, 304.1 ms, is 7.5 times the host figure and would fail the limit if it were the gate. The cause is not isolated; thread oversubscription in the VM is the leading hypothesis and is untested. Rules in the container take 1.9 ms. Resident memory after warm-up is 32 MB for rules and 274 MB for scratch. The image is 1.18 GB (from `docker image ls`), dominated by the CPU PyTorch wheel.

## 9. Decisions

Records are in [`docs/decisions/`](docs/decisions/). They are never edited; a change is a new record that supersedes the old one.

- [0001](docs/decisions/0001-gap-classification-formulation.md): four-way classification of each whitespace gap, over a binary newline decision and over free-form rewriting. The four classes express every operation in the example, and only gaps change, so content is preserved by construction.
- [0002](docs/decisions/0002-self-hosted-small-models.md): serve self-hosted small models and compare a from-scratch model with a fine-tuned one; superseded by 0004.
- [0003](docs/decisions/0003-weights-on-hub-not-in-git.md): publish weights to the Hugging Face Hub at a pinned revision, fetched at image build. Git LFS does not travel in a bundle and committed weights bloat it.
- [0004](docs/decisions/0004-model-strategy-restated.md): the same model plan, with the claims about a hosted LLM restated as trade-offs after an external review, and an ablation added that trains the encoder from random initialization to isolate pretraining.
- [0005](docs/decisions/0005-clean-text-sources.md): Wikipedia plus generated documents; Wikitext-103 rejected because its whitespace is tokenized, not clean.
- [0006](docs/decisions/0006-rules-baseline-frozen.md): B1 frozen at the first version that beats identity on V1 and V2 with no wrong joins on V3. Three table-like passages were removed from V2 under the reviewer rule, not by tuning rules.
- [0007](docs/decisions/0007-scratch-model-class-weighting.md): unweighted cross-entropy for the from-scratch model. Inverse-frequency weights gave V1 macro-F1 0.741 and V3 damage 0.0447, failing the gate.
- [0008](docs/decisions/0008-served-model.md): serve the rules baseline. Both systems pass both candidate conditions; the rules have the higher V2 macro-F1 (0.806 against 0.733) and no wrong joins.

## 10. Known failures and limits

- **The challenge example is reproduced by the rules and not by the model.** The model puts a paragraph break after `3.2.3` instead of a space, and a paragraph break before the first bullet where the expected output has a single newline (decision 0007). The rest of the example is correct. The rules reproduce it exactly (T0 macro-F1 1.000).
- **V2 regression and T2 wrong joins.** The model is below the rules and below identity on V2, and makes 9.16 (V2) and 6.58 (T2) wrong joins per thousand gaps. A wrong join glues two words together.
- **The V3 gate margin is thin.** The chosen epoch has V3 damage 0.0016 against the gate of 0.0026, but three of the eight epochs were above the gate (decision 0007). Selection used V1 macro-F1, so passing the gate at epoch 8 is partly luck.
- **The rules also damage clean text.** V3 damage is 0.0026, exactly at the gate, and their PARA F1 is below identity on V1 and V2.
- **The realistic evidence is ten documents**: four in V2, six in T2, 29 and 40 passages.
- **The generated documents are not reproducible by script**, and the dataset is not yet on the Hub.
- **The container latency gap**: scratch at 304.1 ms against 40.6 ms on the host, cause not isolated.
- **Out of scope by design**: hyphenated line breaks and breaks deleted without whitespace.
- **Test-set record from a dirty tree.** `experiments/results/test-sets.json` was rendered at commit `4955e06adaa2` with uncommitted files: the decision record 0008 and the README of the next commit, `9cb6c5d`. They were not yet committed when the evaluation ran. The code was that of `4955e06`, so the numbers do not depend on the difference.

## 11. What would be done next

In priority order, each with the number it targets:

1. The fine-tuned cased encoder of design 4.5 (milestone M5), with model selection on V2. Target: V2 macro-F1 above 0.806 under the damage gate of 0.0026.
2. A V2-aware or wrong-join-penalised selection rule for the from-scratch model (decision 0007 consequences). Target: V2 wrong joins from 9.16 per thousand toward the 0.00 of the rules.
3. More realistic corruptions in the training data. Target: the V1 to V2 gap, 0.922 against 0.733 macro-F1, and the NL recall of 0.37 on V2.
4. The container thread experiment. Target: scratch p50 at 2,000 characters from 304.1 ms toward the host's 40.6 ms.
5. The Hugging Face Space (milestone M6), which runs the same image.
6. ONNX export only if latency is missed; the host figure (40.6 ms against 300 ms) does not call for it.

## 12. Process

The requirements, design and implementation plan were committed before any code (requirement A7); `git log` shows them first. Work then went in milestones M1 (data, baselines, evaluation), M2 (from-scratch model), M3 (service, Docker, benchmark) and M4 (this report), each a short-lived branch merged into `main` by pull request with CI. Decision records were written on the day of the choice. Library code is test-first. Each plan task was executed with a fresh implementer and a separate reviewer (subagent-driven development). The author's development environment was Claude Code (and Codex); every generated change was reviewed and every decision is the author's.

Locations:

- Weights: https://huggingface.co/jalalhussein1982/newline-fixer-scratch, revision `6c311e757d17e89c80b7b86908043637a4f56e28`.
- Wikipedia source: `wikimedia/wikipedia`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
- Built dataset: not yet published; Hub URL pending.
