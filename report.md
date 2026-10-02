# newline-fixer: report

## 1. Abstract

This service repairs whitespace in English text. It splits the input into non-whitespace tokens, predicts for each gap between two tokens one of four classes (JOIN, SPACE, NL, PARA), and re-joins the tokens with the predicted gaps. It never changes a non-whitespace character. Five systems are built and measured on the same sets: an identity baseline (B0), a rule baseline (B1), a from-scratch BiLSTM with character features (scratch), a pretrained encoder fine-tuned for the task (finetuned, `microsoft/deberta-v3-xsmall`), and the same encoder trained from random initialization (finetuned-ablation), which isolates the value of pretraining. The service serves the fine-tuned encoder by default, because the decision rule picked it (decision 0010, which supersedes 0008): it beats the rules on the realistic dev set (V2 macro-F1 0.898 against 0.806) and on the realistic test set, but it still makes some wrong joins where the rules make none (4.58 per thousand gaps on V2, 2.19 on T2), it does not reproduce the challenge example exactly, and in the container it is slower than the 300 ms limit (342.9 ms at 2,000 characters; the limit is stated on the host, where it takes 92.8 ms). `NF_MODEL=rules` and `NF_MODEL=scratch` serve the baselines. The demo page is at `/` of the running service. The same image runs as a Hugging Face Space, https://huggingface.co/spaces/jalalhussein1982/newline-fixer (live service at https://jalalhussein1982-newline-fixer.hf.space, demo page at its root), on free `cpu-basic` hardware (2 vCPU); free Spaces sleep after 48 hours idle and take about a minute to wake.

## 2. How to run

Docker, serving the fine-tuned encoder (the default; the image holds the weights of both learned models, fetched at build time by pinned Hub revisions):

```bash
docker build -t newline-fixer:local .
docker run --rm -p 8000:8000 newline-fixer:local
```

Docker, serving a baseline instead:

```bash
docker run --rm -p 8000:8000 -e NF_MODEL=rules newline-fixer:local
docker run --rm -p 8000:8000 -e NF_MODEL=scratch newline-fixer:local
```

Local:

```bash
uv sync --all-extras
make serve                       # http://localhost:8000, demo page at /
```

Space (no install; the live service of the same image on free `cpu-basic`, 2 vCPU): https://huggingface.co/spaces/jalalhussein1982/newline-fixer; the demo page is at https://jalalhussein1982-newline-fixer.hf.space/ and the example against it is

```bash
curl -s https://jalalhussein1982-newline-fixer.hf.space/v1/fix -H 'content-type: application/json' -d '{"text": "3.2.3 Applications of Attention\n in our Model The Transformer uses multi-head attention in three different ways: • In \"encoder-decoder attention\" layers,\n the que\nries come from the previous decoder layer."}'
```

The challenge example, through the API:

```bash
curl -s localhost:8000/v1/fix -H 'content-type: application/json' \
  -d '{"text": "3.2.3 Applications of Attention\n in our Model The Transformer uses multi-head attention in three different ways: • In \"encoder-decoder attention\" layers,\n the que\nries come from the previous decoder layer."}'
```

The response with the default model (`finetuned`; `latency_ms` varies from run to run and the first request after start-up is the slowest):

```json
{"text":"3.2.3 Applications of Attention in our Model\n\nThe Transformer uses multi-head attention in three different ways:\n\n• In \"encoder-decoder attention\" layers, the queries come from the previous decoder layer.","stats":{"tokens":30,"gaps":29,"changed":5,"model":"finetuned","latency_ms":503.03}}
```

This is not the expected output: the model puts a paragraph break (a blank line) before the first bullet where the expected output has a single newline (section 10). With `NF_MODEL=rules` the response matches the expected output exactly:

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
| `NF_MODEL` | `finetuned` | `identity`, `rules`, `scratch` or `finetuned` (decision 0010 sets the default) |
| `NF_MODEL_REVISION` | the published revision of the selected model | Hub revision of the selected model's weights |
| `NF_WEIGHTS` | unset | a local run directory or `hf:repo@revision`; overrides `NF_MODEL_REVISION` and the per-model variables |
| `NF_WEIGHTS_SCRATCH`, `NF_WEIGHTS_FINETUNED` | unset | weights source for that model (the image sets them to `/app/weights` and `/app/weights-finetuned`) |
| `NF_MAX_CHARS` | `100000` | inputs longer than this get 413 |
| `NF_LOG_LEVEL` | `INFO` | level of the JSON request log on stdout |
| `NF_DEVICE` | `cpu` | torch device for the learned models |

Inside the image both revisions are fixed at build time (`--build-arg NF_MODEL_REVISION=<rev>` for scratch, `--build-arg NF_FINETUNED_REVISION=<rev>` for the fine-tuned encoder), so `NF_MODEL_REVISION` has no effect on a running container. One JSON line per request goes to stdout; request text is never logged. The [README](README.md) has the offline build.

Tests. `make check` runs lint, type check and pytest. CI ([workflow](.github/workflows/ci.yml)) has two jobs: `check` runs `make check`; `container` runs `scripts/container_check.py`, which builds the image, starts it with `NF_MODEL=rules` (its default), waits for health, posts the challenge example and fails unless the output matches; `uv run python scripts/container_check.py --no-build --model finetuned --expect-mismatch` starts the image's default model, requires health 200 and content preservation, and expects the known one-gap difference on the example; CI runs both checks. The suite covers the API contract and error codes (`tests/test_service_api.py`), content preservation for any text (Hypothesis tests in `tests/test_service_api.py`, `tests/test_scratch_fixer.py`, `tests/test_text.py`), clean input unchanged, a long input that needs windows, the size limit (413), readiness (503 before load), metrics and request logging (`tests/test_service_observability.py`), configuration, the fine-tuned encoder's window encoding, fixer and trainer (`tests/test_finetune_encoding.py`, `tests/test_finetuned_fixer.py`, `tests/test_finetune_trainer.py`), and the container check.

## 3. The problem as formulated

[Decision 0001](docs/decisions/0001-gap-classification-formulation.md) fixes the formulation. A token is a maximal run of non-whitespace characters. A gap is the whitespace between two tokens. Each gap becomes one of four classes: JOIN (the empty string), SPACE, NL (one newline) or PARA (two newlines). A whitespace run is normalized to a class by counting newlines: two or more is PARA, one is NL, none is SPACE. Leading and trailing whitespace is dropped. The content invariant is that the non-whitespace characters of the output equal those of the input; reconstruction enforces it and tests check it for every system.

Out of scope, by design (design 2.3): a break that was deleted without leaving whitespace (`ways:•In`), hyphenated line breaks, and re-wrapping to a column width. The realistic sets join hyphenations before labelling.

The formulation makes the five operations in the challenge example measurable: three removed newlines (one inside a word, which is a JOIN), one inserted line break before a bullet (NL) and one inserted paragraph break after a heading (PARA).

## 4. Data

Two clean sources, chosen in [decision 0005](docs/decisions/0005-clean-text-sources.md). Wikipedia: 5,000 documents from `wikimedia/wikipedia` at pinned revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` (reduced from a 20,000 target). Generated structured documents: 2,000 documents with numbered headings, bullet lists and mixed registers, written in a Claude session under the pipeline's prompt rules, not through the API, so the script cannot regenerate them. Wikitext-103 was rejected because its "raw" variant is tokenized. After filtering and deduplication the split is by document: train 6,299, validation 350, test 350.

The training and synthetic evaluation inputs come from a seeded corruptor. It replaces true newlines by spaces with a probability that grows with severity, and inserts newlines at arbitrary character positions, including inside words. One document in ten is left uncorrupted.

The dataset (clean splits, evaluation sets, the 2,000 generated documents, the split file and a manifest of content hashes) is on the Hugging Face Hub at https://huggingface.co/datasets/jalalhussein1982/newline-fixer-data, revision `42b4d8333b2bb89901f0be1680756be628382c88`. The committed evaluation sets under `data/sets/` and the realistic passages under `data/realistic/` are enough to re-run the evaluation.

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

Realistic sets. They come from ten real PDFs: word2vec, fasttext, nist-800-63 and the GNU Bash manual (V2); attention, bert, resnet, adam, nist-ai-rmf and gnu-make (T2). Text was extracted with `pdftotext` and cut into passages, 8 per document at most, and every passage is split between V2 and T2 by document. Targets were proposed in a Claude session (whitespace only) and then reviewed: the four dev documents and `attention` by the author; the other five by Claude Fable 5.1 against the review questions, signed off by the author. Hyphenations are joined before labelling; the number of passages that needed this adjustment is 0 for both sets (counted by comparing each raw extraction with its adjusted input), so no kept passage contained a hyphenated line end. Two targets were edited during review (`adam/00`, `adam/07`, a section number joined with its heading; see `data/README.md`). Eleven passages were excluded as having no sensible newline target: tables and diagram labels, equation debris, and one two-column table extracted with its columns interleaved (3 from V2, 8 from T2). The unreachable boundaries are 0 for V2 and 0 for T2. Ten documents is limited breadth: the realistic result is evidence, not an estimate for all text.

## 5. Systems

**B0, identity.** Returns the current gap classes. Reconstruction emits canonical whitespace, so B0 is not byte identity; string-level change is measured separately.

**B1, rules.** Applied per gap, first match wins: a lexicon rule joins a split word when the joined form is a known word and a part is not (`que` + `ries`); an NL or PARA before a lowercase word or closing punctuation becomes SPACE; a list marker after terminal punctuation gets NL; a heading-like line (a section number, or a short title-case line) gets PARA; otherwise the gap is kept. [Decision 0006](docs/decisions/0006-rules-baseline-frozen.md) froze B1 at V1 0.635 and V2 0.806 macro-F1. It sets the clean-damage gate at B1's own V3 value, 0.0026, which a served model must not exceed.

**Scratch, a model written from scratch.** A 30,000-word embedding, a character CNN per token (so that fragments such as `que` and capitalization are visible), an embedding of the current gap class, a two-layer BiLSTM, and a classifier over the states on both sides of the gap: 5,551,692 parameters (design 4.4). Training examples are re-corrupted every epoch. It was trained on a Colab T4 at about 52 s per epoch; the M1 Mac measured 12.1 minutes per epoch on MPS, so training moved. [Decision 0007](docs/decisions/0007-scratch-model-class-weighting.md): unweighted cross-entropy beat inverse-frequency class weights on every metric. The weights are on the Hub at the pinned revision `6c311e757d17e89c80b7b86908043637a4f56e28`, never in git ([decision 0003](docs/decisions/0003-weights-on-hub-not-in-git.md)).

<!-- rendered by scripts/report_tables.py at 4819a84 -->

| run | device | epochs | minutes | best epoch | V1 macro-F1 | V3 damage | params | commit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| finetuned-ablation | cuda | 3 | 15.4 | 3 | 0.839 | 0.0057 | 70,646,404 | `6093ac8f9125` |
| finetuned | cuda | 3 | 15.0 | 3 | 0.954 | 0.0008 | 70,646,404 | `6093ac8f9125` |
| ft-deberta-select | cuda | 1 | 3.8 | 1 | 0.937 | 0.0023 | 70,646,404 | `6093ac8f9125` |
| ft-distilbert-select | cuda | 1 | 2.4 | 1 | 0.901 | 0.0024 | 65,195,524 | `6093ac8f9125` |
| scratch-v1-inverse | cuda | 5 | 5.2 | 3 | 0.741 | 0.0447 | 5,551,692 | `dd2a39b955d0` |
| scratch-v1 | cuda | 8 | 8.6 | 8 | 0.922 | 0.0016 | 5,551,692 | `dd2a39b955d0` |

`scratch` and `finetuned` rows: weights revisions `6c311e757d17e89c80b7b86908043637a4f56e28` of `jalalhussein1982/newline-fixer-scratch` and `11d6b26e80dfa2c9606702cd2755a63c9dce99ed` of `jalalhussein1982/newline-fixer-finetuned` (section 12). `finetuned-ablation` rows: the same architecture from random initialization, weights not published.

**The fine-tuned pretrained encoder (design 4.5), `finetuned`.** Two pretrained encoders were trained for one epoch on a Colab T4 and compared on V1 macro-F1 under a latency condition on the M1 Mac CPU: p50 per 256-token window within three times the scratch model's (22.5 ms, so 67.4 ms). [Decision 0009](docs/decisions/0009-encoder-candidate.md) picked `microsoft/deberta-v3-xsmall` (V1 macro-F1 0.937, 65.5 ms per window) over `distilbert-base-cased` (0.901, 51.2 ms); both were within the limit, and DeBERTa's margin on it is thin. The selection used one epoch and one seed, so the 0.035 gap is not a significance claim. The full run uses the same training data and window scheme as scratch, with the `[NL]` and `[PP]` markers added as special tokens (embeddings resized) and labels on the first subword of each token. Schedule: learning rate 5e-5, batch size 16, three epochs (best epoch 3), weight decay 0.01, warmup fraction 0.06, mixed precision (fp16) on a T4, seed 1; 70,646,404 parameters; 15.0 minutes of training. The learning rate is the top of the 3e-5 to 5e-5 range design 4.5 gives, used for both candidates; it was not tuned.

<!-- rendered by scripts/report_tables.py at 4819a84 -->

Scratch p50 per 256-token window: 22.5 ms; limit (3x): 67.4 ms.

| run | pretrained | params | V1 macro-F1 | p50 ms / window | p95 ms | within limit |
|---|---|---:|---:|---:|---:|---|
| ft-deberta-select | microsoft/deberta-v3-xsmall | 70,646,404 | 0.937 | 65.5 | 76.0 | yes |
| ft-distilbert-select | distilbert-base-cased | 65,195,524 | 0.901 | 51.2 | 53.6 | yes |

**The ablation, `finetuned-ablation`.** The same architecture (70.6M parameters) from random initialization, with identical data, windows and schedule (15.4 minutes on the same T4; [decision 0004](docs/decisions/0004-model-strategy-restated.md)). Only the weights at initialization differ, so the difference between `finetuned` and `finetuned-ablation` is the measured value of pretraining, from one seed per arm. Its weights are not published: it is evidence, not a servable model.

## 6. Evaluation method

Metrics (design 5.2), per system and set:

- precision, recall and F1 per gap class, and macro-F1 over the classes with support in the reference (JOIN has no support on clean sets);
- break-F1: newline (NL or PARA) against none (JOIN or SPACE);
- wrong-join rate per thousand gaps: JOIN predicted where the reference is not JOIN. It is reported everywhere because it is the one error that changes words;
- changed gaps: the fraction of gaps whose class differs from the input; on the clean sets V3 and T3 and in severity band 0 this is clean damage, elsewhere it includes correct repairs;
- string-level change: the fraction of passages whose output differs from the raw or the normalized input;
- paragraph match: the fraction of reference paragraphs that occur unchanged among the output paragraphs.

The tables below show macro-F1, break-F1, PARA F1, wrong-join rate, changed gaps and paragraph match. The per-class precision, recall and F1 for the four classes, with support, are in the per-class tables in section 7; `experiments/README.md` adds the string-level numbers.

The decision rule (design 5.3), verbatim:

1. Candidates are the systems whose clean-damage rate on V3 is at most a threshold set on V3 once B1 exists, expected to be around one changed gap per thousand, and whose p50 latency for a 2,000-character input on the M1 Mac CPU is under 300 ms.
2. Among candidates, the one with the highest macro-F1 on V2 is served, with wrong-join rate as the tie-breaker.
3. If no learned model qualifies, B1 is served and the report says why.

The discipline: every choice (B1's thresholds, the class weighting, the served model) used dev sets only. Test sets T0 to T3 were evaluated once for identity, rules and scratch at commit `4955e06adaa2`, after decision 0008, and once for finetuned and finetuned-ablation at commit `44480da7717a`, after decision 0010 had chosen the served model on the dev sets; the two records are merged for the tables below (equal set hashes required). No choice used a test set.

## 7. Results

**The verdict on requirement Q2.** The requirement is "The learned model must be shown to add value over both, or the report must say that it does not." The fine-tuned encoder adds value over both baselines and over the from-scratch model, on the real passages as well as the synthetic ones. On the realistic dev set V2 its macro-F1 is 0.898, against 0.806 for the rules, 0.753 for identity and 0.733 for scratch; on the realistic test set T2 it is 0.637, against 0.498, 0.487 and 0.536, and its break-F1 is 0.801, above the rules' 0.760 (scratch 0.669). On synthetic corruptions it scores 0.954 on V1 and on T1, against 0.635 and 0.627 for the rules. On the clean sets it changes fewer gaps than the rules or scratch (damage 0.0008 on V3 against 0.0026 and 0.0016; 0.0011 on T3 against 0.0102 and 0.0023). The ablation shows where the value comes from: the same encoder from random initialization scores 0.614 on V2 and 0.441 on T2, below identity on both, so pretraining is worth 0.284 macro-F1 on V2, 0.196 on T2, 0.115 on V1 and 0.109 on T1 (one seed per arm). The remaining costs are real. The model makes wrong joins where the rules make none: 4.58 per thousand gaps on V2, 2.19 on T2 and 0.11 on T3 (scratch: 9.16, 6.58 and 0.11). Its V2 damage, 0.0641, is above the rules' 0.0604. On the challenge example (T0) it scores 0.667 macro-F1 and paragraph match 0.500, against 1.000 and 1.000 for the rules: it puts a paragraph break before the first bullet. This report therefore says that the fine-tuned model is shown to add value over both baselines on real text, and it is served (decision 0010); the scratch model is not.

Dev sets:

<!-- rendered by scripts/report_tables.py at 4819a84 -->

Rendered from commit `20eb57a61d0a`, sets V1=4f22b6469bbd, V2=574867bf0d4d, V3=07db0ab68315.

| set | system | gaps | macro-F1 | break-F1 | PARA F1 | wrong-join /1k | changed gaps | paragraph match |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | identity | 123610 | 0.418 | 0.346 | 0.424 | 0.00 | 0.0000 | 0.161 |
| V1 | rules | 123610 | 0.635 | 0.513 | 0.410 | 0.00 | 0.0186 | 0.145 |
| V1 | scratch | 123610 | 0.922 | 0.885 | 0.861 | 0.11 | 0.0365 | 0.656 |
| V1 | finetuned | 123610 | 0.954 | 0.940 | 0.899 | 0.02 | 0.0384 | 0.747 |
| V1 | finetuned-ablation | 123610 | 0.839 | 0.763 | 0.749 | 0.68 | 0.0337 | 0.428 |
| V2 | identity | 2401 | 0.753 | 0.714 | 0.864 | 0.00 | 0.0000 | 0.555 |
| V2 | rules | 2401 | 0.806 | 0.869 | 0.776 | 0.00 | 0.0604 | 0.526 |
| V2 | scratch | 2401 | 0.733 | 0.737 | 0.785 | 9.16 | 0.0804 | 0.453 |
| V2 | finetuned | 2401 | 0.898 | 0.915 | 0.886 | 4.58 | 0.0641 | 0.708 |
| V2 | finetuned-ablation | 2401 | 0.614 | 0.547 | 0.548 | 6.66 | 0.0979 | 0.226 |
| V3 | identity | 9187 | 1.000 | 1.000 | 1.000 | 0.00 | 0.0000 | 1.000 |
| V3 | rules | 9187 | 0.940 | 0.988 | 0.933 | 0.00 | 0.0026 | 0.921 |
| V3 | scratch | 9187 | 0.975 | 0.971 | 0.988 | 0.00 | 0.0016 | 0.959 |
| V3 | finetuned | 9187 | 0.991 | 0.986 | 0.985 | 0.00 | 0.0008 | 0.974 |
| V3 | finetuned-ablation | 9187 | 0.926 | 0.900 | 0.909 | 0.44 | 0.0057 | 0.794 |

`scratch` and `finetuned` rows: weights revisions `6c311e757d17e89c80b7b86908043637a4f56e28` of `jalalhussein1982/newline-fixer-scratch` and `11d6b26e80dfa2c9606702cd2755a63c9dce99ed` of `jalalhussein1982/newline-fixer-finetuned` (section 12). `finetuned-ablation` rows: the same architecture from random initialization, weights not published.

Test sets (each system evaluated once; the provenance line lists the commits of both records and is marked dirty because the older one was, see section 10):

<!-- rendered by scripts/report_tables.py at 4819a84 -->

Rendered from commits `4955e06adaa2` + `44480da7717a` (dirty tree), sets T0=95d8fe63481b, T1=a3d16ebe012c, T2=11ba1ea6f18c, T3=aaceae2a74c8.

| set | system | gaps | macro-F1 | break-F1 | PARA F1 | wrong-join /1k | changed gaps | paragraph match |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | identity | 29 | 0.231 | 0.000 | 0.000 | 0.00 | 0.0000 | 0.000 |
| T0 | rules | 29 | 1.000 | 1.000 | 1.000 | 0.00 | 0.1724 | 1.000 |
| T0 | scratch | 29 | 0.620 | 0.800 | 0.500 | 0.00 | 0.2069 | 0.000 |
| T0 | finetuned | 29 | 0.667 | 1.000 | 0.667 | 0.00 | 0.1724 | 0.500 |
| T0 | finetuned-ablation | 29 | 0.667 | 1.000 | 0.667 | 0.00 | 0.1724 | 0.500 |
| T1 | identity | 120009 | 0.426 | 0.356 | 0.455 | 0.00 | 0.0000 | 0.187 |
| T1 | rules | 120009 | 0.627 | 0.508 | 0.436 | 0.00 | 0.0188 | 0.174 |
| T1 | scratch | 120009 | 0.921 | 0.884 | 0.861 | 0.13 | 0.0382 | 0.667 |
| T1 | finetuned | 120009 | 0.954 | 0.937 | 0.897 | 0.07 | 0.0403 | 0.750 |
| T1 | finetuned-ablation | 120009 | 0.845 | 0.777 | 0.753 | 0.62 | 0.0342 | 0.449 |
| T2 | identity | 3645 | 0.487 | 0.587 | 0.708 | 0.00 | 0.0000 | 0.368 |
| T2 | rules | 3645 | 0.498 | 0.760 | 0.667 | 0.00 | 0.0601 | 0.382 |
| T2 | scratch | 3645 | 0.536 | 0.669 | 0.651 | 6.58 | 0.0829 | 0.375 |
| T2 | finetuned | 3645 | 0.637 | 0.801 | 0.762 | 2.19 | 0.0782 | 0.553 |
| T2 | finetuned-ablation | 3645 | 0.441 | 0.537 | 0.385 | 5.49 | 0.0941 | 0.145 |
| T3 | identity | 18392 | 1.000 | 1.000 | 1.000 | 0.00 | 0.0000 | 1.000 |
| T3 | rules | 18392 | 0.811 | 0.978 | 0.759 | 0.00 | 0.0102 | 0.893 |
| T3 | scratch | 18392 | 0.977 | 0.968 | 0.982 | 0.11 | 0.0023 | 0.929 |
| T3 | finetuned | 18392 | 0.989 | 0.984 | 0.970 | 0.11 | 0.0011 | 0.964 |
| T3 | finetuned-ablation | 18392 | 0.925 | 0.891 | 0.927 | 0.22 | 0.0073 | 0.832 |

`scratch` and `finetuned` rows: weights revisions `6c311e757d17e89c80b7b86908043637a4f56e28` of `jalalhussein1982/newline-fixer-scratch` and `11d6b26e80dfa2c9606702cd2755a63c9dce99ed` of `jalalhussein1982/newline-fixer-finetuned` (section 12). `finetuned-ablation` rows: the same architecture from random initialization, weights not published.

V2 macro-F1 averages three classes (JOIN has no support there) while T2 averages four (JOIN has a support of one gap, the per-class table shows it), so V2 and T2 values are not comparable; without JOIN, T2 macro-F1 is finetuned 0.782, scratch 0.689, rules 0.664, identity 0.650 and finetuned-ablation 0.558.

### Per class, dev sets (V1, V2)

<!-- rendered by scripts/report_tables.py at 4819a84 -->

Rendered from commit `20eb57a61d0a`, sets V1=4f22b6469bbd, V2=574867bf0d4d, V3=07db0ab68315.

| set | system | class | support | precision | recall | F1 |
|---|---|---|---:|---:|---:|---:|
| V1 | identity | JOIN | 967 | 0.000 | 0.000 | 0.000 |
| V1 | identity | SPACE | 118422 | 0.976 | 0.991 | 0.983 |
| V1 | identity | NL | 1680 | 0.224 | 0.328 | 0.266 |
| V1 | identity | PARA | 2541 | 0.798 | 0.288 | 0.424 |
| V1 | rules | JOIN | 967 | 1.000 | 0.517 | 0.682 |
| V1 | rules | SPACE | 118422 | 0.974 | 0.998 | 0.986 |
| V1 | rules | NL | 1680 | 0.656 | 0.355 | 0.460 |
| V1 | rules | PARA | 2541 | 0.756 | 0.281 | 0.410 |
| V1 | scratch | JOIN | 967 | 0.986 | 0.963 | 0.974 |
| V1 | scratch | SPACE | 118422 | 0.994 | 0.998 | 0.996 |
| V1 | scratch | NL | 1680 | 0.881 | 0.835 | 0.857 |
| V1 | scratch | PARA | 2541 | 0.954 | 0.786 | 0.861 |
| V1 | finetuned | JOIN | 967 | 0.997 | 0.994 | 0.995 |
| V1 | finetuned | SPACE | 118422 | 0.997 | 0.999 | 0.998 |
| V1 | finetuned | NL | 1680 | 0.932 | 0.917 | 0.924 |
| V1 | finetuned | PARA | 2541 | 0.943 | 0.859 | 0.899 |
| V1 | finetuned-ablation | JOIN | 967 | 0.913 | 0.911 | 0.912 |
| V1 | finetuned-ablation | SPACE | 118422 | 0.988 | 0.997 | 0.992 |
| V1 | finetuned-ablation | NL | 1680 | 0.808 | 0.621 | 0.702 |
| V1 | finetuned-ablation | PARA | 2541 | 0.893 | 0.645 | 0.749 |
| V2 | identity | JOIN | 0 | 0.000 | 0.000 | 0.000 |
| V2 | identity | SPACE | 2239 | 1.000 | 0.942 | 0.970 |
| V2 | identity | NL | 54 | 0.269 | 1.000 | 0.424 |
| V2 | identity | PARA | 108 | 0.945 | 0.796 | 0.864 |
| V2 | rules | JOIN | 0 | 0.000 | 0.000 | 0.000 |
| V2 | rules | SPACE | 2239 | 0.991 | 0.990 | 0.990 |
| V2 | rules | NL | 54 | 0.569 | 0.759 | 0.651 |
| V2 | rules | PARA | 108 | 0.839 | 0.722 | 0.776 |
| V2 | scratch | JOIN | 0 | 0.000 | 0.000 | 0.000 |
| V2 | scratch | SPACE | 2239 | 0.984 | 0.992 | 0.988 |
| V2 | scratch | NL | 54 | 0.500 | 0.370 | 0.426 |
| V2 | scratch | PARA | 108 | 0.904 | 0.694 | 0.785 |
| V2 | finetuned | JOIN | 0 | 0.000 | 0.000 | 0.000 |
| V2 | finetuned | SPACE | 2239 | 0.997 | 0.994 | 0.996 |
| V2 | finetuned | NL | 54 | 0.750 | 0.889 | 0.814 |
| V2 | finetuned | PARA | 108 | 0.957 | 0.824 | 0.886 |
| V2 | finetuned-ablation | JOIN | 0 | 0.000 | 0.000 | 0.000 |
| V2 | finetuned-ablation | SPACE | 2239 | 0.965 | 0.987 | 0.976 |
| V2 | finetuned-ablation | NL | 54 | 0.412 | 0.259 | 0.318 |
| V2 | finetuned-ablation | PARA | 108 | 0.767 | 0.426 | 0.548 |

`scratch` and `finetuned` rows: weights revisions `6c311e757d17e89c80b7b86908043637a4f56e28` of `jalalhussein1982/newline-fixer-scratch` and `11d6b26e80dfa2c9606702cd2755a63c9dce99ed` of `jalalhussein1982/newline-fixer-finetuned` (section 12). `finetuned-ablation` rows: the same architecture from random initialization, weights not published.

### Per class, test sets (T1, T2)

<!-- rendered by scripts/report_tables.py at 4819a84 -->

Rendered from commits `4955e06adaa2` + `44480da7717a` (dirty tree), sets T0=95d8fe63481b, T1=a3d16ebe012c, T2=11ba1ea6f18c, T3=aaceae2a74c8.

| set | system | class | support | precision | recall | F1 |
|---|---|---|---:|---:|---:|---:|
| T1 | identity | JOIN | 961 | 0.000 | 0.000 | 0.000 |
| T1 | identity | SPACE | 114599 | 0.974 | 0.991 | 0.982 |
| T1 | identity | NL | 1884 | 0.237 | 0.301 | 0.265 |
| T1 | identity | PARA | 2565 | 0.798 | 0.318 | 0.455 |
| T1 | rules | JOIN | 961 | 1.000 | 0.520 | 0.684 |
| T1 | rules | SPACE | 114599 | 0.972 | 0.997 | 0.985 |
| T1 | rules | NL | 1884 | 0.633 | 0.293 | 0.401 |
| T1 | rules | PARA | 2565 | 0.728 | 0.312 | 0.436 |
| T1 | scratch | JOIN | 961 | 0.983 | 0.953 | 0.968 |
| T1 | scratch | SPACE | 114599 | 0.993 | 0.998 | 0.996 |
| T1 | scratch | NL | 1884 | 0.883 | 0.840 | 0.861 |
| T1 | scratch | PARA | 2565 | 0.947 | 0.789 | 0.861 |
| T1 | finetuned | JOIN | 961 | 0.991 | 0.988 | 0.989 |
| T1 | finetuned | SPACE | 114599 | 0.997 | 0.999 | 0.998 |
| T1 | finetuned | NL | 1884 | 0.931 | 0.931 | 0.931 |
| T1 | finetuned | PARA | 2565 | 0.942 | 0.857 | 0.897 |
| T1 | finetuned-ablation | JOIN | 961 | 0.920 | 0.884 | 0.902 |
| T1 | finetuned-ablation | SPACE | 114599 | 0.987 | 0.997 | 0.992 |
| T1 | finetuned-ablation | NL | 1884 | 0.841 | 0.651 | 0.734 |
| T1 | finetuned-ablation | PARA | 2565 | 0.898 | 0.648 | 0.753 |
| T2 | identity | JOIN | 1 | 0.000 | 0.000 | 0.000 |
| T2 | identity | SPACE | 3486 | 1.000 | 0.937 | 0.967 |
| T2 | identity | NL | 46 | 0.159 | 0.978 | 0.274 |
| T2 | identity | PARA | 112 | 0.763 | 0.661 | 0.708 |
| T2 | rules | JOIN | 1 | 0.000 | 0.000 | 0.000 |
| T2 | rules | SPACE | 3486 | 0.991 | 0.985 | 0.988 |
| T2 | rules | NL | 46 | 0.262 | 0.478 | 0.338 |
| T2 | rules | PARA | 112 | 0.726 | 0.616 | 0.667 |
| T2 | scratch | JOIN | 1 | 0.040 | 1.000 | 0.077 |
| T2 | scratch | SPACE | 3486 | 0.989 | 0.983 | 0.986 |
| T2 | scratch | NL | 46 | 0.393 | 0.478 | 0.431 |
| T2 | scratch | PARA | 112 | 0.690 | 0.616 | 0.651 |
| T2 | finetuned | JOIN | 1 | 0.111 | 1.000 | 0.200 |
| T2 | finetuned | SPACE | 3486 | 0.996 | 0.984 | 0.990 |
| T2 | finetuned | NL | 46 | 0.480 | 0.783 | 0.595 |
| T2 | finetuned | PARA | 112 | 0.739 | 0.786 | 0.762 |
| T2 | finetuned-ablation | JOIN | 1 | 0.048 | 1.000 | 0.091 |
| T2 | finetuned-ablation | SPACE | 3486 | 0.979 | 0.979 | 0.979 |
| T2 | finetuned-ablation | NL | 46 | 0.257 | 0.391 | 0.310 |
| T2 | finetuned-ablation | PARA | 112 | 0.500 | 0.312 | 0.385 |

`scratch` and `finetuned` rows: weights revisions `6c311e757d17e89c80b7b86908043637a4f56e28` of `jalalhussein1982/newline-fixer-scratch` and `11d6b26e80dfa2c9606702cd2755a63c9dce99ed` of `jalalhussein1982/newline-fixer-finetuned` (section 12). `finetuned-ablation` rows: the same architecture from random initialization, weights not published.

T1 by severity band, rules, scratch and finetuned:

<!-- rendered by scripts/report_tables.py at 4819a84 -->

Rendered from commits `4955e06adaa2` + `44480da7717a` (dirty tree), sets T0=95d8fe63481b, T1=a3d16ebe012c, T2=11ba1ea6f18c, T3=aaceae2a74c8.

| severity band | system | items | gaps | macro-F1 | wrong-join /1k | changed gaps |
|---|---|---:|---:|---:|---:|---:|
| 0 | rules | 53 | 18575 | 0.806 | 0.00 | 0.0095 |
| 0 | scratch | 53 | 18575 | 0.987 | 0.00 | 0.0012 |
| 0 | finetuned | 53 | 18575 | 0.993 | 0.00 | 0.0007 |
| (0,0.33] | rules | 92 | 30364 | 0.656 | 0.00 | 0.0080 |
| (0,0.33] | scratch | 92 | 30364 | 0.922 | 0.03 | 0.0241 |
| (0,0.33] | finetuned | 92 | 30364 | 0.958 | 0.00 | 0.0254 |
| (0.33,0.66] | rules | 104 | 38495 | 0.607 | 0.00 | 0.0207 |
| (0.33,0.66] | scratch | 104 | 38495 | 0.917 | 0.13 | 0.0439 |
| (0.33,0.66] | finetuned | 104 | 38495 | 0.944 | 0.18 | 0.0460 |
| (0.66,1] | rules | 101 | 32575 | 0.502 | 0.00 | 0.0320 |
| (0.66,1] | scratch | 101 | 32575 | 0.895 | 0.31 | 0.0658 |
| (0.66,1] | finetuned | 101 | 32575 | 0.942 | 0.06 | 0.0702 |

`scratch` and `finetuned` rows: weights revisions `6c311e757d17e89c80b7b86908043637a4f56e28` of `jalalhussein1982/newline-fixer-scratch` and `11d6b26e80dfa2c9606702cd2755a63c9dce99ed` of `jalalhussein1982/newline-fixer-finetuned` (section 12). `finetuned-ablation` rows: the same architecture from random initialization, weights not published.

Where the fine-tuned model wins. JOIN: on V1 its JOIN F1 is 0.995 against 0.682 for the rules (the per-class tables); the rules join a split word only when a lexicon lookup succeeds. Real text: on V2 its NL F1 is 0.814 against 0.651 for the rules and its PARA F1 0.886 against 0.776, and paragraph match is 0.708 against 0.526 on V2 and 0.553 against 0.382 on T2. Clean text: paragraph match is 0.974 on V3 and 0.964 on T3 against 0.921 and 0.893 for the rules. The severity table shows a gain in every band: macro-F1 0.993 against 0.806 at severity 0, and 0.942 against 0.502 at the highest band; its clean damage in band 0 is 0.0007.

Where it loses. Wrong joins: 4.58 per thousand gaps on V2, 2.19 on T2, up to 0.18 in the middle severity band of T1, against 0.00 for the rules in every one of those (a wrong join glues two words together). Damage on V2 is 0.0641 against 0.0604. On T2 the single JOIN gap is found (recall 1.0) but nine gaps are predicted JOIN (precision 0.111). The challenge example is not reproduced (section 10). On the V2 passages the model is above identity in every class that has support, but its NL precision there is 0.750, so some of its inserted line breaks are wrong.

Why. The design's risk table predicted that models learn the corruptor, not the task. The synthetic corruptions are uniform random breaks at the same rates in training and in V1 and T1; the PDF extractions differ from them. The drop from V1 to V2 is 0.056 macro-F1 for the fine-tuned model (0.954 to 0.898), 0.189 for scratch (0.922 to 0.733) and 0.225 for the random-init ablation (0.839 to 0.614), so pretraining shrinks the gap without closing it. The explanation was not tested beyond this comparison.

## 8. Service numbers

<!-- rendered by scripts/report_tables.py at 4819a84 -->

| label | system | p50 / p95 ms @500 | @2,000 | @10,000 | chars/s (batch 8) | RSS MB | disk MB | commit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| container-finetuned-threads-1 | finetuned | 106.7 / 125.3 | 460.6 / 493.2 | 4166.8 / 4296.7 | 11,429 | - | - | `3ea41718cacc` |
| container-finetuned-threads-2 | finetuned | 179.0 / 199.2 | 388.1 / 457.4 | 3462.8 / 3714.2 | 7,983 | - | - | `3ea41718cacc` (dirty) |
| container-finetuned-threads-4 | finetuned | 150.6 / 180.0 | 326.7 / 400.9 | 2897.1 / 3501.2 | 8,180 | - | - | `3ea41718cacc` (dirty) |
| container-finetuned | finetuned | 121.8 / 158.9 | 342.9 / 585.1 | 2584.1 / 2954.5 | 9,458 | - | - | `3390fa73a69c` |
| container-rules | rules | 1.4 / 4.3 | 1.9 / 3.1 | 4.2 / 5.0 | 1,502,660 | - | - | `3ef9677a888b` |
| container-scratch | scratch | 54.7 / 61.8 | 304.1 / 318.5 | 1977.4 / 2056.9 | 23,563 | - | - | `3ef9677a888b` |
| m1-mac-cpu | identity | 0.1 / 0.1 | 0.3 / 0.3 | 1.5 / 1.6 | 6,976,161 | 24 | 0.0 | `610ca964767e` |
| m1-mac-cpu | rules | 0.1 / 0.1 | 0.5 / 0.5 | 2.7 / 2.7 | 4,354,047 | 32 | 0.2 | `610ca964767e` |
| m1-mac-cpu | scratch | 7.3 / 7.5 | 40.2 / 42.0 | 267.8 / 283.9 | 79,701 | 297 | 22.6 | `610ca964767e` |
| m1-mac-cpu | finetuned | 28.8 / 34.5 | 92.8 / 96.4 | 807.9 / 855.7 | 24,898 | 662 | 290.9 | `610ca964767e` |
| space | finetuned | 168.0 / 184.2 | 342.2 / 379.0 | 2266.2 / 2576.4 | 3,224 | - | - | `4819a84bb26a` |

`scratch` and `finetuned` rows: weights revisions `6c311e757d17e89c80b7b86908043637a4f56e28` of `jalalhussein1982/newline-fixer-scratch` and `11d6b26e80dfa2c9606702cd2755a63c9dce99ed` of `jalalhussein1982/newline-fixer-finetuned` (section 12). `finetuned-ablation` rows: the same architecture from random initialization, weights not published.

Latency is one request at a time; throughput is eight concurrent requests of 2,000 characters. Rows labelled `m1-mac-cpu` are in-process measurements on an Apple M1 (8 GB) on CPU. Rows labelled `container-*` go through HTTP against the image running in Docker Desktop's Linux VM on the same Mac, so they carry no size or memory figures.

The design states the latency rule on the host CPU, so the host p50 at 2,000 characters is the figure that enters the decision rule: rules 0.5 ms, scratch 40.2 ms, finetuned 92.8 ms, limit 300 ms. All three pass. The container figure for finetuned, 342.9 ms (p95 585.1 ms), is 3.7 times the host figure and would fail the limit if it were the gate. Decision 0010 extrapolated about 700 ms from scratch's ratio of 7.6 (304.1 ms in the container against 40.2 on the host); the measurement is half that, because the ratio is not constant across models. The cause of the container overhead is not isolated; the thread experiment (below) ruled out oversubscription as the main cause. Rules in the container take 1.9 ms. Resident memory after warm-up is 32 MB for rules, 297 MB for scratch and 662 MB for finetuned; the fine-tuned weights are 290.9 MB on disk against 22.6 MB for scratch. The image holds the weights of both learned models and is 1.89 GB on disk (535 MB content size, from `docker image ls`), dominated by the CPU PyTorch wheel and the two weight sets. Container throughput for finetuned is 9,458 characters per second against 24,898 on the host.

The thread experiment (decision 0011) kept torch's default thread count: the best setting, 4 threads, was only 4.7% better than the default at 2,000 characters (326.7 against 342.9 ms), under the 10% margin fixed beforehand. The `space` row is the same image on the Hugging Face Space (free `cpu-basic`, 2 vCPU), measured over HTTP from the author's Mac, so it includes the network round trip: p50 342.2 ms (p95 379.0 ms) at 2,000 characters, the same as the local container (342.9 ms, 8 vCPUs; a near-coincidence, since the Space figure includes the round trip from the Mac) and 3.7 times the host's 92.8 ms; its throughput is lower (3,224 against 9,458 chars/s), consistent with two vCPUs serving eight concurrent requests, though the network path differs and the cause was not isolated.

## 9. Decisions

Records are in [`docs/decisions/`](docs/decisions/). They are never edited; a change is a new record (decision 0011 carries marked clarifications and a dated postscript added before its acceptance on this branch).

- [0001](docs/decisions/0001-gap-classification-formulation.md): four-way classification of each whitespace gap, over a binary newline decision and over free-form rewriting. The four classes express every operation in the example, and only gaps change, so content is preserved by construction.
- [0002](docs/decisions/0002-self-hosted-small-models.md): serve self-hosted small models and compare a from-scratch model with a fine-tuned one; superseded by 0004.
- [0003](docs/decisions/0003-weights-on-hub-not-in-git.md): publish weights to the Hugging Face Hub at a pinned revision, fetched at image build. Git LFS does not travel in a bundle and committed weights bloat it.
- [0004](docs/decisions/0004-model-strategy-restated.md): the same model plan, with the claims about a hosted LLM restated as trade-offs after an external review, and an ablation added that trains the encoder from random initialization to isolate pretraining.
- [0005](docs/decisions/0005-clean-text-sources.md): Wikipedia plus generated documents; Wikitext-103 rejected because its whitespace is tokenized, not clean.
- [0006](docs/decisions/0006-rules-baseline-frozen.md): B1 frozen at the first version that beats identity on V1 and V2 with no wrong joins on V3. Three table-like passages were removed from V2 under the reviewer rule, not by tuning rules.
- [0007](docs/decisions/0007-scratch-model-class-weighting.md): unweighted cross-entropy for the from-scratch model. Inverse-frequency weights gave V1 macro-F1 0.741 and V3 damage 0.0447, failing the gate.
- [0008](docs/decisions/0008-served-model.md): serve the rules baseline; superseded by 0010. Both systems pass both candidate conditions; the rules have the higher V2 macro-F1 (0.806 against 0.733) and no wrong joins.
- [0009](docs/decisions/0009-encoder-candidate.md): fine-tune `microsoft/deberta-v3-xsmall`. After one epoch each, it led `distilbert-base-cased` on V1 macro-F1 (0.937 against 0.901) and both were within the per-window latency limit of 67.4 ms (65.5 and 51.2 ms); one epoch and one seed, so the gap is not a significance claim.
- [0010](docs/decisions/0010-served-model-after-m5.md): serve the fine-tuned encoder, superseding 0008. Rules, scratch and finetuned pass both candidate conditions; finetuned has the highest V2 macro-F1 (0.898 against 0.806 and 0.733). The decision records the caveats that this report measures: wrong joins, the challenge example, and the cost in weight size, memory and container latency. The random-init ablation (V3 damage 0.0057, over the gate) is evidence for what pretraining is worth, not a candidate.

## 10. Known failures and limits

- **Requirement A2 (the challenge example through the API) holds for `NF_MODEL=rules` and not for the default model, which differs on one gap; the default follows decision 0010's rule (Q2) over A2, deliberately.**
- **The challenge example is reproduced by the rules and not by the served model.** The fine-tuned model gets the heading and the lead-in sentence right and puts a paragraph break (a blank line) before the first bullet where the expected output has a single newline. T0 macro-F1 is 0.667 (paragraph match 0.500) against 1.000 for the rules. Scratch failed it in two places (T0 0.620). Anyone who tries the README example against the default image will see the difference; `NF_MODEL=rules` gives the exact output.
- **Wrong joins.** The fine-tuned model makes 4.58 (V2), 2.19 (T2) and 0.11 (T3) wrong joins per thousand gaps where the rules make none. A wrong join glues two words together. It is lower than scratch (9.16, 6.58, 0.11) and is the worst error the service can make.
- **Container latency.** The served model takes 342.9 ms p50 at 2,000 characters in the container (p95 585.1 ms), above the 300 ms limit, which the design states on the host (92.8 ms). The cause of the container overhead is not isolated. A user running the default image on a CPU like the Docker Desktop VM's will wait about a third of a second for 2,000 characters and 2.6 seconds for 10,000.
- **One seed.** Every learned number is one training run per model; the ablation difference on V2 (0.284) is large, but the size of the gap and the ranking of finetuned against scratch on T2 are single measurements. The encoder was chosen from one epoch of one seed, with a latency margin of 65.5 ms against 67.4 ms.
- **Weight size and memory.** 290.9 MB of weights and 662 MB resident, against 22.6 MB and 297 MB for scratch and 32 MB for the rules.
- **The rules also damage clean text.** V3 damage is 0.0026, exactly at the gate, and their PARA F1 is below identity on V1 and V2. On T3, the clean test set, the rules change 0.0102 of gaps (four times the gate set on V3), with paragraph match 0.893 and macro-F1 0.811, where finetuned changes 0.0011 with 0.964 and 0.989. The gate set on V3 did not generalise for the rules; finetuned's own T3 cost is 0.11 wrong joins per thousand gaps.
- **The realistic evidence is ten documents**: four in V2, six in T2, 29 and 40 passages.
- **The generated documents are reproducible only by download, not by script.**
- **Out of scope by design**: hyphenated line breaks and breaks deleted without whitespace.
- **Test-set records.** `experiments/results/test-sets.json` (identity, rules, scratch) was rendered at commit `4955e06adaa2` with uncommitted files: the decision record 0008 and the README of the next commit, `9cb6c5d`. The code was that of `4955e06`, so the numbers do not depend on the difference. `experiments/results/test-sets-m5.json` (finetuned and its ablation) was rendered from a clean tree at `44480da7717a`; the merged tables are marked dirty because the older record was. The candidate-selection record `experiments/results/m5-candidates.json` was also measured from a dirty tree (decision 0010).

## 11. What would be done next

In priority order, each with the number it targets:

1. Isolate the container latency (the thread count is not the main cause (decision 0011); a second host or a native Linux CPU would test the VM explanation). Target: finetuned p50 at 2,000 characters in the container from 342.9 ms toward the host's 92.8 ms, under 300 ms.
2. More realistic corruptions in the training data. Target: wrong joins on V2 from 4.58 per thousand toward the 0.00 of the rules, and the V1 to V2 gap of 0.056 macro-F1 (0.954 against 0.898).
3. ONNX export and thread tuning on the Space are the remaining latency levers; neither is needed now (item 5).
4. A second seed for finetuned and its ablation, to put a spread on the 0.284 pretraining effect on V2.
5. ONNX export only if the Space p50 at 2,000 characters exceeded 1,000 ms (decision 0011). It is 342.2 ms, so ONNX is not pursued; the host figure (92.8 ms against 300 ms) does not call for it either.

## 12. Process

The requirements, design and implementation plan were committed before any code (requirement A7); `git log` shows them first. Work then went in milestones M1 (data, baselines, evaluation), M2 (from-scratch model), M3 (service, Docker, benchmark), M4 (the report) and M5 (the fine-tuned encoder and its ablation, then this update), each a short-lived branch merged into `main` by pull request with CI. Decision records were written on the day of the choice. Library code is test-first. Each plan task was executed with a fresh implementer and a separate reviewer (subagent-driven development). The author's development environment was Claude Code (and Codex); every generated change was reviewed and every decision is the author's.

Locations:

- Hugging Face Space (deployment target): https://huggingface.co/spaces/jalalhussein1982/newline-fixer, a Docker Space that builds the same `Dockerfile` as the local image (decision 0011 and section 8 give its latency).
- Weights, scratch: https://huggingface.co/jalalhussein1982/newline-fixer-scratch, revision `6c311e757d17e89c80b7b86908043637a4f56e28`.
- Weights, fine-tuned encoder: https://huggingface.co/jalalhussein1982/newline-fixer-finetuned, revision `11d6b26e80dfa2c9606702cd2755a63c9dce99ed`. The ablation weights are not published.
- Colab notebooks: `notebooks/train_scratch_colab.ipynb` and `notebooks/train_finetune_colab.ipynb` (candidate selection and the two full runs of M5). The Colab runs used transformers 5, which writes `extra_special_tokens` as a list in the tokenizer config; the repository pins transformers below 5 (4.57.6 in the lock file), so `FinetunedFixer.load` maps the list to the mapping form before loading (decision 0009).
- Wikipedia source: `wikimedia/wikipedia`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
- Built dataset: https://huggingface.co/datasets/jalalhussein1982/newline-fixer-data, revision `42b4d8333b2bb89901f0be1680756be628382c88`.
