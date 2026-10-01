# Implementation plan (roadmap)

Status: v1, 2026-10-01. Implements `02-design.md`. This file is the map; each milestone
has its own detailed, step-level plan under `docs/plans/`, written when the milestone
starts, because later milestones depend on numbers produced by earlier ones.

Conventions for every milestone:

- Library code is written test-first. A task is the smallest unit with its own tests.
- One commit per task or short series. Commit messages say what and why.
- A decision with real alternatives gets a record in `docs/decisions/` the day it is made.
- `make check` (lint, types, tests) passes at every commit on `main`.
- Nothing under `data/` is committed except what the design marks as committed: the
  split file, the evaluation sets, and the realistic passages with their review notes.

| Milestone | Detailed plan | Tasks | Done when |
|---|---|---|---|
| M0 | this document | requirements, design, plan, decision records | committed |
| M1 | `plans/2026-10-01-m1-data-baselines-eval.md` | 14 tasks, listed below | B0 and B1 evaluated on V1, V2, V3; first results table committed; B1 frozen by decision record |
| M2 | `plans/<date>-m2-scratch-model.md` | written at M2 start | M1 model trained, evaluated on dev sets, run recorded in `experiments/` |
| M3 | `plans/<date>-m3-service.md` | written at M3 start | API, Docker, tests, benchmark; served model chosen by the decision rule |
| M4 | `plans/<date>-m4-report.md` | written at M4 start | `report.md` complete, test sets evaluated once, bundle produced |
| M5 | `plans/<date>-m5-finetuned-model.md` | written at M5 start | M2 selected, trained, ablated, published; report updated |
| M6 | `plans/<date>-m6-extensions.md` | written at M6 start | Space deployed; further items only while a measured number improves |

## M1: data, baselines, evaluation harness

1. Project scaffold: `pyproject.toml`, `uv.lock`, ruff, mypy, pytest, Makefile.
2. Text core: gap classes, tokenize, join, normalize, content.
3. Label derivation and reachability; the challenge example as a fixture.
4. Seeded corruptor.
5. Fixer protocol, identity fixer, windowing with coverage check, `fix()`.
6. Lexicon and the rules fixer; the example passes through B1.
7. Gap-level and paragraph metrics.
8. Dataset records, filters, deduplication, grouped split, passage cutting.
9. Wikipedia loader with pinned revision.
10. Generated documents through the Claude API, cached on disk.
11. `build_data` script: assemble, split, corrupt, clean sets, lexicon resource, manifest, publish.
12. Realistic sets: extraction, cutting, hyphen joining, proposed targets, review, validation.
13. Evaluation runner and results table; first results; decision record freezing B1.
14. CI workflow, `data/README.md`, README update.

## M2: from-scratch model (outline)

- `models/scratch.py`: vocabulary, character CNN, BiLSTM, gap classifier, as in design 4.4.
- `scripts/train_scratch.py`: windows from the train split, early stopping on V1 macro-F1,
  run record in `experiments/`.
- Registry entry `scratch`; weights loading from a local path or the Hub.
- Evaluate on V1, V2, V3; compare with B1; decision record on class weighting.

## M3: service (outline)

- `service/app.py`: `POST /v1/fix`, `GET /healthz`, `GET /metrics`, `GET /`, as in design 6.
- Structured logging, configuration through `NF_*` variables, size limit.
- Dockerfile, multi-stage, non-root, healthcheck; container smoke test.
- `scripts/bench.py`: latency and throughput at three input lengths.
- Decision record: served model by the rule in design 5.3.

## M4: report (outline)

- Evaluate T0, T1, T2, T3 once for all systems; render the table.
- `report.md`: how to run, approach, decisions, results, failures, next steps.
- Produce the bundle; verify it clones and builds from a clean directory.

## M5: fine-tuned model (outline)

- Candidate selection run for the two encoders; decision record with numbers.
- `models/finetuned.py`, `scripts/train_finetune.py`, the Colab notebook.
- Ablation from random initialization. Publish weights. Re-run the decision rule.
- Update the report and the Hub model card.

## M6: extensions (outline)

- Hugging Face Space from the same image.
- Only while a measured number improves: ONNX export, a Markdown source, hyphen handling.
