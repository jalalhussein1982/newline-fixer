# Requirements

Status: draft v1, 2026-10-01. Source: the challenge `README.md` from BottleCapAI.
This document states what must be built and how success is judged. It does not say how.

## 1. Objective

Design and implement a machine learning service that fixes newline placement in English
natural-language text, exposed over an HTTP API. State-of-the-art quality is not required.
The service must "efficiently generate reasonable answers".

## 2. Problem definition

"Fixing newlines" means restoring the whitespace structure of a text whose line breaks
have been lost, moved or inserted in the wrong places. The challenge gives one example,
section 3.2.3 of "Attention Is All You Need":

Input:

```
3.2.3 Applications of Attention
 in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers,
 the que
ries come from the previous decoder layer.[...]
```

Expected output:

```
3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:
• In "encoder-decoder attention" layers, the queries come from the previous decoder layer.
[...]
```

The example contains every operation the service must perform:

| Input fragment | Output fragment | Operation |
|---|---|---|
| `Attention\n in our Model` | `Attention in our Model` | remove a spurious newline (a space is already present) |
| `Model The Transformer` | `Model\n\nThe Transformer` | insert a paragraph break after a heading |
| `ways: • In` | `ways:\n• In` | insert a line break before a list item |
| `layers,\n the` | `layers, the` | remove a spurious newline |
| `que\nries` | `queries` | remove a spurious newline inside a word, joining the parts |

From this, the problem is defined as follows.

- **P1. Only whitespace changes.** The sequence of non-whitespace characters in the output
  is identical to the input. The service never adds, removes or alters a letter, digit or
  punctuation mark.
- **P2. Every gap is re-decided.** Each run of whitespace between two non-whitespace tokens
  becomes exactly one of: nothing (join), one space, one newline (line break), or two
  newlines (paragraph break).
- **P3. Both directions.** The service removes wrong newlines and inserts missing ones.
  A solution that only inserts, or only removes, does not meet the objective.
- **P4. Clean input stays clean.** Text that already has correct newlines is returned
  unchanged, or as close to unchanged as the model allows. Damage to good input is a
  failure mode that is measured.

Out of scope, unless the design document says otherwise: hyphenated line breaks
(`que-\nries`), missing spaces between words (`ways:•In`), re-wrapping to a column width,
languages other than English, and Markdown or HTML structure beyond plain newlines.

## 3. Functional requirements

- **F1.** An HTTP API with at least one endpoint that accepts text and returns the text
  with fixed newlines.
- **F2.** The API is implemented in Python. The framework is free to choose.
- **F3.** The model architecture and the way the model is obtained are free to choose.
- **F4.** The service handles the challenge example end to end and produces the expected
  output, up to the trailing `[...]` marker.
- **F5.** The service accepts inputs longer than the model's context window without
  failing. The design document fixes the policy.

## 4. Quality requirements

- **Q1. Efficiency.** The service runs on CPU with latency and throughput that are
  measured and reported. The design document sets the targets.
- **Q2. Reasonable answers.** Output quality is measured against a do-nothing baseline and
  a rule-based baseline on held-out data. The learned model must be shown to add value
  over both, or the report must say that it does not.
- **Q3. Monitoring.** The running service exposes enough to be watched in production: a
  health check and basic request, error and latency signals at minimum.
- **Q4. Reproducibility.** Data generation, training and evaluation can be re-run from the
  repository with documented commands. Model weights are included or fetched by a
  documented, pinned procedure.

## 5. Deliverables

| # | Deliverable | Required by the challenge |
|---|---|---|
| D1 | HTTP API, text in, fixed text out | yes |
| D2 | `Dockerfile` that runs the service | yes |
| D3 | Tests for the service | yes |
| D4 | Metrics evaluating the model | yes |
| D5 | `report.md`: how to run, approach, decisions, results | yes |
| D6 | Deployed demo with a minimal UI (Hugging Face Spaces), link in the report | optional, encouraged |
| D7 | Git bundle of the repository with all branches, sent by e-mail | yes, the submission format |

## 6. Judging criteria, as stated

"Your drive, interest in ML and ability to create useful working products with it,
adapting models to certain tasks, skill in developing, evaluating, monitoring, and
deploying production ready ML services, and general programming skills."

Each criterion maps to something in this repository:

| Criterion | Where it is shown |
|---|---|
| Adapting models to a task | data generation, training, the comparison against baselines |
| Developing | code structure, tests, commit history |
| Evaluating | metrics, test sets, the results section of the report |
| Monitoring | health and metrics endpoints, request logging |
| Deploying | Dockerfile, optional Space |
| General programming | everything above |

## 7. Acceptance criteria

The project is complete when all of the following hold.

- A1. `docker build` and `docker run` start the service with one documented command each.
- A2. The challenge example returns the expected output through the API.
- A3. Tests pass in the container and cover the API contract, the content-preservation
  invariant (P1), the clean-input case (P4) and the long-input policy (F5).
- A4. Evaluation reports per-class precision, recall and F1 for the four gap classes, on at
  least one synthetic held-out set and one small realistic set, for the do-nothing
  baseline, the rule-based baseline and the learned model.
- A5. Latency and throughput are reported for named input lengths on a named CPU.
- A6. `report.md` explains how to run, the approach, the decisions with reasons, the
  results, the known failures and what would be done next.
- A7. The repository history shows the plan before the code.

## 8. Constraints and assumptions

- No deadline was given. The target is a complete first-round submission, not a research
  project. The design document sets a time budget.
- Development hardware: Apple M1 with 8 GB memory. GPU training, if needed, uses a free
  hosted notebook (Colab, Kaggle or Modal). The model must serve on CPU.
- Assumption: the challenge input was produced by a corruptor that replaced true newlines
  with spaces and inserted newlines at arbitrary character positions. The evidence is the
  leading-space pattern (`\n in`) and the mid-word split. This assumption drives the
  synthetic data generator and is tested against a realistic set.
- Assumption: no clarifying questions are needed. The task e-mail says the openness is
  intentional and the example answers the questions that matter.
