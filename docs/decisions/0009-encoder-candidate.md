# 0009. The encoder candidate for the full fine-tune: deberta-v3-xsmall

Date: 2026-10-02. Status: accepted.

## Context

Design 4.5 selects the pretrained encoder for the M5 fine-tune between two candidates by
one-epoch V1 macro-F1, subject to CPU latency per 256-token window within three times the
scratch model's. Both candidates were trained for one epoch on a Colab T4 (seed 1, 20,000
examples at most, token budget 192; records in `experiments/training/ft-deberta-select.json`
and `ft-distilbert-select.json`). Latency was measured on the M1 Mac CPU by
`scripts/select_encoder.py` (20 timed calls after 3 warm-up calls, one 256-token window
from V3; record `experiments/results/m5-candidates.json`).

## Options

1. `microsoft/deberta-v3-xsmall`.
2. `distilbert-base-cased`.

## Decision

Option 1. The rule: the highest one-epoch V1 macro-F1 among the candidates within the
latency limit. The scratch model's p50 is 22.46 ms per window, so the limit is 3 x 22.46 =
67.38 ms. Both candidates are within it, so the rule reduces to V1 macro-F1.

| candidate | parameters | V1 macro-F1 | V1 wrong joins /1k | V3 damage | p50 ms / window | p95 ms / window | within limit (67.38 ms) |
|---|---:|---:|---:|---:|---:|---:|---|
| `microsoft/deberta-v3-xsmall` | 70,646,404 | 0.9366 | 0.113 | 0.00229 | 65.52 | 76.04 | yes |
| `distilbert-base-cased` | 65,195,524 | 0.9011 | 0.194 | 0.00239 | 51.24 | 53.58 | yes |

DeBERTa leads on V1 macro-F1 by 0.035 and on wrong joins per 1k on V1 (0.113 against
0.194), and both candidates sit under the V3 damage gate of 0.0026 (decision 0006) after one
epoch (0.00229 and 0.00239). It uses 3.8 minutes of T4 time for the epoch against 2.4. The
margin on latency is thin: DeBERTa's p50 of 65.52 ms is 97% of the limit (2.9x the scratch
p50), and its p95 of 76.04 ms is above the limit. The rule is stated on p50, so it passes, but
a rerun on a busier machine could cross it. The chosen model string for the notebook is
`MODEL = "microsoft/deberta-v3-xsmall"`.

The latency condition here is the per-window selection condition of design 4.5 only. The
serving rule of design 5.3 (p50 under 300 ms at 2,000 characters on the host) is a separate
test applied to the trained model in decision 0010; passing this selection does not imply it.

## Consequences

- The full fine-tune uses `microsoft/deberta-v3-xsmall`; the notebook's full-run cells read
  `MODEL` and take this value. DistilBERT is dropped from M5.
- The selection used one epoch and one seed, so the 0.035 gap is not a significance claim;
  the full run reports V1, V2 and V3 for the chosen model only.
- The selection runs were written by transformers 5 on Colab, whose tokenizer config stores
  `extra_special_tokens` as a list; the repository pins transformers below 5, so
  `FinetunedFixer.load` maps the list to the mapping form before loading. Token ids of
  `[NL]` and `[PP]` were checked after loading (DeBERTa 128001 and 128002, DistilBERT 28996
  and 28997).
