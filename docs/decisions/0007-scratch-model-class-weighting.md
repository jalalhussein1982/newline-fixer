# 0007. Class weighting for the from-scratch model; first trained model selected

Date: 2026-10-02. Status: accepted.

## Context

Design 4.4 fixes the architecture and optimizer of M1 and the risk table says class
weights are tried on V1 because JOIN and PARA are rare. Decision 0006 froze B1 at
macro-F1 0.635 (V1) and 0.806 (V2) with a clean-damage gate of 0.0026 on V3.

## Options

1. Unweighted cross-entropy (`scratch-v1`).
2. Inverse-frequency class weights normalized to mean one (`scratch-v1-inverse`).

## Decision

Option 1: run `scratch-v1` is the current from-scratch model. The selection rule is
best V1 macro-F1 subject to V3 damage at most 0.0026; only `scratch-v1` passes the gate.
Numbers from `experiments/training/*.json` and `experiments/results/m2-scratch.json`
(the `scratch-v1-inverse` row other than V1 macro-F1, V3 damage and V1 wrong-join comes
from a one-off evaluation of its saved weights with the same `scripts/evaluate.py`; it is
not committed because that script only scores the current run):

| run | best epoch | V1 macro-F1 | V1 JOIN F1 | V1 PARA F1 | V2 macro-F1 | V3 damage | wrong-join /1k (V1/V2/V3) |
|---|---|---|---|---|---|---|---|
| scratch-v1 | 8 | 0.922 | 0.974 | 0.861 | 0.733 | 0.0016 | 0.11/9.16/0.00 |
| scratch-v1-inverse | 3 | 0.741 | 0.831 | 0.573 | 0.608 | 0.0447 | 3.15/34.99/0.00 |
| rules (0006) | | 0.635 | 0.682 | 0.410 | 0.806 | 0.0026 | 0.00/0.00/0.00 |

`scratch-v1` passes the gate (0.0016 against 0.0026); `scratch-v1-inverse` fails it
(0.0447). On V1 the winner beats B1 by 0.287 macro-F1 (0.922 against 0.635), JOIN F1 by
0.292 and PARA F1 by 0.451. On V2, the human-reviewed real passages, it does worse than
B1 (0.733 against 0.806) and even than identity (0.753): NL recall is 0.37 and PARA
recall 0.69, and it makes 9.16 wrong joins per thousand gaps (22 of 2,401) where B1
makes none. Paragraph match on V2 is 0.453 against 0.526 for rules. The V1 gain is
therefore real on the synthetic corruptions the model trained on but does not carry to
V2 yet. Three further points belong to the record:

- The V3 gate is met without a wide margin. V3 damage of `scratch-v1` moved between
  epochs: 0.00163, 0.00120, 0.00098, 0.00327, 0.00229, 0.00305, 0.00316, 0.00163 for
  epochs 1 to 8. It was above the gate at epochs 4, 6 and 7 and returned under it at
  epoch 8. Model selection uses V1 macro-F1 only, so epoch 8 passing is
  partly luck of the last step; a different seed or one more epoch could land above 0.0026.
- `scratch-v1` makes wrong joins where B1 makes none: 0.11 per thousand on V1 (13 gaps),
  9.16 on V2 and 0.00 on V3. A wrong join glues two words together and is the worst error
  the product can make; the V2 figure is far above the "under one per thousand" bar that
  decision 0006 applied to B1.
- The inverse weighting hurt every metric. V1 macro-F1 fell from 0.922 to 0.741, V3
  damage rose from 0.0016 to 0.0447 (27 times the winner's, 17 times the gate), and wrong
  joins per thousand rose from 0.11 to 3.15 on V1 (about 30 times) and from 9.16 to 34.99 on V2.
  Its recall on PARA and NL is high (0.95 and 0.90 on V1) but precision is about 0.41
  and 0.43, so it over-inserts breaks and shreds clean text. The run stopped early after
  epoch 5 (patience 2); its best V1 macro-F1 was at epoch 3 and none of its five epochs
  passed the gate (V3 damage 0.037 to 0.058).

Challenge example: `scratch` does not reproduce the expected output. It puts a paragraph
break after `3.2.3` instead of joining the section number to its title with a space, and
puts a paragraph break before the first bullet where the expected output has a single
newline. The rest of the example is correct.

Training facts: cuda (Google Colab T4, `notebooks/train_scratch_colab.ipynb`); design 4.4
said the M1 Mac, which measured 12.1 min/epoch on MPS, so the runs moved to Colab.
`scratch-v1`: 8 epochs, 8.6 minutes (about 52 s per epoch). `scratch-v1-inverse`: 5 epochs
before early stopping, 5.2 minutes. 5,551,692 parameters (word vocabulary 30,002, 383
characters; Task 3 counted 5,549,036 with a 300-character table). 11,470 training
examples per epoch from 6,299 documents, seed 1, class counts in epoch 0 JOIN 16,262,
SPACE 2,057,047, NL 31,125, PARA 47,140.

## Consequences

- `experiments/runs/current` and `NF_WEIGHTS` default to `scratch-v1`; M3 serves it only
  if it passes design 5.3 (V3 gate and latency), otherwise B1. Given the V2 wrong-join
  rate and the narrow V3 margin, B1 stays the safe serving default until a model is
  checked on V2 as well.
- The loss profile suggests the next work is on the V2 gap, not on class weights: more
  realistic corruptions and real-document data for M5, a V3-damage term or a V2-aware
  selection rule, and a wrong-join penalty (for example a higher cost on predicting JOIN)
  for any M1 follow-up. Model selection on V1 alone is also worth revisiting.
- The weights are published in Task 8; this record is updated by a new record, never edited.
