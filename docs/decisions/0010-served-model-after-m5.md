# 0010. The served model after M5: the fine-tuned encoder by the decision rule

Date: 2026-10-02. Status: accepted; supersedes 0008.

## Context

Decision 0008 applied design 5.3 to `rules` and `scratch` and served B1, because scratch-v1
lost on V2 macro-F1 (0.733 against 0.806). It named the fine-tuned encoder of M5 as one route
to a better answer. M5 added `finetuned`: `microsoft/deberta-v3-xsmall`, chosen in decision
0009, fine-tuned for three epochs (best epoch 3) on the same training data and window scheme
as scratch-v1. Design 5.3 is unchanged: candidates need V3 clean damage at most 0.0026 and
host p50 under 300 ms for a 2,000-character input; among candidates the highest V2 macro-F1
is served, wrong-join rate breaking ties.

Per decision 0004, M5 also trained the same encoder from random initialization with the
identical data and schedule (`finetuned-ablation`). It isolates the effect of pretraining and
is evidence, not a serving candidate.

## Options

1. Keep `rules` (B1), as in 0008.
2. Serve `finetuned`.
3. Serve `scratch` (scratch-v1).

## Decision

Option 2. Numbers from `experiments/results/m5-finetuned.json` and
`experiments/bench/m1-mac-cpu.json` (re-measured for all four host rows in one record, so
rules is 0.4 ms here where 0008 quoted 0.5 and scratch 40.5 where it quoted 40.6), as rendered
in `experiments/README.md`:

| system | V3 damage (gate 0.0026) | p50 ms @2,000 chars, M1 CPU (limit 300) | candidate | V2 macro-F1 | V2 wrong-join /1k | V1 macro-F1 |
|---|---|---|---|---|---|---|
| rules | 0.0026 | 0.4 | yes | 0.806 | 0.00 | 0.635 |
| scratch | 0.0016 | 40.5 | yes | 0.733 | 9.16 | 0.922 |
| finetuned | 0.0008 | 97.1 | yes | 0.898 | 4.58 | 0.954 |
| finetuned-ablation | 0.0057 | not measured | no (damage over the gate; not a candidate by design) | 0.614 | 6.66 | 0.839 |

Rule 1 admits rules, scratch and finetuned. Finetuned is well inside both limits (damage
0.0008 against 0.0026, 97.1 ms against 300 ms). Rule 2 picks it on V2 macro-F1: 0.898
against 0.806 for rules, 0.733 for scratch and 0.753 for the identity baseline. Unlike
scratch it is above identity on V2, by 0.145, which is what 0008 said a replacement had to
show. The wrong-join tie-break is not needed to choose, and it points the other way: finetuned
makes 4.58 wrong joins per 1k gaps on V2 where rules makes none. The score gain is real, and
so is this cost (see the caveats below).

The ablation reached V3 damage 0.0057 (about 2.2 times the gate), so it would not have
qualified even if it had been a candidate.

### What pretraining is worth

Fine-tuned against random-init, same architecture (70.6M parameters), data and schedule, so
the difference is the measured value of pretraining:

| set | metric | finetuned | random-init | difference |
|---|---|---|---|---|
| V1 | macro-F1 | 0.954 | 0.839 | +0.115 |
| V2 | macro-F1 | 0.898 | 0.614 | +0.284 |
| V3 | damage | 0.0008 | 0.0057 | 0.0049 lower |

The effect is largest on V2, the real-text set: random-init falls 0.139 below identity there
(0.614 against 0.753), while the pretrained encoder lands 0.145 above it. That gap is not
explained by capacity or data. Wrong joins on V1 are 0.02 per 1k fine-tuned and 0.68 random-init.
Pretraining is what moves the encoder from learning the corruptor toward learning the task.
This is one seed per arm; the V2 difference is large enough that seed noise is unlikely to
reverse its sign, but the size of the gap is a single measurement.

### Caveats

- Wrong joins. 4.58 per 1k on V2 is below scratch (9.16) and above rules (0.00). A wrong join
  merges two lines that were separate and is the worst error this service can make. On T2-like
  text (real documents with unusual layout) scratch reached 6.58 per 1k in
  `experiments/results/test-sets.json`; finetuned has not been measured on T2 yet, and T0 to T3 are not used for any choice. Expect the same shape there and treat it as the risk to report.
- V2 damage is 0.0641 for finetuned and 0.0604 for rules: the gain in macro-F1 does not come
  with a reduction in damage on real text.
- The challenge example. `finetuned` does not reproduce `EXAMPLE_OUTPUT` exactly. Its output
  has the heading and the lead-in sentence right but puts a blank line (PARA) before the first
  bullet where the expected output has a single newline (NL). Rules reproduces it (0008).
  The example is one input and V2 is the evidence, but a reader who tries the README example
  will see the difference.
- Cost. 291 MB of weights and about 570 MB resident memory against 22.6 MB and 281 MB for
  scratch. The container latency of the encoder has not been measured; scratch ran 7.5 times
  slower in the container than on the host (304.1 against 40.5 ms), and the same ratio would
  put finetuned at roughly 700 ms there. That is an extrapolation, not a measurement, and the
  design states the latency rule on the host, which finetuned passes. It is the item to measure
  before the M6 Space deployment.
- `experiments/results/m5-candidates.json` (the encoder selection of 0009) was measured from a
  dirty tree (untracked run records and an uncommitted loader fix); the latency path was
  unaffected.

## Consequences

- The served model becomes `finetuned`, superseding 0008. The default does not flip in this
  commit: `DEFAULT_MODEL` in `service/config.py` and `NF_MODEL` in the Dockerfile stay `rules`
  because the weights are not published yet (`PUBLISHED_REVISION_FINETUNED` is empty) and the
  image cannot bake in unpublished weights. Task 7 publishes the weights to the Hub, pins the
  revision, and flips both defaults and the README sentence; until then `NF_MODEL=finetuned`
  with `NF_WEIGHTS_FINETUNED` pointing at a local copy serves the model.
- The report says: the served model is a pretrained encoder fine-tuned on synthetic corruption,
  not the from-scratch model; it beats the rules baseline on real text (V2 0.898 against 0.806)
  and the pretraining ablation shows why; it makes wrong joins the rules do not (4.58 per 1k on
  V2), does not reproduce the challenge example exactly, and is far heavier than the rules.
- M6 should measure the encoder in the container before choosing the Space hardware. ONNX
  export is worth trying only if latency is what stands in the way of serving this model
  there; it is not needed for the host rule, which passes at 97.1 ms.
- Test sets T0 to T3 are evaluated for `finetuned` once, after Task 7, and are not used for
  any further choice.
