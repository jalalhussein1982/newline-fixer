# 0008. The served model after M3: B1 by the decision rule

Date: 2026-10-02. Status: accepted.

## Context

Design 5.3 chooses the served model on development data only: candidates must have V3
clean damage at most 0.0026 (decision 0006) and p50 latency under 300 ms for a
2,000-character input on the M1 Mac CPU; among candidates the highest V2 macro-F1 is
served, wrong-join rate breaking ties; if no learned model qualifies, B1 is served and the
report says why. M2 produced `scratch-v1` (decision 0007), published at revision
`6c311e757d17e89c80b7b86908043637a4f56e28`. M3 built the service and measured the
service-level numbers (`experiments/bench/`).

## Options

1. Serve `rules` (B1).
2. Serve `scratch` (scratch-v1).
3. Serve `scratch` for its V1 strength and accept the V2 regression.

## Decision

Option 1. The numbers, from `experiments/results/m2-scratch.json` and
`experiments/bench/m1-mac-cpu.json` (as rendered in `experiments/README.md`):

| system | V3 damage (gate 0.0026) | p50 ms @2,000 chars, M1 CPU (limit 300) | candidate | V2 macro-F1 | V2 wrong-join /1k |
|---|---|---|---|---|---|
| rules | 0.0026 | 0.5 | yes | 0.806 | 0.00 |
| scratch | 0.0016 | 40.6 | yes | 0.733 | 9.16 |

Both systems pass both candidate conditions: rules sits exactly at the V3 damage limit,
which "at most" admits, and scratch is well inside both limits (0.0016 damage, 40.6 ms
against 300 ms). Scratch is therefore a candidate that loses, not a model that never
qualified. Rule 2 picks `rules` on V2 macro-F1 (0.806 against 0.733), and the wrong-join
rate points the same way (0.00 against 9.16 per 1k), so no tie-break is needed. Option 3
is rejected because V2 is the real-text set the design named for exactly this choice (risk
table: "models learn the corruptor, not the task"): scratch is strong on the synthetic V1
(macro-F1 0.922 against 0.635 for rules) but below the identity baseline on V2 (0.733
against 0.753). The challenge example: rules reproduces it, scratch does not (decision
0007).

Container numbers (`experiments/bench/container-*.json`, same Mac, HTTP through Docker
Desktop's Linux VM): rules p50 1.9 ms and scratch p50 304.1 ms (p95 318.5) at 2,000
characters; image size 1.18 GB (from `docker image ls`, not in a table); resident memory
after warm-up rules 32 MB, scratch 274 MB (in-process record). The design states the
latency rule on the M1 Mac CPU, so the host figure (40.6 ms) is the gate. The container
figure for scratch is above 300 ms and would fail the limit if it were the gate. That
matters for M6, because the Hugging Face Space runs the image: a scratch deployment there
would sit at the edge of the limit or beyond it, whereas rules has a wide margin (1.9 ms).

## Consequences

- `NF_MODEL` defaults to `rules` in `service/config.py` and in the image. `NF_MODEL=scratch`
  serves the published scratch-v1 for comparison; the demo and the report say which is
  which.
- The learned model does not yet add value on real text (requirement Q2); the report must
  say so. The route to change this decision is a model that beats 0.806 on V2 under the
  gates: a V2-aware selection rule, a wrong-join penalty, more realistic corruptions, or
  the fine-tuned encoder of M5. A new record supersedes this one when that happens.
- Test sets T0 to T3 are evaluated once for all systems after this decision
  (`experiments/results/test-sets.json`); they are not used for any further choice.
