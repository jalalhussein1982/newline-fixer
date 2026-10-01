# 0006. The rules baseline is frozen

Date: 2026-10-02. Status: proposed. Becomes accepted when V2 numbers are added after the realistic targets are reviewed.

## Context

Design 4.3 defines B1 and says it is checked on V1 and V2, then frozen so later
improvements go into the models and the comparison stays honest.

## Options

1. Keep tuning B1 alongside the models.
2. Freeze B1 at the first version that beats identity on V1 and V2 and keeps wrong joins
   under one per thousand gaps on V3.

## Decision

Option 2. B1 is frozen at commit `a3e5a6b`. Numbers on the dev sets at that commit:

| set | macro-F1 identity | macro-F1 rules | wrong-join /1k rules | damage rules (V3) |
|---|---|---|---|---|
| V1 | 0.435 | 0.577 | 0.00 | |
| V2 | pending | pending | pending | |
| V3 | 1.000 | 0.805 | 0.00 | 0.0110 |

Changes made while checking: none. Rules beat identity on V1, identity had damage 0.0000
and str≠norm 0.000 on V3, and wrong joins on V3 were 0.00 per thousand, so no threshold
(`MAX_HEADING_TOKENS`, `STOP`, `min_count`) was touched. V2 does not exist yet.

## Consequences

- Any later change to `rules.py` needs a new decision record and re-runs every table.
- The clean-damage threshold for the serving rule (design 5.3) is set from the rules
  number on V3: 0.0110 (about 11 changed gaps per thousand, higher than the roughly one
  per thousand the design expected; the damage is mostly PARA/NL gaps on clean text,
  V3 PARA F1 0.746).
- On the unit-test-sized set T0 rules scores macro-F1 0.745 (identity 0.231) but does not
  reproduce the example output: it leaves "que ries" unjoined because the bundled
  Wikipedia-only lexicon lacks "queries". Lexicon and rules were not changed.
