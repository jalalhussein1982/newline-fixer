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
| V1 | 0.435 | 0.577 | 0.00 | 0.0144 |
| V2 | pending | pending | pending | |
| V3 | 1.000 | 0.805 | 0.00 | 0.0110 |

Changes made while checking: none. Rules beat identity on V1, identity had damage 0.0000
and str≠norm 0.000 on V3, and wrong joins on V3 were 0.00 per thousand, so no threshold
(`MAX_HEADING_TOKENS`, `STOP`, `min_count`) was touched. V2 does not exist yet.

Where rules trails identity on V1: PARA F1 0.408 vs 0.439; paragraph match 0.142 vs 0.158. Macro-F1 and break-F1 are higher for rules.

Attribution of the 101 gaps rules changes on clean V3 (damage 0.0110): 75 are rule 4 upgrading NL to PARA after a one- or two-token title-case line (a Wikipedia heading followed by a list); 11 are rule 3 inserting NL before an en dash after `m.`; 9 are rule 2; 6 are rule 4 firing on a bare-integer line start because `SECTION_NUMBER` accepts `\d+` without a dot.

## Consequences

- Any later change to `rules.py` needs a new decision record and re-runs every table.
- The clean-damage threshold for the serving rule (design 5.3) is set from the rules
  number on V3: 0.0110 (about 11 changed gaps per thousand, higher than the roughly one
  per thousand the design expected; the damage is mostly PARA/NL gaps on clean text,
  V3 PARA F1 0.746).
- Choice at acceptance (author): keep B1 as written and accept a clean-damage gate of 0.0110, about eleven times the design's expectation of one per thousand, or tighten rule 4 (require a dotted section number when there is no break; treat a title-case line as a heading only when the current gap is PARA or the next line starts a list) and re-run before accepting.
- On the unit-test-sized set T0 rules scores macro-F1 0.745 (identity 0.231) but does not
  reproduce the example output: it leaves "que ries" unjoined because the bundled
  Wikipedia-only lexicon lacks "queries". Lexicon and rules were not changed.
