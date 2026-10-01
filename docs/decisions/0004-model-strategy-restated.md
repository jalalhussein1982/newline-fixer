# 0004. Model strategy restated as trade-offs; add an ablation that isolates pretraining

Date: 2026-10-01. Status: accepted. Supersedes 0002.

## Context

Record 0002 chose to serve self-hosted small models and to compare a from-scratch model
with a fine-tuned pretrained encoder. An external review found three claims in it
stronger than the evidence: that a hosted LLM necessarily gives the best quality, that it
necessarily fails the efficiency bar, and that fine-tuning implies not understanding the
model. It also noted that comparing a BiLSTM with a pretrained transformer does not
isolate the effect of pretraining, because architecture and capacity differ too.

## Options

1. Keep 0002 as written.
2. Restate the reasoning as trade-offs and hypotheses, keep the same choice, and add one
   ablation run that trains the chosen M2 encoder from random initialization.
3. Drop the second model to reduce scope.

## Decision

Option 2. The choice stands; the reasoning is corrected.

- A hosted LLM is expected, not proven, to give the highest raw quality on arbitrary
  text. It is not served because it adds a per-request external dependency and cost, a
  secret in the container, and shows no model adaptation, which the challenge judges.
  Whether it would meet a latency target is untested and does not need to be.
- Fine-tuning a pretrained encoder is a complete and legitimate approach. The from-scratch
  model is kept because the author wants to own one model end to end, and because the
  comparison is informative for the report.
- One extra run, the M2 encoder from random initialization with identical data and
  schedule, isolates the contribution of the pretrained weights.

## Consequences

- The report presents three learned results: M1, M2, and M2 from random initialization.
- Milestones are ordered so that a submittable state exists after the first trained
  model and the service, before the second model.
