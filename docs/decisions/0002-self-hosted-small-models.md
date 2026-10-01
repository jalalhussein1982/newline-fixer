# 0002. Serve self-hosted small models, compare a from-scratch model with a fine-tuned one

Date: 2026-10-01. Status: superseded by 0004.

## Context

The challenge allows any architecture and any way of obtaining the model, but asks that
the service "efficiently generate reasonable answers" and judges "adapting models to
certain tasks". There is no fixed time budget, and the author wants to own and explain
the model.

## Options

1. **Hosted LLM behind the endpoint.** Best quality on arbitrary text, no training.
   Needs a secret in the container, costs per request, shows no model adaptation, and
   fails the efficiency bar.
2. **One fine-tuned small pretrained encoder.** Fast path to a good model; the model
   internals are inherited rather than understood.
3. **One from-scratch model.** Fully owned, likely weaker than a pretrained encoder.
4. **Both 2 and 3, measured against each other and against rules.** Most work, most
   learning, strongest evidence in the report.

## Decision

Option 4. A hosted LLM is not used for serving. It may propose labels for the realistic
test set, with every label reviewed by hand, and may generate clean source documents.

## Consequences

- Two training pipelines to build and document.
- The report can show the value of pretraining on this task with numbers.
- Serving stays on CPU with no external dependency at request time.
