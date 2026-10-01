# 0001. Formulate the task as four-way gap classification

Date: 2026-10-01. Status: accepted.

## Context

The challenge asks for "fixed newlines" without defining correctness. The one example
shows five operations: three removals of a wrong newline, one of them inside a word, one
inserted line break before a list item, and one inserted paragraph break after a heading.
A formulation was needed that makes all five measurable and that cannot change content.

## Options

1. **Binary per position: newline or not.** Simple, but it cannot express the paragraph
   break in the example, and joining a split word has no representation.
2. **Four-way class per whitespace gap: join, space, newline, paragraph break.** Covers
   every operation in the example. Content preservation holds by construction because
   only gaps change.
3. **Free-form rewriting (sequence to sequence).** Can express anything, including
   hyphen and missing-space repair, but can alter content and needs a repair layer, and
   decoding is slow on CPU.

## Decision

Option 2. Every system, including the rule baseline, implements the same per-gap
interface, so all are measured with the same metrics.

## Consequences

- Breaks that left no whitespace behind and hyphenated splits are out of scope; the
  requirements say so and the realistic set joins hyphens before labelling.
- Labels are derived exactly from clean and corrupted text pairs, which makes the data
  pipeline simple and testable.
- The metric set is fixed: per-class precision, recall and F1 over gaps.
