# data

Only these are committed: this file, `split.json` (group ids per split), `sets/`
(evaluation sets as JSONL) and `realistic/` (raw and adjusted real passages with
reviewed targets). Everything else is built by `scripts/build_data.py` and published to
the Hugging Face Hub with a manifest of content hashes. See `docs/02-design.md` section 3.

## Licence

The Wikipedia-derived files (`clean/*.jsonl` documents whose source is the dump at the
pinned revision, and the evaluation sets built from them) are derivatives of Wikipedia
text and are released under the same licence, CC BY-SA 4.0, with attribution to Wikipedia
and its contributors. The generated documents (`raw/generated/*.txt`) were written for
this project and are released under CC BY-SA 4.0 as well, so the published dataset has
one licence. The realistic passages under `realistic/` are short excerpts of published
PDFs kept in this repository for evaluation only; they are not part of the published
dataset.

## Current build

- Hub dataset URL: https://huggingface.co/datasets/jalalhussein1982/newline-fixer-data
- Dataset version: 1, published at revision `a57d9702a8f2434cf9fa0248029454dfc60255b7`.
- Wikipedia: 5000 documents, pinned revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` (seed 1; reduced from the 20000 target).
- Generated documents: 2000 (seed 1), written in a Claude session under the pipeline's prompt rules rather than through the API, and cached as `raw/generated/00000.txt` to `01999.txt`; `generate` recorded the model id `claude-cowork` (see `raw/generated.meta.json`). All 2000 passed `validate_generated` and survived filtering and deduplication.
- Splits after filtering and deduplication (seed 1): train 6299, val 350, test 350 documents.
- Evaluation sets under `sets/`: V1 350, V3 100, T1 350, T3 200, T0 1 items. V1 and T1 are capped by the 350-document val and test splits.
- Lexicon: 27948 words built from the train split.

Re-running `assemble` redraws every split, every evaluation set and the lexicon; after it, re-run `evaluate` and `results_table` and update decision 0006 before accepting it.

## Realistic sets

Built on 2026-10-02 from ten real PDFs. Raw passages were extracted with `pdftotext` and
cut (seed 1); targets were proposed in a Claude session (whitespace-only) and then reviewed.

- **V2 (dev)**: 29 passages, 0 unreachable boundaries: word2vec 6, fasttext 7, nist-800-63 8, bash-manual 8.
- **T2 (test)**: 40 passages, 0 unreachable boundaries: attention 8, bert 8, resnet 5, adam 4, nist-ai-rmf 7, gnu-make 8.

Source change: `python-tutorial` (a zip of many PDFs) was replaced in the dev list by
`bash-manual` (GNU Bash Reference Manual). Passages that are mostly non-letters (letter
ratio below 0.6) are dropped at cut time; 8 were cut per document, 80 in total.

Review provenance (see `review.json`): the four dev documents and `attention` were
reviewed by JH; `bert`, `resnet`, `adam`, `nist-ai-rmf` and `gnu-make` were read against
the four review questions by Claude Fable 5.1 on JH's behalf and signed off by JH. Two
targets were edited during review (`adam/00`, `adam/07`: a section number joined with its
heading). Eleven passages were left out as having no sensible newline target:
`word2vec/04`, `word2vec/07`, `fasttext/06` (tables and diagram labels; they were in a
first V2 build and are discussed in decision 0006), `resnet/02`, `resnet/04`,
`resnet/07`, `adam/01`, `adam/02`, `adam/05`, `adam/06` (equation and table debris),
`nist-ai-rmf/03` (a two-column table extracted with its columns interleaved).

To add or redo passages: `extract` and `cut` for the document, `propose` with
`ANTHROPIC_API_KEY` set (or write the target by hand), review, add the key to
`review.json`, then `uv run python scripts/make_realistic_set.py build` and re-run the
evaluation.
