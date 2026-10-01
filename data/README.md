# data

Only these are committed: this file, `split.json` (group ids per split), `sets/`
(evaluation sets as JSONL) and `realistic/` (raw and adjusted real passages with
reviewed targets). Everything else is built by `scripts/build_data.py` and published to
the Hugging Face Hub with a manifest of content hashes. See `docs/02-design.md` section 3.

## Current build

- Dataset version: 1 (partial, Wikipedia only).
- Wikipedia: 5000 documents, pinned revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` (seed 1; reduced from the 20000 target).
- Generated documents: 0. `ANTHROPIC_API_KEY` was unset, so `generate` was skipped (see `raw/generated.meta.json`).
- Splits after filtering and deduplication (seed 1): train 4499, val 250, test 250 documents.
- Evaluation sets under `sets/`: V1 250, V3 100, T1 250, T3 200, T0 1 items. V1 and T1 are capped by the 250-document val and test splits.
- Lexicon: 22726 words built from the train split.

### Pending

```bash
uv run python scripts/build_data.py generate --n 2000 --seed 1
uv run python scripts/build_data.py assemble --seed 1 && uv run python scripts/build_data.py sets --seed 1 && uv run python scripts/build_data.py lexicon
uv run python scripts/build_data.py publish --repo <your-hf-user>/newline-fixer-data
```

## Realistic sets (pending)

Raw passages were extracted with `pdftotext` and cut (seed 1), but no targets exist yet: V2 and T2 are not built. Targets must be proposed by the model (needs `ANTHROPIC_API_KEY`) and then reviewed by hand by the author.

Source change: `python-tutorial` (a zip of many PDFs) was replaced in the dev list by `bash-manual` (GNU Bash Reference Manual). All ten downloads and extractions succeeded.

Cut passages per document (8 each, 80 total; none lost to the filter because every document had enough prose chunks, but the sampled passages changed). Passages that are mostly non-letters (letter ratio below 0.6: indexes, tables, formulas) are filtered out at cut time. The reviewer should still skip any passage without a sensible newline target, so V2 may end below 25 passages and more documents may be needed.

| Role | Document | Passages |
|------|----------|----------|
| dev | word2vec | 8 |
| dev | fasttext | 8 |
| dev | nist-800-63 | 8 |
| dev | bash-manual | 8 |
| test | attention | 8 |
| test | bert | 8 |
| test | resnet | 8 |
| test | adam | 8 |
| test | nist-ai-rmf | 8 |
| test | gnu-make | 8 |

To finish, from the repository root:

1. For each doc (word2vec, fasttext, nist-800-63, bash-manual, attention, bert, resnet, adam, nist-ai-rmf, gnu-make), with `ANTHROPIC_API_KEY` set:
   `uv run python scripts/make_realistic_set.py propose --doc <doc>`
2. Review: edit `data/realistic/<doc>/<nn>.target.txt` next to its `.input.txt`, then add `"<doc>/<nn>"` to `data/realistic/review.json` with your initials and the date (`{"reviewer": "JH", "date": "...", "note": ""}`). Only reviewed passages enter the sets.
3. `uv run python scripts/make_realistic_set.py build`, then record the printed unreachable counts here.
