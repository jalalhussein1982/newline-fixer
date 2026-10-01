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
