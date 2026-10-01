# data

Only these are committed: this file, `split.json` (group ids per split), `sets/`
(evaluation sets as JSONL) and `realistic/` (raw and adjusted real passages with
reviewed targets). Everything else is built by `scripts/build_data.py` and published to
the Hugging Face Hub with a manifest of content hashes. See `docs/02-design.md` section 3.
