"""Publish a trained run to the Hugging Face Hub (decision 0003). Usage:
  hf auth login   # once
  uv run python scripts/publish_weights.py --run-id scratch-v1 --repo <user>/newline-fixer-scratch
Prints the commit revision to pin with NF_WEIGHTS=hf:<repo>@<revision> and records it under
the "hub" key of experiments/training/<run-id>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("experiments/runs")
RECORDS = Path("experiments/training")
FILES = ["config.json", "words.json", "chars.json", "model.pt", "run.json", "README.md"]


def model_card(record: dict[str, object], repo: str) -> str:
    best = record.get("best", {})
    assert isinstance(best, dict)
    return (
        "---\nlibrary_name: pytorch\ntags: [text-cleaning, newline-restoration]\n---\n\n"
        f"# newline-fixer from-scratch model ({record['run_id']})\n\n"
        "A character-aware BiLSTM that predicts the whitespace class (join, space, newline, paragraph) "
        "between consecutive tokens of English text. Trained with https://github.com/jalalhussein1982/newline-fixer "
        f"at commit `{str(record.get('git_commit', ''))[:12]}`.\n\n"
        f"- Dev macro-F1 V1 {best.get('V1_macro_f1', 0.0):.3f}, clean-text damage V3 {best.get('V3_damage', 0.0):.4f}\n"
        f"- Parameters {int(str(record.get('n_params', 0))):,}; seed {record.get('seed')}; best epoch {record.get('best_epoch')}\n\n"
        f"Load with `ScratchFixer.load('hf:{repo}@<revision>')`; files: config.json, words.json, chars.json, model.pt, run.json.\n"
    )


def main() -> None:
    from huggingface_hub import HfApi

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--repo", required=True)
    a = p.parse_args()
    run_dir = RUNS / a.run_id
    record_path = RECORDS / f"{a.run_id}.json"
    if not (run_dir / "model.pt").exists():
        raise SystemExit(f"no weights at {run_dir}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    (run_dir / "README.md").write_text(model_card(record, a.repo), encoding="utf-8")
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    info = api.upload_folder(
        folder_path=str(run_dir),
        repo_id=a.repo,
        repo_type="model",
        allow_patterns=FILES,
        commit_message=f"{a.run_id} from {str(record.get('git_commit', ''))[:12]}",
    )
    revision = info.oid
    record["hub"] = {"repo": a.repo, "revision": revision}
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"published https://huggingface.co/{a.repo} revision {revision}")
    print(f"pin with NF_WEIGHTS=hf:{a.repo}@{revision}")
    print(f"record updated: {record_path}")


if __name__ == "__main__":
    main()
