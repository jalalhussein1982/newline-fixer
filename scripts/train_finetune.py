"""Fine-tune a pretrained encoder. Usage:
  uv run python scripts/train_finetune.py --run-id ft-v1 --model distilbert-base-cased
      [--random-init --epochs 3 --batch 16 --lr 5e-5 --seed 1 --max-docs N --max-examples N
      --device cuda|cpu]
Writes weights to experiments/runs/<run-id>/ (git-ignored) and the record to
experiments/training/<run-id>.json (committed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newline_fixer.data.records import CleanDoc, EvalItem, read_jsonl
from newline_fixer.models.device import select_device
from newline_fixer.models.finetune_trainer import FinetuneTrainConfig, train_finetune

CLEAN = Path("data/clean")
SETS = Path("data/sets")
RUNS = Path("experiments/runs")
RECORDS = Path("experiments/training")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-docs", type=int)
    p.add_argument("--max-examples", type=int)
    p.add_argument("--device")
    a = p.parse_args()
    tcfg = FinetuneTrainConfig(
        run_id=a.run_id,
        pretrained=a.model,
        random_init=a.random_init,
        epochs=a.epochs,
        batch_size=a.batch,
        lr=a.lr,
        seed=a.seed,
        max_docs=a.max_docs,
        max_examples=a.max_examples,
    )
    dev = {s: read_jsonl(SETS / f"{s}.jsonl", EvalItem) for s in ("V1", "V3")}
    docs = read_jsonl(CLEAN / "train.jsonl", CleanDoc)
    device = select_device(a.device)
    print(f"fine-tuning {a.run_id} ({a.model}) on {device} with {len(docs)} documents")
    record = train_finetune(tcfg, docs, dev, RUNS / a.run_id, device)
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"{a.run_id}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch {record['best_epoch']}: {record['best']}")


if __name__ == "__main__":
    main()
