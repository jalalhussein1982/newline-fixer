"""Train the from-scratch model. Usage:
  uv run python scripts/train_scratch.py --run-id scratch-v1 [--epochs 8 --batch 32 --lr 2e-3 --seed 1
      --class-weights none|inverse --max-docs N --device cpu|mps]
Writes weights to experiments/runs/<run-id>/ (git-ignored) and the record to
experiments/training/<run-id>.json (committed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newline_fixer.data.records import CleanDoc, EvalItem, read_jsonl
from newline_fixer.models.device import select_device
from newline_fixer.models.scratch_config import ScratchConfig
from newline_fixer.models.trainer import TrainConfig, train

CLEAN = Path("data/clean")
SETS = Path("data/sets")
RUNS = Path("experiments/runs")
RECORDS = Path("experiments/training")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--class-weights", default="none", choices=["none", "inverse"])
    p.add_argument("--max-docs", type=int)
    p.add_argument("--device")
    a = p.parse_args()
    tcfg = TrainConfig(
        run_id=a.run_id,
        epochs=a.epochs,
        batch_size=a.batch,
        lr=a.lr,
        seed=a.seed,
        class_weights=a.class_weights,
        max_docs=a.max_docs,
    )
    dev = {s: read_jsonl(SETS / f"{s}.jsonl", EvalItem) for s in ("V1", "V3")}
    docs = read_jsonl(CLEAN / "train.jsonl", CleanDoc)
    device = select_device(a.device)
    print(f"training {a.run_id} on {device} with {len(docs)} documents")
    record = train(ScratchConfig(), tcfg, docs, dev, RUNS / a.run_id, device)
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"{a.run_id}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch {record['best_epoch']}: {record['best']}")


if __name__ == "__main__":
    main()
