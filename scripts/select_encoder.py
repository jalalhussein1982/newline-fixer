"""Encoder selection (design 4.5): CPU latency per 256-token window against the scratch model.
Usage:
  uv run python scripts/select_encoder.py --runs ft-deberta-select,ft-distilbert-select \\
      --out experiments/results/m5-candidates.json
Needs experiments/runs/current (scratch), experiments/runs/<run> and experiments/training/<run>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

import torch

from newline_fixer.data.records import EvalItem, read_jsonl
from newline_fixer.eval.bench import percentile, select_window, time_calls
from newline_fixer.eval.report import candidates_table
from newline_fixer.models.base import Fixer
from newline_fixer.models.finetuned import FinetunedFixer
from newline_fixer.models.scratch import ScratchFixer

LIMIT_FACTOR = 3.0


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", required=True, help="comma-separated run ids under experiments/runs")
    p.add_argument("--out", default="experiments/results/m5-candidates.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--sets-dir", default="data/sets")
    a = p.parse_args()
    cpu = torch.device("cpu")
    tokens, current = select_window(read_jsonl(Path(a.sets_dir) / "V3.jsonl", EvalItem))

    def timed(fixer: Fixer) -> tuple[float, float]:
        times = time_calls(lambda: fixer.predict(tokens, current), n=a.n, warmup=3)
        return round(percentile(times, 50), 2), round(percentile(times, 95), 2)

    scratch_p50, _ = timed(ScratchFixer.load("experiments/runs/current", cpu))
    limit = LIMIT_FACTOR * scratch_p50
    candidates: dict[str, dict[str, object]] = {}
    for run in a.runs.split(","):
        rec = json.loads(Path(f"experiments/training/{run}.json").read_text(encoding="utf-8"))
        p50, p95 = timed(FinetunedFixer.load(Path("experiments/runs") / run, cpu))
        candidates[run] = {
            "pretrained": rec["pretrained"],
            "V1_macro_f1": rec["best"]["V1_macro_f1"],
            "p50_ms": p50,
            "p95_ms": p95,
            "within_limit": p50 <= limit,
            "n_params": rec["n_params"],
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    record = {
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "dirty": dirty,
        "platform": platform.platform(),
        "window_tokens": len(tokens),
        "scratch_p50_ms": scratch_p50,
        "limit_ms": limit,
        "candidates": candidates,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(candidates_table(record))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
