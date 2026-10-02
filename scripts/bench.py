"""Service benchmark (design 5.2). Usage:
  uv run python scripts/bench.py --systems identity,rules,scratch --label m1-mac-cpu --out experiments/bench/m1-mac-cpu.json
  uv run python scripts/bench.py --url http://localhost:8000 --label container-rules --out experiments/bench/container-rules.json
In-process mode loads each system on CPU (scratch from NF_WEIGHTS or the published revision);
HTTP mode measures whatever model the server at --url serves.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

from newline_fixer.data.records import EvalItem, read_jsonl
from newline_fixer.eval.bench import bench_fixer, bench_http, bench_inputs
from newline_fixer.service.config import Settings, load_fixer


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--systems", default="rules")
    p.add_argument("--url")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--sets-dir", default="data/sets")
    a = p.parse_args()
    inputs = bench_inputs(read_jsonl(Path(a.sets_dir) / "V3.jsonl", EvalItem))
    systems: dict[str, object] = {}
    if a.url:
        name, rec = bench_http(a.url, inputs, n=a.n)
        systems[name] = rec
    else:
        for name in a.systems.split(","):
            fixer = load_fixer(Settings(model=name, device="cpu"))
            systems[name] = bench_fixer(fixer, inputs, n=a.n)
            print(name, json.dumps(systems[name]))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    record = {
        "label": a.label,
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mode": "http" if a.url else "in-process",
        "device": "cpu",
        "systems": systems,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
