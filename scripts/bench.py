"""Service benchmark (design 5.2). Usage:
  uv run python scripts/bench.py --systems identity,rules,scratch --label m1-mac-cpu --out experiments/bench/m1-mac-cpu.json
  uv run python scripts/bench.py --url http://localhost:8000 --label container-rules --out experiments/bench/container-rules.json
In-process mode loads each system on CPU in its own subprocess (--worker), so memory is per system
(scratch and finetuned from NF_WEIGHTS / NF_WEIGHTS_<MODEL> or the published revision);
HTTP mode measures whatever model the server at --url serves.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
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
    p.add_argument("--label", default="")
    p.add_argument("--out", default="")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--sets-dir", default="data/sets")
    p.add_argument(
        "--worker",
        action="store_true",
        help="benchmark the single system in --systems; print its record as JSON",
    )
    a = p.parse_args()
    inputs = bench_inputs(read_jsonl(Path(a.sets_dir) / "V3.jsonl", EvalItem))
    if a.worker:
        fixer = load_fixer(
            Settings.from_env({**os.environ, "NF_MODEL": a.systems, "NF_DEVICE": "cpu"})
        )
        print(json.dumps(bench_fixer(fixer, inputs, n=a.n)))
        return
    if not a.label or not a.out:
        p.error("--label and --out are required")
    systems: dict[str, object] = {}
    if a.url:
        name, rec = bench_http(a.url, inputs, n=a.n)
        systems[name] = rec
    else:
        for name in a.systems.split(","):
            cmd = [
                sys.executable,
                "scripts/bench.py",
                "--worker",
                "--systems",
                name,
                "--n",
                str(a.n),
                "--sets-dir",
                a.sets_dir,
            ]
            stdout = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout
            systems[name] = json.loads(stdout.strip().splitlines()[-1])
            print(name, json.dumps(systems[name]))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    record = {
        "label": a.label,
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "dirty": dirty,
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
