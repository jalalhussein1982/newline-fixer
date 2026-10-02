"""Evaluate systems on sets. Usage:
uv run python scripts/evaluate.py --systems identity,rules --sets V1,V2,V3 --out experiments/results/m1-baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newline_fixer.eval.runner import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", required=True)
    p.add_argument("--sets", required=True)
    p.add_argument("--sets-dir", default="data/sets")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    results = run(a.systems.split(","), a.sets.split(","), Path(a.sets_dir))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
