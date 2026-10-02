"""Print the tables report.md embeds, rendered from committed records. Usage:
  uv run python scripts/report_tables.py > "$TMPDIR/tables.md"
Paste each section into report.md under the matching heading; never edit a number by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from newline_fixer.eval.report import (
    per_class_table,
    realistic_facts,
    service_table,
    severity_table,
    summary_table,
    training_table,
)

SYSTEMS = ["identity", "rules", "scratch"]


def load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def main() -> None:
    dev = load("experiments/results/m2-scratch.json")
    test = load("experiments/results/test-sets.json")
    print("## dev sets\n\n" + summary_table(dev, ["V1", "V2", "V3"], SYSTEMS))
    print("## test sets\n\n" + summary_table(test, ["T0", "T1", "T2", "T3"], SYSTEMS))
    print("## per class, dev sets\n\n" + per_class_table(dev, ["V1", "V2"], SYSTEMS))
    print("## per class, test sets\n\n" + per_class_table(test, ["T1", "T2"], SYSTEMS))
    print("## T1 by severity band\n\n" + severity_table(test, "T1", ["rules", "scratch"]))
    bench = [load(str(p)) for p in sorted(Path("experiments/bench").glob("*.json"))]
    print("## service\n\n" + service_table(bench))
    training = [load(str(p)) for p in sorted(Path("experiments/training").glob("*.json"))]
    print("## training\n\n" + training_table(training))
    facts = realistic_facts(
        Path("data/realistic"),
        load("data/realistic/review.json"),
        load("data/realistic/sources.json"),
    )
    fence = "`" * 3
    print(f"## realistic sets\n\n{fence}json\n{json.dumps(facts, indent=2)}\n{fence}")


if __name__ == "__main__":
    main()
