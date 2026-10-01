"""Render experiments/results/*.json into experiments/README.md."""

from __future__ import annotations

import json
from pathlib import Path

from newline_fixer.eval.table import render_table


def main() -> None:
    parts = ["# Experiments\n", "Rendered by `scripts/results_table.py`; do not edit by hand.\n"]
    for path in sorted(Path("experiments/results").glob("*.json")):
        parts.append(f"\n## {path.stem}\n\n" + render_table(json.loads(path.read_text())))
    Path("experiments/README.md").write_text("\n".join(parts))
    print("wrote experiments/README.md")


if __name__ == "__main__":
    main()
