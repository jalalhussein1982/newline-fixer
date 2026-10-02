"""Render experiments/results/*.json into experiments/README.md."""

from __future__ import annotations

import json
from pathlib import Path

from newline_fixer.eval.table import render_table, render_training_table


def main() -> None:
    parts = ["# Experiments\n", "Rendered by `scripts/results_table.py`; do not edit by hand.\n"]
    for path in sorted(Path("experiments/results").glob("*.json")):
        parts.append(f"\n## {path.stem}\n\n" + render_table(json.loads(path.read_text())))
    training = sorted(Path("experiments/training").glob("*.json"))
    if training:
        parts.append(
            "\n## Training runs\n\n"
            + render_training_table([json.loads(p.read_text()) for p in training])
        )
    Path("experiments/README.md").write_text("\n".join(parts))
    print("wrote experiments/README.md")


if __name__ == "__main__":
    main()
