"""Render evaluation results as Markdown."""

from __future__ import annotations

from typing import Any


def render_table(results: dict[str, Any]) -> str:
    systems: dict[str, dict[str, Any]] = results["systems"]
    sets: list[str] = results["sets"]
    lines = [
        f"Results at commit `{results.get('git_commit', '')[:12]}`, {results.get('run_at', '')}.",
        "",
        "| set | system | gaps | macro-F1 | classes | break-F1 | JOIN F1 | PARA F1 | wrong-join /1k | damage | str≠raw | str≠norm | para match |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sets:
        for name, per_set in systems.items():
            r = per_set[s]
            g = r["gap"]
            pc = g["per_class"]
            lines.append(
                f"| {s} | {name} | {g['n_gaps']} | {g['macro_f1']:.3f} | {','.join(g['macro_classes'])} | "
                f"{g['break_f1']:.3f} | {pc['JOIN']['f1']:.3f} | {pc['PARA']['f1']:.3f} | "
                f"{g['wrong_join_per_1000']:.2f} | {g['damage_rate']:.4f} | "
                f"{r['string_changed_vs_raw']:.3f} | {r['string_changed_vs_normalized']:.3f} | "
                f"{r['paragraph_match_rate']:.3f} |"
            )
    return "\n".join(lines) + "\n"
