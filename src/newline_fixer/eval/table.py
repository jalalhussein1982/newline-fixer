"""Render evaluation results as Markdown."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def render_table(results: dict[str, Any]) -> str:
    systems: dict[str, dict[str, Any]] = results["systems"]
    sets: list[str] = results["sets"]
    dirty = " (dirty tree)" if results.get("dirty") else ""
    lines = [
        f"Results at commit `{results.get('git_commit', '')[:12]}`{dirty}, {results.get('run_at', '')}.",
    ]
    hashes: dict[str, str] | None = results.get("sets_sha256")
    if hashes:
        sets_line = "Sets: " + ", ".join(f"{k}={v[:12]}" for k, v in hashes.items())
        meta = results.get("sets_meta")
        if meta:
            sets_line += f" (built from seed {meta.get('seed')} at commit {str(meta.get('git_commit', ''))[:12]})"
        lines.append(sets_line)
    lines += [
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


def render_training_table(records: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| run | commit | class weights | best epoch / run | V1 macro-F1 | V2 macro-F1 | V3 damage | params | device | minutes |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in records:
        best = r.get("best", {})
        v2 = best.get("V2_macro_f1")
        lines.append(
            f"| {r['run_id']} | `{str(r.get('git_commit', ''))[:12]}` | {r.get('train_config', {}).get('class_weights', '')} | "
            f"{r.get('best_epoch', '')} / {len(r.get('epochs', []))} | {best.get('V1_macro_f1', 0.0):.3f} | "
            f"{'' if v2 is None else f'{v2:.3f}'} | {best.get('V3_damage', 0.0):.4f} | {int(r.get('n_params', 0)):,} | "
            f"{r.get('device', '')} | {float(r.get('seconds', 0.0)) / 60:.1f} |"
        )
    return "\n".join(lines) + "\n"


def _mb(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_bench_table(records: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| label | system | mode | device | disk MB | RSS MB | p50 / p95 ms @500 | @2,000 | @10,000 | chars/s (batch 8) | commit |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        for name, s in r["systems"].items():
            lat = s["latency_ms"]
            cells = " | ".join(
                f"{lat[k]['p50']:.1f} / {lat[k]['p95']:.1f}" for k in ("500", "2000", "10000")
            )
            lines.append(
                f"| {r['label']} | {name} | {r['mode']} | {r['device']} | {_mb(s['disk_mb'], 1)} | {_mb(s['rss_mb'], 0)} | "
                f"{cells} | {int(s['throughput_chars_per_s']):,} | `{str(r.get('git_commit', ''))[:12]}`{' (dirty)' if r.get('dirty') else ''} |"
            )
    return "\n".join(lines) + "\n"
