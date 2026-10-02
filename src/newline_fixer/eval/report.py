"""Tables for report.md, rendered from committed records so no number is typed by hand."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .table import _mb  # dash for None, used by the bench table


def _provenance(results: dict[str, Any]) -> str:
    dirty = " (dirty tree)" if results.get("dirty") else ""
    sets = ", ".join(f"{k}={v[:12]}" for k, v in results.get("sets_sha256", {}).items())
    return f"Rendered from commit `{str(results.get('git_commit', ''))[:12]}`{dirty}, sets {sets}."


def summary_table(results: dict[str, Any], sets: Sequence[str], systems: Sequence[str]) -> str:
    lines = [
        _provenance(results),
        "",
        "| set | system | gaps | macro-F1 | break-F1 | PARA F1 | wrong-join /1k | clean damage | paragraph match |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sets:
        for name in systems:
            r = results["systems"][name][s]
            g = r["gap"]
            lines.append(
                f"| {s} | {name} | {g['n_gaps']} | {g['macro_f1']:.3f} | {g['break_f1']:.3f} | "
                f"{g['per_class']['PARA']['f1']:.3f} | {g['wrong_join_per_1000']:.2f} | "
                f"{g['damage_rate']:.4f} | {r['paragraph_match_rate']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def severity_table(results: dict[str, Any], set_name: str, systems: Sequence[str]) -> str:
    lines = [
        _provenance(results),
        "",
        "| severity band | system | items | gaps | macro-F1 | wrong-join /1k | damage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    bands = list(results["systems"][systems[0]][set_name]["by_severity"])
    for band in bands:
        for name in systems:
            b = results["systems"][name][set_name]["by_severity"][band]
            lines.append(
                f"| {band} | {name} | {b['n_items']} | {b['n_gaps']} | {b['macro_f1']:.3f} | "
                f"{b['wrong_join_per_1000']:.2f} | {b['damage_rate']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def service_table(records: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| label | system | p50 / p95 ms @500 | @2,000 | @10,000 | chars/s (batch 8) | RSS MB | disk MB | commit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        dirty = " (dirty)" if r.get("dirty") else ""
        for name, s in r["systems"].items():
            lat = s["latency_ms"]
            cells = " | ".join(
                f"{lat[k]['p50']:.1f} / {lat[k]['p95']:.1f}" for k in ("500", "2000", "10000")
            )
            lines.append(
                f"| {r['label']} | {name} | {cells} | {int(s['throughput_chars_per_s']):,} | "
                f"{_mb(s.get('rss_mb'), 0)} | {_mb(s.get('disk_mb'), 1)} | `{str(r.get('git_commit', ''))[:12]}`{dirty} |"
            )
    return "\n".join(lines) + "\n"


def training_table(records: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| run | device | epochs | minutes | best epoch | V1 macro-F1 | V3 damage | params | commit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        best = r.get("best", {})
        lines.append(
            f"| {r['run_id']} | {r.get('device', '')} | {len(r.get('epochs', []))} | "
            f"{float(r.get('seconds', 0.0)) / 60:.1f} | {r.get('best_epoch', '')} | "
            f"{best.get('V1_macro_f1', 0.0):.3f} | {best.get('V3_damage', 0.0):.4f} | "
            f"{int(r.get('n_params', 0)):,} | `{str(r.get('git_commit', ''))[:12]}` |"
        )
    return "\n".join(lines) + "\n"


def realistic_facts(
    realistic_dir: Path, review: dict[str, Any], sources: dict[str, Any]
) -> dict[str, Any]:
    """Per set: passages in the set, passages adjusted for hyphenation, passages excluded, documents."""
    facts: dict[str, Any] = {}
    for set_name, split in (("V2", "dev"), ("T2", "test")):
        docs = [entry["doc"] for entry in sources.get(split, [])]
        passages = adjusted = excluded = 0
        for doc in docs:
            for raw in sorted((realistic_dir / doc).glob("*.raw.txt")):
                stem = raw.name.split(".")[0]
                if f"{doc}/{stem}" not in review:
                    excluded += 1
                    continue
                passages += 1
                input_path = raw.with_name(f"{stem}.input.txt")
                adjusted += int(
                    raw.read_text(encoding="utf-8") != input_path.read_text(encoding="utf-8")
                )
        facts[set_name] = {
            "passages": passages,
            "adjusted": adjusted,
            "excluded": excluded,
            "documents": docs,
        }
    return facts
