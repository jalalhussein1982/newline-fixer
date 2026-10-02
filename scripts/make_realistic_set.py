"""Build the realistic sets. Usage:

uv run python scripts/make_realistic_set.py extract --doc attention --pdf ~/Downloads/1706.03762.pdf
uv run python scripts/make_realistic_set.py cut --doc attention --seed 1
uv run python scripts/make_realistic_set.py propose --doc attention [--model ID]
# review: edit data/realistic/<doc>/<nn>.target.txt, then add "<doc>/<nn>" to review.json
uv run python scripts/make_realistic_set.py build
"""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

from newline_fixer.data.realistic import (
    cut_raw_passages,
    join_hyphenation,
    load_reviewed,
    propose_target,
)
from newline_fixer.data.records import write_jsonl

ROOT = Path("data/realistic")
SETS = Path("data/sets")


def cmd_extract(args: argparse.Namespace) -> None:
    out = ROOT / args.doc / "full.raw.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftotext", "-enc", "UTF-8", args.pdf, str(out)], check=True)
    print(f"extracted {out} ({out.stat().st_size} bytes)")


def cmd_cut(args: argparse.Namespace) -> None:
    doc_dir = ROOT / args.doc
    raw = (doc_dir / "full.raw.txt").read_text()
    for i, passage in enumerate(cut_raw_passages(raw, random.Random(args.seed))):
        (doc_dir / f"{i:02d}.raw.txt").write_text(passage)
        adjusted, _ = join_hyphenation(passage)
        (doc_dir / f"{i:02d}.input.txt").write_text(adjusted)
    (doc_dir / "full.raw.txt").unlink()  # the full document is not committed
    print(f"cut {args.doc}: {len(list(doc_dir.glob('*.input.txt')))} passages")


def cmd_propose(args: argparse.Namespace) -> None:
    for inp in sorted((ROOT / args.doc).glob("*.input.txt")):
        target = inp.with_name(inp.name.replace(".input.", ".target."))
        if target.exists():
            continue
        target.write_text(propose_target(inp.read_text(), args.model))
        print(f"proposed {target}")


def cmd_build(args: argparse.Namespace) -> None:
    for role, name in (("dev", "V2"), ("test", "T2")):
        items = load_reviewed(ROOT, role)
        write_jsonl(SETS / f"{name}.jsonl", items)
        unreachable = sum(int(i.meta["unreachable"]) for i in items)  # type: ignore[call-overload]
        print(f"{name}: {len(items)} reviewed passages, {unreachable} unreachable boundaries")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--doc", required=True)
    e.add_argument("--pdf", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--doc", required=True)
    c.add_argument("--seed", type=int, default=1)
    pr = sub.add_parser("propose")
    pr.add_argument("--doc", required=True)
    pr.add_argument("--model", default="claude-sonnet-5-5")
    sub.add_parser("build")
    args = p.parse_args()
    {"extract": cmd_extract, "cut": cmd_cut, "propose": cmd_propose, "build": cmd_build}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
