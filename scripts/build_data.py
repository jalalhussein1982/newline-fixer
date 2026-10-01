"""Build the datasets. Usage:

  uv run python scripts/build_data.py wikipedia --n 20000 --seed 1
  uv run python scripts/build_data.py generate --n 2000 --seed 1 [--model ID]
  uv run python scripts/build_data.py assemble --seed 1
  uv run python scripts/build_data.py sets --seed 1
  uv run python scripts/build_data.py lexicon
  uv run python scripts/build_data.py publish --repo USER/newline-fixer-data

Outputs under data/ (see data/README.md for what is committed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from newline_fixer.corrupt import CorruptConfig
from newline_fixer.data.build import assemble, build_lexicon, make_clean_set, make_corrupted_set
from newline_fixer.data.generated import generate_docs
from newline_fixer.data.manifest import file_sha256, write_manifest
from newline_fixer.data.records import CleanDoc, EvalItem, read_jsonl, write_jsonl
from newline_fixer.data.wikipedia import iter_wikipedia, resolve_revision
from newline_fixer.example import EXAMPLE_INPUT, EXAMPLE_OUTPUT

DATA = Path("data")
RAW = DATA / "raw"
CLEAN = DATA / "clean"
SETS = DATA / "sets"
LEXICON = Path("src/newline_fixer/resources/lexicon.txt")
DATASET_VERSION = "1"


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def cmd_wikipedia(args: argparse.Namespace) -> None:
    revision = args.revision or resolve_revision()
    docs = list(iter_wikipedia(args.n, args.seed, revision))
    n = write_jsonl(RAW / "wikipedia.jsonl", docs)
    (RAW / "wikipedia.meta.json").write_text(
        json.dumps({"revision": revision, "n": n, "seed": args.seed})
    )
    print(f"wikipedia: {n} docs, revision {revision}")


def cmd_generate(args: argparse.Namespace) -> None:
    docs = generate_docs(args.n, args.seed, RAW / "generated", args.model)
    n = write_jsonl(RAW / "generated.jsonl", docs)
    (RAW / "generated.meta.json").write_text(
        json.dumps(
            {"model": args.model, "n": n, "seed": args.seed, "date": dt.date.today().isoformat()}
        )
    )
    print(f"generated: {n} docs")


def cmd_assemble(args: argparse.Namespace) -> None:
    docs = read_jsonl(RAW / "wikipedia.jsonl", CleanDoc) + read_jsonl(
        RAW / "generated.jsonl", CleanDoc
    )
    parts = assemble(docs, seed=args.seed)
    for side, ds in parts.items():
        write_jsonl(CLEAN / f"{side}.jsonl", ds)
        print(f"{side}: {len(ds)} docs")
    split = {side: sorted({d.group for d in ds}) for side, ds in parts.items()}
    (DATA / "split.json").write_text(
        json.dumps({"seed": args.seed, "groups": split}, indent=0) + "\n"
    )


def cmd_sets(args: argparse.Namespace) -> None:
    val = read_jsonl(CLEAN / "val.jsonl", CleanDoc)
    test = read_jsonl(CLEAN / "test.jsonl", CleanDoc)
    write_jsonl(SETS / "V1.jsonl", make_corrupted_set(val, args.seed, limit=500))
    write_jsonl(SETS / "T1.jsonl", make_corrupted_set(test, args.seed + 1, limit=500))
    write_jsonl(SETS / "V3.jsonl", make_clean_set(val, args.seed, n_passages=100))
    write_jsonl(SETS / "T3.jsonl", make_clean_set(test, args.seed + 1, n_passages=200))
    write_jsonl(
        SETS / "T0.jsonl",
        [EvalItem("readme-example", "challenge", EXAMPLE_INPUT, EXAMPLE_OUTPUT, 1.0, {})],
    )
    (SETS / "meta.json").write_text(
        json.dumps(
            {"seed": args.seed, "corruptor": asdict(CorruptConfig()), "git_commit": git_commit()},
            indent=2,
        )
        + "\n"
    )
    print("sets written:", sorted(p.name for p in SETS.iterdir()))


def cmd_lexicon(args: argparse.Namespace) -> None:
    train = read_jsonl(CLEAN / "train.jsonl", CleanDoc)
    lx = build_lexicon(train)
    lx.write(LEXICON)
    print(f"lexicon: {len(lx)} words -> {LEXICON}")


def cmd_publish(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    files = [
        *CLEAN.glob("*.jsonl"),
        *SETS.glob("*.jsonl"),
        *RAW.glob("*.meta.json"),
        DATA / "split.json",
    ]
    write_manifest(
        DATA / "manifest.json",
        {
            "dataset_version": DATASET_VERSION,
            "git_commit": git_commit(),
            "built": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "files": {str(p.relative_to(DATA)): file_sha256(p) for p in files},
        },
    )
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(DATA),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=[
            "clean/*.jsonl",
            "sets/*.jsonl",
            "raw/*.meta.json",
            "raw/generated/*.txt",
            "split.json",
            "manifest.json",
            "README.md",
        ],
        commit_message=f"dataset v{DATASET_VERSION} from {git_commit()[:12]}",
    )
    print(f"published to https://huggingface.co/datasets/{args.repo}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wikipedia")
    w.add_argument("--n", type=int, default=20000)
    w.add_argument("--seed", type=int, default=1)
    w.add_argument("--revision")
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=2000)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--model", default="claude-haiku-4-5-20251001")
    a = sub.add_parser("assemble")
    a.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("sets")
    s.add_argument("--seed", type=int, default=1)
    sub.add_parser("lexicon")
    pub = sub.add_parser("publish")
    pub.add_argument("--repo", required=True)
    args = p.parse_args()
    {
        "wikipedia": cmd_wikipedia,
        "generate": cmd_generate,
        "assemble": cmd_assemble,
        "sets": cmd_sets,
        "lexicon": cmd_lexicon,
        "publish": cmd_publish,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
