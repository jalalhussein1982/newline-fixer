"""Dataset manifest with content hashes (design section 3.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, entries: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


PUBLISH_PATTERNS: tuple[str, ...] = (
    "clean/*.jsonl",
    "sets/*.jsonl",
    "sets/meta.json",
    "raw/*.meta.json",
    "raw/generated/*.txt",
    "split.json",
    "README.md",
)


def published_files(data_dir: Path) -> list[Path]:
    """Every file under data_dir that the publish step uploads, sorted, manifest excluded."""
    out: set[Path] = set()
    for pattern in PUBLISH_PATTERNS:
        out.update(p for p in data_dir.glob(pattern) if p.is_file())
    return sorted(out)


def build_manifest(
    data_dir: Path, dataset_version: str, git_commit: str, built: str
) -> dict[str, object]:
    return {
        "dataset_version": dataset_version,
        "git_commit": git_commit,
        "built": built,
        "files": {
            p.relative_to(data_dir).as_posix(): file_sha256(p) for p in published_files(data_dir)
        },
    }
