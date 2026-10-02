import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from newline_fixer.data.build import assemble, build_lexicon, make_clean_set, make_corrupted_set
from newline_fixer.data.manifest import build_manifest, file_sha256, published_files, write_manifest
from newline_fixer.data.records import CleanDoc
from newline_fixer.text import content, derive_labels


def doc(i: int, text: str, group: str | None = None) -> CleanDoc:
    return CleanDoc.make(f"d{i}", "test", str(i), group or f"g{i}", text)


def body(i: int) -> str:
    return "\n\n".join(f"Heading {i}-{k}\n\nParagraph {k}. " + f"words {i} " * 30 for k in range(4))


def test_assemble_filters_dedupes_and_splits_by_group() -> None:
    docs = [doc(i, body(i), group=f"g{i % 50}") for i in range(200)]
    docs.append(doc(999, "short"))
    docs.append(doc(998, body(1)))
    parts = assemble(docs, seed=3)
    assert sum(len(v) for v in parts.values()) == 200
    group_side: dict[str, str] = {}
    for side, ds in parts.items():
        for d in ds:
            assert group_side.setdefault(d.group, side) == side


def test_corrupted_set_items_are_consistent() -> None:
    docs = [doc(i, body(i)) for i in range(20)]
    items = make_corrupted_set(docs, seed=5, limit=10)
    assert len(items) == 10
    for it in items:
        assert content(it.input) == content(it.target)
        derive_labels(it.input, it.target)
        assert 0.0 <= it.severity <= 1.0
    assert items == make_corrupted_set(docs, seed=5, limit=10)


def test_clean_set_items_are_identity_pairs() -> None:
    docs = [doc(i, body(i)) for i in range(20)]
    items = make_clean_set(docs, seed=5, n_passages=15)
    assert len(items) == 15
    for it in items:
        assert it.input == it.target and it.severity == 0.0
        assert 300 <= len(it.input) <= 800


def test_build_lexicon_counts_alpha_tokens() -> None:
    lx = build_lexicon([doc(1, "alpha alpha alpha beta beta 42 42 42")], min_count=3)
    assert lx.known("alpha") and not lx.known("beta") and not lx.known("42")


def test_manifest(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert file_sha256(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    write_manifest(tmp_path / "manifest.json", {"files": {"a.txt": file_sha256(f)}})
    assert (tmp_path / "manifest.json").exists()


def test_published_files_matches_patterns_and_excludes_manifest(tmp_path: Path) -> None:
    for rel in (
        "clean/train.jsonl",
        "sets/V1.jsonl",
        "sets/meta.json",
        "raw/wikipedia.meta.json",
        "raw/generated/00000.txt",
        "split.json",
        "README.md",
        "manifest.json",
        "raw/wikipedia.jsonl",
        "raw/pdf/x.pdf",
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    got = [p.relative_to(tmp_path).as_posix() for p in published_files(tmp_path)]
    assert got == sorted(
        [
            "README.md",
            "clean/train.jsonl",
            "raw/generated/00000.txt",
            "raw/wikipedia.meta.json",
            "sets/V1.jsonl",
            "sets/meta.json",
            "split.json",
        ]
    )


def test_build_manifest_hashes_every_published_file(tmp_path: Path) -> None:
    (tmp_path / "sets").mkdir()
    (tmp_path / "sets" / "meta.json").write_text("{}")
    (tmp_path / "split.json").write_text("{}")
    m = build_manifest(tmp_path, "1", "abc", "2026-10-02T00:00:00+00:00")
    assert m["dataset_version"] == "1" and m["git_commit"] == "abc"
    files = m["files"]
    assert isinstance(files, dict)
    assert set(files) == {"sets/meta.json", "split.json"}
    assert files["split.json"] == file_sha256(tmp_path / "split.json")


def test_publish_prints_and_records_hub_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_data_script", Path(__file__).parent.parent / "scripts" / "build_data.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    (tmp_path / "split.json").write_text("{}", encoding="utf-8")
    calls: dict[str, Any] = {}

    class FakeApi:
        def create_repo(self, *args: Any, **kwargs: Any) -> None:
            calls["create_repo"] = (args, kwargs)

        def upload_folder(self, **kwargs: Any) -> Any:
            calls["upload_folder"] = kwargs
            return SimpleNamespace(oid="deadbeef")

    assert script.publish("u/repo", tmp_path, FakeApi()) == "deadbeef"
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "published https://huggingface.co/datasets/u/repo revision deadbeef",
        "pin with revision deadbeef",
    ]
    assert calls["create_repo"][0] == ("u/repo",)
    assert calls["upload_folder"]["repo_id"] == "u/repo"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hub"] == {"repo": "u/repo", "revision": "deadbeef"}
