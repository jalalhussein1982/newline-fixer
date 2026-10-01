"""Wikipedia articles from the `wikimedia/wikipedia` dump (design section 3.1)."""

from __future__ import annotations

from collections.abc import Iterator

from ..text import normalize
from .filters import has_structure, is_hard_wrapped, long_enough
from .records import CleanDoc

TRAILING_SECTIONS = {
    "see also",
    "references",
    "external links",
    "notes",
    "further reading",
    "bibliography",
    "sources",
    "footnotes",
    "gallery",
}


def clean_wikipedia_text(text: str, max_chars: int = 4000) -> str | None:
    """Drop trailing reference sections, truncate on a paragraph boundary, filter."""
    paras = [p.strip() for p in normalize(text).split("\n\n") if p.strip()]
    kept: list[str] = []
    size = 0
    for p in paras:
        if p.lower() in TRAILING_SECTIONS:
            break
        extra = len(p) + (2 if kept else 0)
        if size + extra > max_chars:
            break
        kept.append(p)
        size += extra
    out = "\n\n".join(kept)
    if not (long_enough(out) and has_structure(out)) or is_hard_wrapped(out):
        return None
    return out


def resolve_revision(repo: str = "wikimedia/wikipedia") -> str:
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo).sha
    if not sha:
        raise RuntimeError(f"could not resolve revision for {repo}")
    return sha


def iter_wikipedia(
    n_docs: int, seed: int, revision: str, config: str = "20231101.en"
) -> Iterator[CleanDoc]:
    if n_docs <= 0:
        return
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True, revision=revision
    ).shuffle(seed=seed, buffer_size=10_000)
    produced = 0
    for row in ds:
        cleaned = clean_wikipedia_text(row["text"])
        if cleaned is None:
            continue
        article_id = str(row["id"])
        yield CleanDoc.make(
            f"wiki-{article_id}", "wikipedia", article_id, f"wiki-{article_id}", cleaned
        )
        produced += 1
        if produced >= n_docs:
            return
