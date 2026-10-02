"""LLM-generated structured documents (design section 3.1, decision 0005)."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from ..text import normalize
from .filters import has_structure, long_enough
from .records import CleanDoc

REGISTERS = [
    "a section of a research paper",
    "a user manual",
    "a technical report",
    "a business email",
    "meeting notes",
    "product documentation",
    "a policy memo",
    "a tutorial",
    "a news article",
    "an internal wiki page",
    "a grant proposal",
    "lecture notes",
    "a changelog with explanations",
    "a legal summary",
]
TOPICS = [
    "a garden pump",
    "attention mechanisms",
    "a city budget",
    "bread baking",
    "bicycle repair",
    "a database migration",
    "coral reefs",
    "a new hiring process",
    "solar panel installation",
    "a chess opening",
    "a kitchen renovation",
    "a vaccination campaign",
    "a railway timetable",
    "a mobile app release",
    "soil chemistry",
    "a library catalogue",
    "a marathon training plan",
    "wind turbine maintenance",
    "a school curriculum change",
    "a satellite launch",
]
MARKERS = ["•", "-", "*", "1."]

_MARKDOWN = re.compile(r"(^#+\s|\*\*|__|^```)", re.MULTILINE)


def build_brief(register: str, topic: str, marker: str) -> str:
    return (
        f"Write one self-contained English document of 250 to 600 words in the form of "
        f"{register}. Topic: {topic}.\n"
        "Rules: plain text only, no Markdown syntax (no #, no **, no backticks). Include at "
        'least one heading on its own line (either numbered like "2.1 Scope" or a short '
        f"title-case line), at least one list of three or more items on separate lines each "
        f'starting with "{marker}" (for "1." continue 2., 3.), and at least two ordinary '
        "paragraphs. Separate paragraphs and headings with one blank line. Put list items on "
        "consecutive lines with no blank lines between them. Never wrap lines inside a "
        "paragraph. Output only the document."
    )


def _list_lines(text: str) -> int:
    best = 0
    for marker in MARKERS:
        run = 0
        for line in text.split("\n"):
            starts = line.startswith(marker + " ") or (
                marker == "1." and re.match(r"^\d+\.\s", line) is not None
            )
            run = run + 1 if starts else 0
            best = max(best, run)
    return best


def validate_generated(text: str, marker: str) -> str | None:
    """Return the normalized document if it is usable; `marker` is the requested one."""
    del marker  # any marker from MARKERS is accepted, see the test
    norm = normalize(text)
    if _MARKDOWN.search(norm):
        return None
    if not (long_enough(norm) and has_structure(norm) and "\n\n" in norm):
        return None
    if _list_lines(norm) < 2:
        return None
    return norm


def generate_docs(
    n: int, seed: int, out_dir: Path, model: str, client: Any | None = None
) -> list[CleanDoc]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    choices: list[tuple[str, str, str]] = [
        (rng.choice(REGISTERS), rng.choice(TOPICS), rng.choice(MARKERS)) for _ in range(n)
    ]
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    docs: list[CleanDoc] = []
    for i, (register, topic, marker) in enumerate(choices):
        path = out_dir / f"{i:05d}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = _ask(client, model, build_brief(register, topic, marker), marker)
            path.write_text(text, encoding="utf-8")
        docs.append(
            CleanDoc.make(
                f"gen-{i:05d}", "generated", f"{model}|{register}|{topic}", f"gen-{i:05d}", text
            )
        )
    return docs


def _ask(client: Any, model: str, brief: str, marker: str, attempts: int = 3) -> str:
    for _ in range(attempts):
        msg = client.messages.create(
            model=model, max_tokens=1200, messages=[{"role": "user", "content": brief}]
        )
        text = validate_generated(msg.content[0].text, marker)
        if text is not None:
            return text
    raise RuntimeError("generation failed validation three times")
