from pathlib import Path
from types import SimpleNamespace

from newline_fixer.data.generated import build_brief, generate_docs, validate_generated

GOOD = (
    "2.1 Scope\n\n"
    "This manual covers the installation of the pump in a domestic garden setting. Read it "
    "fully before starting, and keep it near the pump for later reference.\n\n"
    "Tools required:\n"
    "- a torque wrench\n"
    "- two M8 bolts\n"
    "- thread sealant\n\n"
    "Keep the work area dry. The pump must not run without water for more than ten seconds, "
    "because the seals depend on water for cooling."
)


def test_brief_mentions_register_topic_marker() -> None:
    b = build_brief("user manual", "a garden pump", "-")
    assert "user manual" in b and "garden pump" in b and '"-"' in b


def test_validate_accepts_structured_plain_text() -> None:
    assert validate_generated(GOOD, "-") == GOOD


def test_validate_rejects_markdown_or_missing_structure() -> None:
    assert validate_generated("# Title\n\nbody\n\n- item", "-") is None
    assert validate_generated("just one paragraph of text", "-") is None


def test_validate_accepts_any_known_marker() -> None:
    # Models sometimes substitute a marker; any marker from MARKERS is acceptable.
    assert validate_generated(GOOD.replace("- ", "* "), "-") is not None


class FakeMessages:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(text=GOOD)])


def test_generate_docs_caches_and_validates(tmp_path: Path) -> None:
    fake = SimpleNamespace(messages=FakeMessages())
    docs = generate_docs(3, seed=1, out_dir=tmp_path, model="test", client=fake)
    assert len(docs) == 3
    assert fake.messages.calls == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["00000.txt", "00001.txt", "00002.txt"]
    again = generate_docs(3, seed=1, out_dir=tmp_path, model="test", client=fake)
    assert fake.messages.calls == 3
    assert again == docs
    assert docs[0].source == "generated" and docs[0].group == "gen-00000"
