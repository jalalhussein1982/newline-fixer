from pathlib import Path

import pytest

from newline_fixer.models.scratch import parse_weights_source, resolve_weights


def test_parse_weights_source() -> None:
    assert parse_weights_source("experiments/runs/current") == (
        "dir",
        "experiments/runs/current",
        None,
    )
    assert parse_weights_source("hf:user/repo") == ("hf", "user/repo", None)
    assert parse_weights_source("hf:user/repo@abc123") == ("hf", "user/repo", "abc123")
    with pytest.raises(ValueError):
        parse_weights_source("hf:")


def test_resolve_dir_source_is_path(tmp_path: Path) -> None:
    assert resolve_weights(str(tmp_path)) == tmp_path
    assert resolve_weights(tmp_path) == tmp_path


@pytest.mark.skipif(
    not Path("experiments/runs/current/model.pt").exists(), reason="no trained weights"
)
def test_current_weights_load_and_fix_example() -> None:
    from newline_fixer.example import EXAMPLE_INPUT
    from newline_fixer.models.scratch import ScratchFixer
    from newline_fixer.text import content
    from newline_fixer.windows import fix

    fx = ScratchFixer.load("experiments/runs/current")
    assert content(fix(EXAMPLE_INPUT, fx).text) == content(EXAMPLE_INPUT)


def test_resolve_hf_source_uses_snapshot_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    calls: list[tuple[str, str | None]] = []

    def fake(repo_id: str, revision: str | None = None) -> str:
        calls.append((repo_id, revision))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert resolve_weights("hf:user/repo@abc") == tmp_path
    assert calls == [("user/repo", "abc")]
