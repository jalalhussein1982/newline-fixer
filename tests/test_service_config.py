import pytest
from pydantic import ValidationError

from newline_fixer.service.config import (
    DEFAULT_MODEL,
    HUB_REPO,
    PUBLISHED_REVISION,
    Settings,
    load_fixer,
)
from newline_fixer.service.schemas import FixRequest, FixResponse, FixStats, Health


def test_settings_defaults_follow_the_design() -> None:
    s = Settings()
    assert (s.model, s.max_chars, s.log_level, s.device) == (DEFAULT_MODEL, 100_000, "INFO", "cpu")
    assert s.model_revision == PUBLISHED_REVISION
    assert s.weights is None


def test_settings_from_env_reads_nf_variables() -> None:
    s = Settings.from_env(
        {
            "NF_MODEL": "scratch",
            "NF_MODEL_REVISION": "abc123",
            "NF_MAX_CHARS": "500",
            "NF_LOG_LEVEL": "debug",
            "NF_DEVICE": "cpu",
        }
    )
    assert (s.model, s.model_revision, s.max_chars, s.log_level) == (
        "scratch",
        "abc123",
        500,
        "DEBUG",
    )
    assert s.weights_source() == f"hf:{HUB_REPO}@abc123"


def test_weights_env_overrides_the_revision() -> None:
    s = Settings.from_env({"NF_MODEL": "scratch", "NF_WEIGHTS": "experiments/runs/current"})
    assert s.weights_source() == "experiments/runs/current"
    assert Settings.from_env({"NF_MODEL": "rules"}).weights_source() is None


def test_settings_reject_unknown_model_and_bad_limit() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        Settings(model="gpt")
    with pytest.raises(ValueError, match="NF_MAX_CHARS"):
        Settings(max_chars=0)
    with pytest.raises(ValueError, match="NF_MAX_CHARS"):
        Settings.from_env({"NF_MAX_CHARS": "lots"})


def test_load_fixer_rules_and_identity() -> None:
    assert load_fixer(Settings(model="rules")).name == "rules"
    assert load_fixer(Settings(model="identity")).name == "identity"


def test_fix_request_accepts_text_and_rejects_the_rest() -> None:
    assert FixRequest(text="a b").text == "a b"
    assert FixRequest(text="").text == ""
    with pytest.raises(ValidationError):
        FixRequest.model_validate({})
    with pytest.raises(ValidationError):
        FixRequest.model_validate({"text": 123})
    with pytest.raises(ValidationError):
        FixRequest.model_validate({"text": None})


def test_fix_request_rejects_lone_surrogates() -> None:
    with pytest.raises(ValidationError, match="Unicode"):
        FixRequest(text="ok \ud800 not ok")


def test_response_models_serialize_the_contract() -> None:
    r = FixResponse(
        text="a\nb", stats=FixStats(tokens=2, gaps=1, changed=1, model="rules", latency_ms=0.5)
    )
    assert r.model_dump() == {
        "text": "a\nb",
        "stats": {"tokens": 2, "gaps": 1, "changed": 1, "model": "rules", "latency_ms": 0.5},
    }
    assert Health(status="ok", model="rules", ready=True).model_dump() == {
        "status": "ok",
        "model": "rules",
        "ready": True,
    }
