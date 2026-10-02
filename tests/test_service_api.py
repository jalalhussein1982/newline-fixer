from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import settings as hsettings
from hypothesis import strategies as st

from newline_fixer.models.base import Fixer
from newline_fixer.models.registry import get_fixer
from newline_fixer.service.app import create_app
from newline_fixer.service.config import Settings
from newline_fixer.text import content, normalize

RULES = get_fixer("rules")


def wait_ready(c: TestClient, seconds: float = 5.0) -> None:
    deadline = time.time() + seconds
    while c.get("/healthz").status_code != 200:
        assert time.time() < deadline, "model never became ready"
        time.sleep(0.01)


def make_client(settings: Settings | None = None, fixer: Fixer = RULES) -> TestClient:
    return TestClient(create_app(settings or Settings(model="rules"), loader=lambda _s: fixer))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client() as c:
        wait_ready(c)
        yield c


def test_example_round_trips(client: TestClient, example_input: str, example_output: str) -> None:
    r = client.post("/v1/fix", json={"text": example_input})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == normalize(example_output)
    stats = body["stats"]
    assert stats["model"] == "rules" and stats["tokens"] == 30 and stats["gaps"] == 29
    assert stats["changed"] > 0 and stats["latency_ms"] >= 0


def test_empty_and_whitespace_only_text(client: TestClient) -> None:
    for text in ("", "  \n\n\t "):
        r = client.post("/v1/fix", json={"text": text})
        assert r.status_code == 200
        assert r.json() == {
            "text": "",
            "stats": {
                "tokens": 0,
                "gaps": 0,
                "changed": 0,
                "model": "rules",
                "latency_ms": r.json()["stats"]["latency_ms"],
            },
        }


def test_clean_input_is_unchanged(client: TestClient, example_output: str) -> None:
    clean = normalize(example_output)
    r = client.post("/v1/fix", json={"text": clean})
    assert r.status_code == 200
    assert r.json()["text"] == clean and r.json()["stats"]["changed"] == 0


def test_long_input_needs_windows_and_keeps_content(client: TestClient) -> None:
    words = ["The", "model", "reads", "long", "inputs", "in", "windows."] * 500  # 3,500 tokens
    text = " ".join(words)
    r = client.post("/v1/fix", json={"text": text})
    assert r.status_code == 200
    assert content(r.json()["text"]) == content(text)
    assert r.json()["stats"]["tokens"] == 3500 and r.json()["stats"]["gaps"] == 3499


def test_one_enormous_token_is_returned_unchanged(client: TestClient) -> None:
    token = "x" * 50_000
    r = client.post("/v1/fix", json={"text": token})
    assert r.status_code == 200
    assert r.json()["text"] == token and r.json()["stats"]["gaps"] == 0


def test_size_limit_gives_413() -> None:
    with make_client(Settings(model="rules", max_chars=100)) as c:
        wait_ready(c)
        assert c.post("/v1/fix", json={"text": "a" * 100}).status_code == 200
        r = c.post("/v1/fix", json={"text": "a" * 101})
        assert r.status_code == 413
        assert "100" in r.json()["detail"]


def test_bad_bodies_give_422(client: TestClient) -> None:
    assert client.post("/v1/fix", json={}).status_code == 422
    assert client.post("/v1/fix", json={"text": 5}).status_code == 422
    assert client.post("/v1/fix", json={"text": None}).status_code == 422
    bad = client.post("/v1/fix", content=b"not json", headers={"content-type": "application/json"})
    assert bad.status_code == 422
    # the client cannot encode a lone surrogate itself, so send the JSON escape as raw bytes
    lone = client.post(
        "/v1/fix",
        content=b'{"text": "a \\ud800 b"}',
        headers={"content-type": "application/json"},
    )
    assert lone.status_code == 422


def test_health_is_503_until_the_model_is_loaded() -> None:
    release = threading.Event()

    def slow_loader(_s: Settings) -> Fixer:
        release.wait(5)
        return RULES

    with TestClient(create_app(Settings(model="rules"), loader=slow_loader)) as c:
        h = c.get("/healthz")
        assert h.status_code == 503
        assert h.json() == {"status": "loading", "model": "rules", "ready": False}
        assert c.post("/v1/fix", json={"text": "a b"}).status_code == 503
        release.set()
        wait_ready(c)
        assert c.get("/healthz").json() == {"status": "ok", "model": "rules", "ready": True}


def test_failed_load_is_reported_not_hidden() -> None:
    def broken(_s: Settings) -> Fixer:
        raise FileNotFoundError("no weights at hf:nobody/nothing@dead")

    with TestClient(create_app(Settings(model="rules"), loader=broken)) as c:
        deadline = time.time() + 5
        while c.get("/healthz").json().get("status") == "loading" and time.time() < deadline:
            time.sleep(0.01)
        h = c.get("/healthz")
        assert h.status_code == 503
        assert h.json()["status"] == "error" and h.json()["ready"] is False
        assert "no weights" in h.json()["error"]
        r = c.post("/v1/fix", json={"text": "a b"})
        assert r.status_code == 503 and "no weights" in r.json()["detail"]


@pytest.mark.skipif(
    not Path("experiments/runs/current/model.pt").exists(), reason="no trained weights"
)
def test_scratch_model_serves_the_example(example_input: str) -> None:
    app = create_app(Settings(model="scratch", weights="experiments/runs/current"))
    with TestClient(app) as c:
        wait_ready(c, 30)
        r = c.post("/v1/fix", json={"text": example_input})
        assert r.status_code == 200 and r.json()["stats"]["model"] == "scratch"
        assert content(r.json()["text"]) == content(example_input)


@given(st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=400))
@hsettings(max_examples=60, deadline=None)
def test_api_preserves_content_for_any_text(text: str) -> None:
    with make_client() as c:
        wait_ready(c)
        r = c.post("/v1/fix", json={"text": text})
        assert r.status_code == 200
        assert content(r.json()["text"]) == content(text)
