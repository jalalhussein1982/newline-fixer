from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from newline_fixer.models.registry import get_fixer
from newline_fixer.service.app import create_app
from newline_fixer.service.config import Settings
from newline_fixer.service.metrics import Metrics
from newline_fixer.text import Gap

RULES = get_fixer("rules")


class BoomFixer:
    name = "boom"
    budget = 256

    def token_cost(self, token: str) -> int:
        return 1

    def gap_cost(self, gap: Gap) -> int:
        return 0

    def overhead(self) -> int:
        return 0

    def predict(self, tokens: Sequence[str], current: Sequence[Gap]) -> list[Gap]:
        raise RuntimeError("boom")


@contextmanager
def ready_client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings(model="rules"), loader=lambda _s: RULES)) as c:
        while c.get("/healthz").status_code != 200:
            time.sleep(0.01)
        yield c


def test_metrics_object_counts_and_renders() -> None:
    m = Metrics()
    m.set_model("rules", "")
    m.observe("/v1/fix", 200, 0.012, 40, 3)
    m.observe("/v1/fix", 413, 0.001, None, None)
    m.observe("/nope", 404, 0.001, None, None)
    text = m.render().decode()
    assert 'nf_requests_total{endpoint="/v1/fix",status="200"} 1.0' in text
    assert 'nf_requests_total{endpoint="/v1/fix",status="413"} 1.0' in text
    assert 'nf_model_info{model="rules",weights=""} 1.0' in text
    assert "nf_request_latency_seconds_bucket" in text
    assert "nf_input_chars_count 1.0" in text and "nf_gaps_changed_sum 3.0" in text
    assert "nf_errors_total" in text
    m.observe("/healthz", 503, 0.001, None, None)
    assert 'nf_errors_total{endpoint="/healthz"} 1.0' not in m.render().decode()
    m.observe("/v1/fix", 503, 0.001, None, None)
    assert 'nf_errors_total{endpoint="/v1/fix"} 1.0' in m.render().decode()


def test_metrics_endpoint_reflects_requests() -> None:
    with ready_client() as c:
        c.post("/v1/fix", json={"text": "a\n b"})
        c.post("/v1/fix", json={"text": "a" * 200_000})
        r = c.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        text = r.text
        assert 'nf_requests_total{endpoint="/v1/fix",status="200"} 1.0' in text
        assert 'nf_requests_total{endpoint="/v1/fix",status="413"} 1.0' in text
        assert 'nf_requests_total{endpoint="/healthz",status="200"}' in text
        assert 'nf_model_info{model="rules"' in text


def test_unknown_paths_do_not_create_labels() -> None:
    with ready_client() as c:
        for i in range(3):
            assert c.get(f"/wp-admin/{i}").status_code == 404
        text = c.get("/metrics").text
        assert "wp-admin" not in text
        assert 'nf_requests_total{endpoint="unmatched",status="404"} 3.0' in text


def test_request_log_line_has_the_fields_and_no_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="newline_fixer.service")
    secret = "zebra quokka wombat"
    with ready_client() as c:
        r = c.post("/v1/fix", json={"text": secret}, headers={"x-request-id": "req-42"})
        assert r.headers["x-request-id"] == "req-42"
    lines = [
        json.loads(rec.getMessage())
        for rec in caplog.records
        if rec.name == "newline_fixer.service"
    ]
    fix_lines = [line for line in lines if line["endpoint"] == "/v1/fix"]
    assert len(fix_lines) == 1
    line = fix_lines[0]
    assert set(line) == {
        "ts",
        "request_id",
        "endpoint",
        "status",
        "input_chars",
        "changed",
        "latency_ms",
        "model",
    }
    assert line["request_id"] == "req-42" and line["status"] == 200 and line["model"] == "rules"
    assert line["input_chars"] == len(secret) and line["changed"] == 0
    assert re.match(r"\d{4}-\d{2}-\d{2}T", line["ts"])
    assert all("zebra" not in rec.getMessage() for rec in caplog.records)


def test_generated_request_id_is_returned(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="newline_fixer.service")
    with ready_client() as c:
        r = c.get("/healthz")
        rid = r.headers["x-request-id"]
        assert re.fullmatch(r"[0-9a-f]{16}", rid)
    assert any(json.loads(rec.getMessage())["request_id"] == rid for rec in caplog.records)


def test_unhandled_error_is_a_json_500_with_request_id(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="newline_fixer.service")
    app = create_app(Settings(model="rules"), loader=lambda _s: BoomFixer())
    with TestClient(app, raise_server_exceptions=False) as c:
        while c.get("/healthz").status_code != 200:
            time.sleep(0.01)
        r = c.post("/v1/fix", json={"text": "a b"})
        assert r.status_code == 500
        body = r.json()
        assert body["detail"] == "internal error"
        assert r.headers["x-request-id"] == body["request_id"]
        assert 'nf_errors_total{endpoint="/v1/fix"} 1.0' in c.get("/metrics").text
    lines = [
        json.loads(rec.getMessage())
        for rec in caplog.records
        if rec.name == "newline_fixer.service"
    ]
    assert any(line["endpoint"] == "/v1/fix" and line["status"] == 500 for line in lines)
