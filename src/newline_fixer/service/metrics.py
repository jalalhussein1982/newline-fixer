"""Prometheus metrics of design 6.1. One registry per app so tests never collide."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    Info,
    generate_latest,
)

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
CHARS_BUCKETS = (100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)
CHANGED_BUCKETS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000)
CONTENT_TYPE = CONTENT_TYPE_LATEST


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "nf_requests",
            "Requests by endpoint and status",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.errors = Counter(
            "nf_errors", "Responses with status 500 or above", ["endpoint"], registry=self.registry
        )
        self.latency = Histogram(
            "nf_request_latency_seconds",
            "Request latency",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.input_chars = Histogram(
            "nf_input_chars",
            "Input length of /v1/fix requests in characters",
            buckets=CHARS_BUCKETS,
            registry=self.registry,
        )
        self.gaps_changed = Histogram(
            "nf_gaps_changed",
            "Gaps changed per /v1/fix request",
            buckets=CHANGED_BUCKETS,
            registry=self.registry,
        )
        self.model = Info("nf_model", "Served model", registry=self.registry)
        self.errors.labels(endpoint="/v1/fix")  # the series exists before the first error

    def set_model(self, name: str, weights: str) -> None:
        self.model.info({"model": name, "weights": weights})

    def observe(
        self,
        endpoint: str,
        status: int,
        seconds: float,
        input_chars: int | None,
        changed: int | None,
    ) -> None:
        self.requests.labels(endpoint=endpoint, status=str(status)).inc()
        self.latency.labels(endpoint=endpoint).observe(seconds)
        if status >= 500 and not (endpoint == "/healthz" and status == 503):
            self.errors.labels(endpoint=endpoint).inc()
        if input_chars is not None:
            self.input_chars.observe(input_chars)
        if changed is not None:
            self.gaps_changed.observe(changed)

    def render(self) -> bytes:
        return generate_latest(self.registry)
