"""One JSON line per request (design 6.3). Request text is never logged."""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys

LOGGER = logging.getLogger("newline_fixer.service")


class _JsonStdout(logging.StreamHandler):  # type: ignore[type-arg]
    """Marker subclass so configure_logging is idempotent."""


def configure_logging(level: str) -> None:
    LOGGER.setLevel(level)
    if not any(isinstance(h, _JsonStdout) for h in LOGGER.handlers):
        handler = _JsonStdout(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)


def log_request(
    *,
    request_id: str,
    endpoint: str,
    status: int,
    input_chars: int | None,
    changed: int | None,
    latency_ms: float,
    model: str,
) -> None:
    LOGGER.info(
        json.dumps(
            {
                "ts": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
                "request_id": request_id,
                "endpoint": endpoint,
                "status": status,
                "input_chars": input_chars,
                "changed": changed,
                "latency_ms": round(latency_ms, 2),
                "model": model,
            }
        )
    )
