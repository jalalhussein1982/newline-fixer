"""Service-level numbers of design 5.2: size, memory, latency percentiles, throughput."""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..data.records import EvalItem
from ..models.base import Fixer
from ..windows import fix

LENGTHS: tuple[int, ...] = (500, 2000, 10000)


def bench_inputs(items: Sequence[EvalItem], lengths: Sequence[int] = LENGTHS) -> dict[int, str]:
    """Prefixes of the concatenated passages, one per requested character length."""
    text = "\n\n".join(item.input for item in items)
    need = max(lengths)
    if len(text) < need:
        raise ValueError(f"benchmark text has {len(text)} characters, need {need}")
    return {n: text[:n] for n in lengths}


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; p in [0, 100]."""
    ordered = sorted(values)
    rank = max(1, int(round(p / 100 * len(ordered) + 0.5)))
    return ordered[min(rank, len(ordered)) - 1]


def time_calls(call: Callable[[], object], n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        call()
    out: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        call()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def throughput(
    call: Callable[[str], object], text: str, workers: int = 8, rounds: int = 5
) -> float:
    """Characters per second with `workers` concurrent calls on the same text."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(call, [text] * workers))  # warm-up round
        t0 = time.perf_counter()
        for _ in range(rounds):
            list(pool.map(call, [text] * workers))
        seconds = time.perf_counter() - t0
    return len(text) * workers * rounds / seconds


def rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def disk_mb(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def _latency_block(
    call_for: Callable[[str], Callable[[], object]], inputs: dict[int, str], n: int, warmup: int
) -> dict[str, dict[str, float]]:
    block: dict[str, dict[str, float]] = {}
    for length, text in inputs.items():
        times = time_calls(call_for(text), n, warmup)
        block[str(length)] = {
            "p50": round(percentile(times, 50), 2),
            "p95": round(percentile(times, 95), 2),
        }
    return block


def bench_fixer(
    fixer: Fixer, inputs: dict[int, str], n: int = 20, warmup: int = 3
) -> dict[str, object]:
    """In-process numbers for one fixer (model size from `weights_dir` when it has one)."""
    weights_dir = getattr(fixer, "weights_dir", None)
    disk = (
        disk_mb(Path(weights_dir))
        if weights_dir
        else disk_mb(Path(__file__).resolve().parents[1] / "resources")
    )
    latency = _latency_block(lambda text: lambda: fix(text, fixer), inputs, n, warmup)
    tput = throughput(lambda text: fix(text, fixer), inputs[2000])
    return {
        "disk_mb": round(disk, 2),
        "rss_mb": round(rss_mb(), 1),
        "latency_ms": latency,
        "throughput_chars_per_s": round(tput),
        "n": n,
    }


def bench_http(
    url: str, inputs: dict[int, str], n: int = 20, warmup: int = 3
) -> tuple[str, dict[str, object]]:
    """The same numbers through a running server; returns (served model name, record)."""
    import httpx

    with httpx.Client(base_url=url, timeout=120) as client:
        health = client.get("/healthz")
        health.raise_for_status()
        model = str(health.json()["model"])

        def post(text: str) -> object:
            r = client.post("/v1/fix", json={"text": text})
            r.raise_for_status()
            return r

        latency = _latency_block(lambda text: lambda: post(text), inputs, n, warmup)
        tput = throughput(post, inputs[2000])
    return model, {
        "disk_mb": 0.0,
        "rss_mb": 0.0,
        "latency_ms": latency,
        "throughput_chars_per_s": round(tput),
        "n": n,
    }
