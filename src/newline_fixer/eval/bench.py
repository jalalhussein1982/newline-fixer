"""Service-level numbers of design 5.2: size, memory, latency percentiles, throughput."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..data.records import EvalItem
from ..models.base import Fixer
from ..text import Gap, split
from ..windows import fix

LENGTHS: tuple[int, ...] = (500, 2000, 10000)


def bench_inputs(items: Sequence[EvalItem], lengths: Sequence[int] = LENGTHS) -> dict[int, str]:
    """Prefixes of the concatenated passages, one per requested character length."""
    text = "\n\n".join(item.input for item in items)
    need = max(lengths)
    if len(text) < need:
        raise ValueError(f"benchmark text has {len(text)} characters, need {need}")
    return {n: text[:n] for n in lengths}


def select_window(items: Sequence[EvalItem], n_tokens: int = 256) -> tuple[list[str], list[Gap]]:
    """The first `n_tokens` tokens of the concatenated passages with their current gaps."""
    parts: list[str] = []
    count = 0
    for item in items:
        parts.append(item.input)
        count += len(item.input.split())
        if count >= n_tokens:
            break
    if count < n_tokens:
        raise ValueError(f"benchmark text has {count} tokens, need {n_tokens}")
    tokens, gaps = split("\n\n".join(parts))
    return tokens[:n_tokens], gaps[: n_tokens - 1]


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; p in [0, 100]."""
    ordered = sorted(values)
    rank = min(max(math.ceil(p / 100 * len(ordered)), 1), len(ordered))
    return ordered[rank - 1]


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
    """Current resident set size of this process in MB."""
    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip()) * 1024 / 1e6


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


def _disk_for(fixer: Fixer) -> float:
    weights_dir = getattr(fixer, "weights_dir", None)
    if weights_dir:
        return disk_mb(Path(weights_dir))
    if fixer.name == "rules":
        return disk_mb(Path(__file__).resolve().parents[1] / "resources")
    return 0.0


def bench_fixer(
    fixer: Fixer, inputs: dict[int, str], n: int = 20, warmup: int = 3
) -> dict[str, object]:
    """In-process numbers for one fixer; RSS is read after the latency warm-up and timed calls."""
    latency = _latency_block(lambda text: lambda: fix(text, fixer), inputs, n, warmup)
    resident = rss_mb()
    tput = throughput(lambda text: fix(text, fixer), inputs[2000])
    return {
        "disk_mb": round(_disk_for(fixer), 2),
        "rss_mb": round(resident, 1),
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
        "disk_mb": None,  # the client cannot see the server process
        "rss_mb": None,
        "latency_ms": latency,
        "throughput_chars_per_s": round(tput),
        "n": n,
    }
