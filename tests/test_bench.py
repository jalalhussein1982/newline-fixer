from pathlib import Path

import pytest

from newline_fixer.data.records import EvalItem
from newline_fixer.eval.bench import (
    LENGTHS,
    bench_fixer,
    bench_inputs,
    disk_mb,
    percentile,
    rss_mb,
    throughput,
    time_calls,
)
from newline_fixer.eval.table import render_bench_table
from newline_fixer.models.identity import IdentityFixer


def items(n: int) -> list[EvalItem]:
    return [
        EvalItem(id=str(i), source="t", input=f"Passage {i} " * 60, target=f"Passage {i} " * 60)
        for i in range(n)
    ]


def test_bench_inputs_have_the_requested_lengths() -> None:
    inputs = bench_inputs(items(40))
    assert set(inputs) == set(LENGTHS)
    assert all(len(inputs[n]) == n for n in LENGTHS)


def test_bench_inputs_fail_loudly_when_text_is_short() -> None:
    with pytest.raises(ValueError, match="10000"):
        bench_inputs(items(2))


def test_percentile_is_nearest_rank() -> None:
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(xs, 50) == 3.0 and percentile(xs, 95) == 5.0 and percentile(xs, 0) == 1.0


def test_time_calls_and_throughput_are_positive() -> None:
    times = time_calls(lambda: sum(range(1000)), n=4, warmup=1)
    assert len(times) == 4 and all(t >= 0 for t in times)
    assert throughput(lambda s: len(s), "x" * 1000, workers=2, rounds=2) > 0


def test_memory_and_disk() -> None:
    assert rss_mb() > 1.0
    assert disk_mb(None) == 0.0
    assert disk_mb(Path("src/newline_fixer/resources")) > 0.0


def test_bench_fixer_record_shape() -> None:
    rec = bench_fixer(IdentityFixer(), bench_inputs(items(40)), n=2, warmup=1)
    assert set(rec) == {"disk_mb", "rss_mb", "latency_ms", "throughput_chars_per_s", "n"}
    latency = rec["latency_ms"]
    assert isinstance(latency, dict) and set(latency) == {"500", "2000", "10000"}
    assert rec["n"] == 2


def test_render_bench_table_has_one_row_per_system() -> None:
    record = {
        "label": "test",
        "run_at": "now",
        "git_commit": "abc",
        "platform": "p",
        "machine": "m",
        "mode": "in-process",
        "device": "cpu",
        "systems": {
            "identity": bench_fixer(IdentityFixer(), bench_inputs(items(40)), n=1, warmup=0)
        },
    }
    out = render_bench_table([record])
    assert out.count("| test | identity |") == 1 and "p50" in out
