"""Tests for the comparison-plan latency benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "benchmarks" / "benchmark_comparison_runtime.py"


@pytest.fixture
def benchmark_available() -> bool:
    return BENCHMARK_SCRIPT.is_file()


def test_benchmark_script_runs_without_errors(benchmark_available: bool) -> None:
    """Smoke test: the benchmark script completes without crashing."""
    if not benchmark_available:
        pytest.skip("Benchmark script not found")

    result = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--iterations", "20"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode in (0, 2), (
        f"Unexpected exit code {result.returncode}\n{result.stderr}"
    )


def test_benchmark_output_is_valid_json(benchmark_available: bool) -> None:
    """The benchmark script outputs valid JSON with the expected schema."""
    if not benchmark_available:
        pytest.skip("Benchmark script not found")

    result = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--iterations", "10"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode in (0, 2)

    # The JSON is printed to stdout; stderr may contain threshold warnings.
    output = result.stdout.strip()
    parsed = json.loads(output)
    assert parsed["schema_version"] == "nexus-comparison-latency-benchmark-v1"
    assert parsed["iterations"] == 10
    assert set(parsed["latency_ms"].keys()) == {"p50", "p95", "p99", "mean", "min", "max"}
    for value in parsed["latency_ms"].values():
        assert isinstance(value, (int, float))
        assert value >= 0


def test_mock_label_selector_latency_is_under_50ms(benchmark_available: bool) -> None:
    """Sanity check: the mock label selector path is sub-50 ms per call."""
    if not benchmark_available:
        pytest.skip("Benchmark script not found")

    # Import the mock selector from the benchmark module.
    sys.path.insert(0, str(ROOT))
    from benchmarks.benchmark_comparison_runtime import (
        _evidence_pack,
        _mock_label_selector,
        _question,
    )

    from nexus.realizer.comparison_plan import realize_comparison_plan

    import time

    evidence = _evidence_pack()
    question = _question()

    # Warm-up.
    for _ in range(3):
        realize_comparison_plan(question, evidence, label_selector=_mock_label_selector)

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        realize_comparison_plan(question, evidence, label_selector=_mock_label_selector)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    assert avg_ms < 50.0, f"Mock selector avg {avg_ms:.2f} ms exceeds 50 ms threshold"
