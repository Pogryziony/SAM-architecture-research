"""Latency benchmark for :func:`nexus.realizer.comparison_plan.realize_comparison_plan`.

Uses a mock label selector to isolate symbolic path latency — no PyTorch
dependency, safe for non-torch CI.  Measures p50/p95/p99/mean/min/max over
N iterations with :func:`time.perf_counter`.

Usage::

    python benchmarks/benchmark_comparison_runtime.py
    python benchmarks/benchmark_comparison_runtime.py --iterations 200

Exit codes: 0 when p95 < 500 ms, 2 otherwise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.realizer.comparison_plan import realize_comparison_plan


_SCHEMA_VERSION = "nexus-comparison-latency-benchmark-v1"
_DEFAULT_ITERATIONS = 100
_P95_THRESHOLD_MS = 500.0


def _mock_label_selector(source: str, candidates: tuple[str, str]) -> tuple[str, dict[str, Any]]:
    """Select the label that appears in the serialized plan text.

    Avoids any real model inference so the benchmark isolates the symbolic
    plan build + serialization + candidate scoring codepath.
    """
    assert candidates == ("SAME", "DIFFERENT")
    selected = "SAME" if "[VERIFIED_RELATION] SAME\n" in source else "DIFFERENT"
    return selected, {"strategy": "benchmark_mock"}


def _evidence_pack(value_1: str = "0.5", value_2: str = "0.7") -> dict[str, Any]:
    return {
        "node_facts": [
            {
                "text": f"In configs/a.yaml, rate is set to {value_1}.",
                "source": "configs/a.yaml",
            },
            {
                "text": f"In configs/b.yaml, rate is set to {value_2}.",
                "source": "configs/b.yaml",
            },
        ]
    }


def _question() -> str:
    return (
        "Compare rate in configs/a.yaml and configs/b.yaml. What value does "
        "each source report, and are the values the same or different?"
    )


def _benchmark(iterations: int) -> dict[str, Any]:
    """Run the latency benchmark and return a structured result dict."""
    evidence = _evidence_pack()
    question = _question()

    # Warm-up: run 3 iterations to stabilise caches / JIT / allocators.
    for _ in range(3):
        realize_comparison_plan(
            question, evidence, label_selector=_mock_label_selector,
        )

    latencies_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        realize_comparison_plan(
            question, evidence, label_selector=_mock_label_selector,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed)

    return latencies_ms


def _percentile(values: list[float], pct: float) -> float:
    """Compute the pct-th percentile of *values* using the median-of-nearest method."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (pct / 100.0) * (len(sorted_vals) - 1)
    lower = int(k)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = k - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _report(latencies_ms: list[float], iterations: int) -> dict[str, Any]:
    result = {
        "schema_version": _SCHEMA_VERSION,
        "iterations": iterations,
        "latency_ms": {
            "p50": round(_percentile(latencies_ms, 50), 4),
            "p95": round(_percentile(latencies_ms, 95), 4),
            "p99": round(_percentile(latencies_ms, 99), 4),
            "mean": round(statistics.mean(latencies_ms), 4),
            "min": round(min(latencies_ms), 4),
            "max": round(max(latencies_ms), 4),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark comparison-plan Realizer latency.")
    parser.add_argument(
        "--iterations", type=int, default=_DEFAULT_ITERATIONS,
        help=f"Number of measurement iterations (default: {_DEFAULT_ITERATIONS})",
    )
    args = parser.parse_args()

    latencies = _benchmark(args.iterations)
    result = _report(latencies, args.iterations)

    print(json.dumps(result, indent=2))
    p95 = result["latency_ms"]["p95"]
    if p95 >= _P95_THRESHOLD_MS:
        print(
            f"\n[WARN] p95 latency {p95:.2f} ms exceeds {_P95_THRESHOLD_MS} ms threshold.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
