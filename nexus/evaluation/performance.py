"""End-to-end performance measurement for the exact grounded() profile."""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner


def _rss_mb() -> float | None:
    """Best-effort peak/current RSS in MB. Returns None if unavailable."""
    try:
        from nexus.evaluation.process_resources import process_tree_rss_mb

        tree = process_tree_rss_mb()
        if tree is not None:
            return tree
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import resource  # Unix-only

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return usage / (1024 * 1024)
        return usage / 1024.0
    except Exception:
        return None

def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


@dataclass
class PerformanceSample:
    question_id: str
    latency_ms: float
    peak_rss_mb: float | None
    cold: bool
    stages_ms: dict[str, float] = field(default_factory=dict)


def measure_grounded_e2e(
    runner: NEXUSRunner,
    questions: Sequence[dict[str, Any]],
    *,
    warmup: int = 1,
    repeats: int = 5,
    profile_name: str = "grounded",
    setup_timings: dict[str, float] | None = None,
    graph_meta: dict[str, Any] | None = None,
    scope: str = "unspecified",
) -> dict[str, Any]:
    """Measure cold/warm latency and peak RSS for the runner's config.

    Uses the real ``NEXUSRunner.run`` path. Does not fabricate samples.
    ``setup_timings`` (ms) and ``graph_meta`` are recorded separately from
    steady-state warm latency.
    """
    if not isinstance(runner.config, ProductionNEXUSConfig):
        raise TypeError("runner.config must be ProductionNEXUSConfig")

    samples: list[PerformanceSample] = []
    peak_rss: float | None = None
    outcome_counts = {
        "success": 0,
        "abstention": 0,
        "failure": 0,
        "timeout": 0,
    }
    by_type: dict[str, list[float]] = {}
    cpu_time_s = 0.0

    def _one(qid: str, question: str, cold: bool, qtype: str = "unknown") -> PerformanceSample:
        nonlocal peak_rss, cpu_time_s
        t0 = time.perf_counter()
        c0 = time.process_time()
        qr = runner._run_single(  # noqa: SLF001 — intentional e2e stage timing
            qid,
            question,
            runner.model or __import__(
                "nexus.reasoning.model_interface", fromlist=["DummyModel"]
            ).DummyModel(),
        )
        elapsed = (time.perf_counter() - t0) * 1000
        cpu_time_s += time.process_time() - c0
        rss = _rss_mb()
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        answer = (qr.answer or "").strip()
        cat = qr.failure_category or ""
        if cat == "timed_out":
            outcome_counts["timeout"] += 1
        elif cat.startswith("exception:"):
            outcome_counts["failure"] += 1
        elif (not answer) or ("insufficient" in answer.casefold()):
            outcome_counts["abstention"] += 1
        else:
            outcome_counts["success"] += 1
        by_type.setdefault(qtype or "unknown", []).append(elapsed)
        return PerformanceSample(
            question_id=qid,
            latency_ms=round(elapsed, 3),
            peak_rss_mb=rss,
            cold=cold,
            stages_ms=dict(qr.per_stage_latency_ms or {}),
        )

    # Ensure model exists
    if runner.model is None:
        from nexus.reasoning.model_interface import DummyModel

        runner.model = DummyModel()

    # Cold: first real call after process start
    first = questions[0]
    samples.append(
        _one(
            str(first["id"]),
            str(first["question"]),
            cold=True,
            qtype=str(first.get("question_type") or "unknown"),
        )
    )

    # Warmup remaining
    for i in range(max(0, warmup - 1)):
        q = questions[i % len(questions)]
        _one(
            str(q["id"]),
            str(q["question"]),
            cold=False,
            qtype=str(q.get("question_type") or "unknown"),
        )

    # Measurement repeats across the mixture
    for _rep in range(repeats):
        for q in questions:
            samples.append(
                _one(
                    str(q["id"]),
                    str(q["question"]),
                    cold=False,
                    qtype=str(q.get("question_type") or "unknown"),
                )
            )

    warm = [s.latency_ms for s in samples if not s.cold]
    cold = [s.latency_ms for s in samples if s.cold]
    budget_latency = _percentile(warm, 0.50)
    budget_rss = peak_rss

    latency_gate = "NOT_RUN"
    rss_gate = "NOT_RUN"
    if budget_latency is not None:
        latency_gate = "PASS" if budget_latency <= 500.0 else "FAIL"
    if budget_rss is not None:
        rss_gate = "PASS" if budget_rss <= 500.0 else "FAIL"

    setup = dict(setup_timings or {})
    setup_total = sum(float(v) for v in setup.values()) if setup else None
    steady_mean = (sum(warm) / len(warm)) if warm else None
    total_system_ms = None
    if setup_total is not None and steady_mean is not None:
        # Interpret total-system as setup once + one warm pass over the mixture.
        total_system_ms = round(setup_total + steady_mean * len(questions), 3)

    return {
        "schema_version": "nexus-performance-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "profile": profile_name,
        "realizer_backend": runner.config.realizer_backend,
        "allow_synth_fallback": bool(
            getattr(runner.config, "allow_synth_fallback", True)
        ),
        "config_hash": runner.config.config_hash,
        "config_identity_schema": runner.config.identity_schema,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "graph": dict(graph_meta or {}),
        "setup_timings_ms": setup,
        "methodology": {
            "warmup": warmup,
            "repeats": repeats,
            "questions": len(questions),
            "entry_point": "NEXUSRunner._run_single via measure_grounded_e2e",
            "cold_definition": "first measured call in this process",
            "setup_vs_steady_state": (
                "setup_timings_ms are outside warm percentiles; "
                "total_system_interpretation_ms = setup_total + mean_warm * n_questions"
            ),
            "note": (
                "End-to-end wall time around runner._run_single; "
                "not inferred from separate component campaigns. "
                "Mini-domain results must not be used as full-SAM evidence."
            ),
        },
        "samples": [
            {
                "question_id": s.question_id,
                "latency_ms": s.latency_ms,
                "peak_rss_mb": s.peak_rss_mb,
                "cold": s.cold,
                "stages_ms": s.stages_ms,
            }
            for s in samples
        ],
        "summary": {
            "cold_latency_ms": cold[0] if cold else None,
            "warm_n": len(warm),
            "warm_p50_ms": None if not warm else round(_percentile(warm, 0.50), 3),
            "warm_p95_ms": None if not warm else round(_percentile(warm, 0.95), 3),
            "warm_p99_ms": None if not warm else round(_percentile(warm, 0.99), 3),
            "peak_rss_mb": None if peak_rss is None else round(peak_rss, 3),
            "cpu_time_s": round(cpu_time_s, 3),
            "throughput_qps": (
                None
                if not warm
                else round(1000.0 / (sum(warm) / len(warm)), 4)
            ),
            "outcome_counts": outcome_counts,
            "by_question_type_warm_p50_ms": {
                qtype: round(_percentile(vals, 0.50), 3)
                for qtype, vals in sorted(by_type.items())
                if vals
            },
            "total_system_interpretation_ms": total_system_ms,
        },
        "budgets": {
            "latency_p50_ms_max": 500.0,
            "peak_rss_mb_max": 500.0,
            "latency_p50_gate": latency_gate,
            "peak_rss_gate": rss_gate,
        },
        "status": "VALID" if warm else "NOT_RUN",
    }
