"""Synthetic small/medium/large traversal budget campaign (Stage 2 gate).

Builds deterministic synthetic graphs, runs beam search under default and tight
budgets, and records latency percentiles plus RSS. Truncation must be reported
when budgets are exhausted; peak RSS must stay inside the NEXUS hard limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import canonical_json
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.graph.traversal import TraversalStats, beam_search
from nexus.utils.config import NEXUSConfig

CAMPAIGN_SCHEMA_VERSION = "nexus-traversal-budget-campaign-v1"
PREREGISTRATION_ID = "traversal-budgets-v1"
REFERENCE_CPU_LABEL = "github-actions-ubuntu-latest-x86_64"

# NEXUS hard limits from docs/nexus-auditability-roadmap.md (proposed).
RSS_HARD_LIMIT_MB = 250.0
P95_HARD_LIMIT_MS = 450.0
P50_HARD_LIMIT_MS = 250.0

GRAPH_SPECS: dict[str, dict[str, int]] = {
    "small": {"nodes": 50, "branching": 2},
    "medium": {"nodes": 500, "branching": 3},
    "large": {"nodes": 5_000, "branching": 3},
}


def peak_rss_mb() -> float | None:
    """Best-effort current/peak RSS in MB; None if unavailable."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB; macOS: bytes
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        return usage / 1024.0
    except (ImportError, AttributeError):
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def build_synthetic_graph(node_count: int, branching: int = 2) -> InMemoryGraphStore:
    """Deterministic layered DAG used for budget measurement."""
    graph = InMemoryGraphStore()
    for index in range(node_count):
        graph.add_node(
            Node(
                id=f"n{index}",
                type="Entity",
                properties={"name": f"node-{index}"},
                sources=[f"synthetic://node/{index}"],
            )
        )
    for index in range(node_count):
        for offset in range(1, branching + 1):
            target = index + offset
            if target >= node_count:
                break
            graph.add_edge(
                Edge(
                    type="depends_on",
                    source=f"n{index}",
                    target=f"n{target}",
                    confidence=1.0,
                    evidence=f"synthetic://edge/{index}/{target}",
                )
            )
    return graph


def run_traversal_samples(
    graph: InMemoryGraphStore,
    config: NEXUSConfig,
    *,
    starts: list[str],
    repeats: int = 5,
) -> dict[str, Any]:
    """Run beam search from several starts and summarize latency/truncation."""
    latencies: list[float] = []
    truncated_runs = 0
    reasons: dict[str, int] = {}
    last_stats: dict[str, Any] = {}
    for _ in range(repeats):
        for start in starts:
            stats = TraversalStats()
            t0 = time.perf_counter()
            paths = beam_search(
                graph,
                start_nodes=[start],
                query_entities={start},
                direction="out",
                config=config,
                stats=stats,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            last_stats = stats.to_dict()
            last_stats["paths"] = len(paths)
            if stats.truncated:
                truncated_runs += 1
                reason = stats.truncation_reason or "unknown"
                reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "samples": len(latencies),
        "latency_p50_ms": round(_percentile(latencies, 0.50), 3),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 3),
        "latency_max_ms": round(max(latencies) if latencies else 0.0, 3),
        "truncated_runs": truncated_runs,
        "truncation_reasons": dict(sorted(reasons.items())),
        "last_stats": last_stats,
    }


def run_size_campaign(size_name: str, spec: dict[str, int]) -> dict[str, Any]:
    graph = build_synthetic_graph(spec["nodes"], spec["branching"])
    starts = [f"n{i}" for i in (0, max(0, spec["nodes"] // 4), max(0, spec["nodes"] // 2))]
    default_cfg = NEXUSConfig(max_depth=4, beam_width=25)
    tight_cfg = NEXUSConfig(
        max_depth=4,
        beam_width=25,
        max_expanded_edges=8,
        max_expanded_nodes=8,
        max_traversal_ms=0.0,
    )
    rss_before = peak_rss_mb()
    default_metrics = run_traversal_samples(graph, default_cfg, starts=starts, repeats=3)
    tight_metrics = run_traversal_samples(graph, tight_cfg, starts=starts, repeats=2)
    rss_after = peak_rss_mb()
    peak = None
    if rss_before is not None and rss_after is not None:
        peak = round(max(rss_before, rss_after), 3)
    elif rss_after is not None:
        peak = round(rss_after, 3)
    return {
        "size": size_name,
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "peak_rss_mb": peak,
        "default_budgets": default_metrics,
        "tight_budgets": tight_metrics,
    }


def _reference_identity() -> dict[str, Any]:
    source_sha = ""
    try:
        source_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(_project_root)
        ).strip()
    except Exception:
        source_sha = ""
    return {
        "preregistration_id": PREREGISTRATION_ID,
        "reference_cpu_label": REFERENCE_CPU_LABEL,
        "platform": sys.platform,
        "python_version": platform.python_version(),
        "cpu_model": platform.processor() or "",
        "source_sha": source_sha,
    }


def validate_campaign_artifact(artifact: dict[str, Any]) -> list[str]:
    """Publication guards for the Stage 2 synthetic campaign."""
    errors: list[str] = []
    if artifact.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        errors.append("invalid schema version")
    if artifact.get("preregistration_id") != PREREGISTRATION_ID:
        errors.append("missing or invalid preregistration_id")
    if artifact.get("reference_cpu_label") != REFERENCE_CPU_LABEL:
        errors.append("missing or invalid reference_cpu_label")
    for key in ("platform", "python_version", "cpu_model", "source_sha"):
        if key not in artifact:
            errors.append(f"missing identity field: {key}")
    sizes = artifact.get("sizes")
    if not isinstance(sizes, list) or not sizes:
        errors.append("sizes missing")
        return errors
    requested = artifact.get("sizes_requested")
    if requested is None:
        expected = set(GRAPH_SPECS)
    else:
        expected = set(requested)
        unknown = expected - set(GRAPH_SPECS)
        if unknown:
            errors.append(f"unknown sizes requested: {sorted(unknown)}")
    seen = {row.get("size") for row in sizes if isinstance(row, dict)}
    if seen != expected:
        errors.append(f"expected sizes {sorted(expected)}, got {sorted(x for x in seen if x)}")
    for row in sizes:
        if not isinstance(row, dict):
            errors.append("size row is not an object")
            continue
        label = row.get("size", "?")
        default = row.get("default_budgets") or {}
        tight = row.get("tight_budgets") or {}
        if default.get("truncated_runs", 0) != 0:
            errors.append(f"{label}: default budgets unexpectedly truncated")
        if tight.get("truncated_runs", 0) <= 0:
            errors.append(f"{label}: tight budgets must truncate")
        p95 = float(default.get("latency_p95_ms") or 0.0)
        p50 = float(default.get("latency_p50_ms") or 0.0)
        if p95 > P95_HARD_LIMIT_MS:
            errors.append(f"{label}: p95 {p95} exceeds hard limit {P95_HARD_LIMIT_MS}")
        if p50 > P50_HARD_LIMIT_MS:
            errors.append(f"{label}: p50 {p50} exceeds hard limit {P50_HARD_LIMIT_MS}")
        rss = row.get("peak_rss_mb")
        if rss is not None and float(rss) > RSS_HARD_LIMIT_MB:
            errors.append(f"{label}: RSS {rss} exceeds hard limit {RSS_HARD_LIMIT_MB}")
    if artifact.get("errors"):
        errors.append("artifact contains errors")
    return errors


def run_campaign(sizes_requested: list[str] | None = None) -> dict[str, Any]:
    requested = list(sizes_requested) if sizes_requested else list(GRAPH_SPECS)
    for name in requested:
        if name not in GRAPH_SPECS:
            raise ValueError(f"unknown size: {name}")
    sizes = [run_size_campaign(name, GRAPH_SPECS[name]) for name in requested]
    artifact = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sizes_requested": requested,
        "limits": {
            "rss_hard_limit_mb": RSS_HARD_LIMIT_MB,
            "p50_hard_limit_ms": P50_HARD_LIMIT_MS,
            "p95_hard_limit_ms": P95_HARD_LIMIT_MS,
        },
        "sizes": sizes,
        "errors": [],
        **_reference_identity(),
    }
    guard = validate_campaign_artifact(artifact)
    if guard:
        artifact["errors"] = guard
        raise RuntimeError("campaign publication guard failed: " + "; ".join(guard))
    artifact["status"] = "PASS"
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=tuple(GRAPH_SPECS),
        default=list(GRAPH_SPECS),
        help="Subset of graph sizes (CI uses: --sizes small).",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite: {output}")
    artifact = run_campaign(list(args.sizes))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(canonical_json({
        "status": "PASS",
        "sha256": digest,
        "output": str(output),
        "sizes": list(args.sizes),
        "preregistration_id": PREREGISTRATION_ID,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
