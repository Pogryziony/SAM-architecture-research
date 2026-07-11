"""Candidate-pool diagnostic — train and validation only.

Does not read the consumed frozen split.  Reports exhaustive
canonical-vocabulary ranking statistics and invariant checks.

Usage:
    python -m benchmarks.diagnose_candidate_pool
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from stack.encoder.canonical_mapping import _is_canonical_id
from stack.encoder.loader import get_peak_rss_mb
from stack.encoder.trivial_baseline import candidate_pool


def run_diagnostic(root: str | Path = ".") -> dict:
    root = Path(root)

    utc_now = datetime.now(timezone.utc)
    run_ts = utc_now.strftime("%Y%m%dT%H%M%SZ")

    results = {}
    for split_name, split_path in [
        ("train", root / "stack/encoder/data/train.jsonl"),
        ("validation", root / "stack/encoder/data/val.jsonl"),
    ]:
        questions = [
            json.loads(line)
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        from benchmarks.run_benchmark import build_benchmark_graph
        graph, _ = build_benchmark_graph()

        canonical_count = sum(
            1 for nid in graph._nodes if _is_canonical_id(str(nid))
        )

        pool_sizes: list[int] = []
        lexical_ceiling_hits = 0
        exhaustive_ceiling_hits = 0
        missing_gold_questions = 0
        total_gold = 0
        latencies: list[float] = []

        rss_before = get_peak_rss_mb()

        for record in questions:
            gold = set(str(e) for e in record.get("entities", []))
            total_gold += len(gold)

            t0 = time.perf_counter()
            pool = candidate_pool(record["question"], graph)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            cand_ids = [str(item["node_id"]) for item in pool]
            cand_set = set(cand_ids)
            pool_sizes.append(len(cand_ids))

            # Lexical-only ceiling (before canonical augmentation)
            non_canonical_cand = {
                cid for cid in cand_set if not _is_canonical_id(cid)
            }
            lexical_ceiling_hits += len(gold & non_canonical_cand)

            # Exhaustive canonical ceiling (with augmentation)
            exhaustive_ceiling_hits += len(gold & cand_set)

            if not (gold & cand_set):
                missing_gold_questions += 1

        rss_after = get_peak_rss_mb()
        latencies.sort()

        # Invariant: minimum pool >= canonical count
        min_pool = min(pool_sizes) if pool_sizes else 0
        invariant_ok = min_pool >= canonical_count

        results[split_name] = {
            "questions": len(questions),
            "total_gold_entities": total_gold,
            "canonical_pattern_nodes": canonical_count,
            "pool_size_min": min_pool,
            "pool_size_max": max(pool_sizes) if pool_sizes else 0,
            "pool_size_mean": statistics.mean(pool_sizes) if pool_sizes else 0.0,
            "pool_size_median": statistics.median(pool_sizes) if pool_sizes else 0.0,
            "pool_size_p95": (
                pool_sizes[int(len(pool_sizes) * 0.95)]
                if len(pool_sizes) > 1 else (pool_sizes[0] if pool_sizes else 0)
            ),
            "lexical_only_candidate_ceiling": (
                lexical_ceiling_hits / total_gold if total_gold else 0.0
            ),
            "exhaustive_canonical_candidate_ceiling": (
                exhaustive_ceiling_hits / total_gold if total_gold else 0.0
            ),
            "questions_with_missing_gold": missing_gold_questions,
            "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
            "latency_p95_ms": (
                latencies[int(len(latencies) * 0.95)]
                if len(latencies) > 1 else (latencies[0] if latencies else 0.0)
            ),
            "peak_rss_mb": rss_after - rss_before,
            "invariant_min_pool_ge_canonical": invariant_ok,
        }

        if not invariant_ok:
            raise AssertionError(
                f"{split_name}: minimum pool size {min_pool} < "
                f"canonical node count {canonical_count}. "
                f"Invariant violated: exhaustive canonical-vocabulary "
                f"ranking requires all canonical nodes in every pool."
            )

    # Write artifact
    artifact = {
        "diagnostic": "candidate_pool_v3",
        "run_timestamp_utc": utc_now.isoformat(),
        "split_results": results,
        "architecture": (
            "exhaustive canonical-vocabulary ranking with "
            "lexical and graph-derived candidate augmentation"
        ),
    }

    out_dir = root / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"candidate_pool_diagnostic_{run_ts}.json"

    # Only write if path doesn't exist
    if not out_path.exists():
        out_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return artifact


if __name__ == "__main__":
    result = run_diagnostic()
    for split, stats in result["split_results"].items():
        print(f"\n{split}:")
        print(f"  canonical nodes: {stats['canonical_pattern_nodes']}")
        print(f"  pool sizes: min={stats['pool_size_min']}, max={stats['pool_size_max']}, "
              f"mean={stats['pool_size_mean']:.1f}, p50={stats['pool_size_median']:.1f}, "
              f"p95={stats['pool_size_p95']:.1f}")
        print(f"  lexical ceiling: {stats['lexical_only_candidate_ceiling']:.4f}")
        print(f"  exhaustive ceiling: {stats['exhaustive_canonical_candidate_ceiling']:.4f}")
        print(f"  missing gold questions: {stats['questions_with_missing_gold']}")
        print(f"  invariant (min_pool >= canonical): {stats['invariant_min_pool_ge_canonical']}")
