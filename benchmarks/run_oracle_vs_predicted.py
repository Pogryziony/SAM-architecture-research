"""Paired oracle versus predicted reporting on the same frozen records.

Oracle mode uses gold entry entities; predicted mode uses the normal ER path.
Publication requires both modes and per-question paired deltas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import canonical_json, sha256_file
from benchmarks.run_nexus_oracle import (
    ORACLE_SCHEMA_VERSION,
    _entity_coverage,
    _path_recall,
    _percentile,
    token_f1,
    validate_oracle_records,
)
from benchmarks.scoring import compute_fact_score
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner


PAIRED_SCHEMA_VERSION = "nexus-oracle-vs-predicted-v1"
_ABSTAIN_MARKERS = ("insufficient evidence", "cannot answer", "unable to determine")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _is_abstain(answer: str, reasoning_action: str) -> bool:
    return reasoning_action == "abstain" or any(
        marker in answer.casefold() for marker in _ABSTAIN_MARKERS
    )


def score_mode_rows(
    records: Sequence[dict[str, Any]],
    per_question: Sequence[Any],
) -> list[dict[str, Any]]:
    """Score pipeline rows against frozen oracle gold."""
    rows: list[dict[str, Any]] = []
    for record, qr in zip(records, per_question, strict=True):
        fact = compute_fact_score(qr.answer, record["gold_answer"])["fuzzy_accuracy"]
        tokens = token_f1(qr.answer, record["gold_answer"])
        path_recall = _path_recall(record["gold_path"], qr.reasoning_audit)
        entity_coverage = _entity_coverage(record["gold_entities"], qr.reasoning_audit)
        predicted_abstain = _is_abstain(qr.answer, qr.reasoning_action)
        latency = sum(qr.per_stage_latency_ms.values())
        rows.append({
            "question_id": record["id"],
            "category": record["category"],
            "answer": qr.answer,
            "fact_accuracy": fact,
            "token_f1": round(tokens, 4),
            "gold_path_recall": None if path_recall is None else round(path_recall, 4),
            "gold_entity_coverage": round(entity_coverage, 4),
            "should_abstain": record["should_abstain"],
            "predicted_abstain": predicted_abstain,
            "reasoning_action": qr.reasoning_action,
            "proof_valid": qr.proof_valid,
            "provenance_coverage": qr.provenance_coverage,
            "latency_ms": round(latency, 3),
        })
    return rows


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fact_values = [float(r["fact_accuracy"]) for r in rows if r["fact_accuracy"] is not None]
    token_values = [float(r["token_f1"]) for r in rows]
    path_values = [float(r["gold_path_recall"]) for r in rows if r["gold_path_recall"] is not None]
    entity_values = [float(r["gold_entity_coverage"]) for r in rows]
    latencies = [float(r["latency_ms"]) for r in rows]
    tp = sum(1 for r in rows if r["predicted_abstain"] and r["should_abstain"])
    fp = sum(1 for r in rows if r["predicted_abstain"] and not r["should_abstain"])
    fn = sum(1 for r in rows if not r["predicted_abstain"] and r["should_abstain"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    abstention_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "fact_accuracy_mean": round(sum(fact_values) / len(fact_values), 4) if fact_values else None,
        "token_f1_mean": round(sum(token_values) / len(token_values), 4) if token_values else 0.0,
        "gold_path_recall_mean": round(sum(path_values) / len(path_values), 4) if path_values else None,
        "gold_entity_coverage_mean": round(sum(entity_values) / len(entity_values), 4) if entity_values else 0.0,
        "proof_valid_rate": round(sum(1 for r in rows if r["proof_valid"]) / len(rows), 4) if rows else 0.0,
        "provenance_coverage_mean": round(
            sum(float(r["provenance_coverage"]) for r in rows) / len(rows), 4
        ) if rows else 0.0,
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(abstention_f1, 4),
        "latency_p50_ms": round(_percentile(latencies, 0.50), 3),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 3),
    }


def pair_rows(
    oracle_rows: Sequence[dict[str, Any]],
    predicted_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join oracle and predicted scores by question_id with deltas."""
    predicted_by_id = {row["question_id"]: row for row in predicted_rows}
    paired: list[dict[str, Any]] = []
    for oracle in oracle_rows:
        qid = oracle["question_id"]
        predicted = predicted_by_id.get(qid)
        if predicted is None:
            raise ValueError(f"missing predicted row for {qid}")
        o_fact = oracle.get("fact_accuracy")
        p_fact = predicted.get("fact_accuracy")
        delta_fact = None
        if o_fact is not None and p_fact is not None:
            delta_fact = round(float(o_fact) - float(p_fact), 4)
        paired.append({
            "question_id": qid,
            "category": oracle["category"],
            "oracle": oracle,
            "predicted": predicted,
            "delta": {
                "fact_accuracy": delta_fact,
                "token_f1": round(float(oracle["token_f1"]) - float(predicted["token_f1"]), 4),
                "gold_entity_coverage": round(
                    float(oracle["gold_entity_coverage"]) - float(predicted["gold_entity_coverage"]),
                    4,
                ),
            },
        })
    return paired


def validate_paired_artifact(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema_version") != PAIRED_SCHEMA_VERSION:
        errors.append("invalid schema version")
    for mode in ("oracle", "predicted"):
        block = artifact.get(mode)
        if not isinstance(block, dict):
            errors.append(f"missing {mode} block")
            continue
        if block.get("evaluation_mode") != mode:
            errors.append(f"{mode} evaluation_mode mismatch")
        if not isinstance(block.get("metrics"), dict):
            errors.append(f"{mode} metrics missing")
    paired = artifact.get("paired")
    if not isinstance(paired, list) or not paired:
        errors.append("paired rows missing")
    elif any(not row.get("question_id") or "delta" not in row for row in paired):
        errors.append("incomplete paired rows")
    dataset = artifact.get("dataset", {})
    if not dataset.get("sha256") and not dataset.get("file_sha256"):
        errors.append("missing dataset identity")
    return errors


def run_paired_benchmark(
    records: list[dict[str, Any]],
    graph: InMemoryGraphStore,
    *,
    source_sha: str,
    dataset_identity: dict[str, Any],
) -> dict[str, Any]:
    errors = validate_oracle_records(records)
    if errors:
        raise ValueError("invalid oracle records: " + "; ".join(errors))

    config = ProductionNEXUSConfig.lexical_only()
    runner = NEXUSRunner(graph, config)

    oracle_pipeline = runner.run_oracle(records, source_sha=source_sha)
    if oracle_pipeline.errors:
        raise RuntimeError("oracle pipeline failed: " + "; ".join(oracle_pipeline.errors))
    predicted_pipeline = runner.run(records, source_sha=source_sha)
    if predicted_pipeline.errors:
        raise RuntimeError("predicted pipeline failed: " + "; ".join(predicted_pipeline.errors))

    oracle_rows = score_mode_rows(records, oracle_pipeline.per_question)
    predicted_rows = score_mode_rows(records, predicted_pipeline.per_question)
    paired = pair_rows(oracle_rows, predicted_rows)
    artifact = {
        "schema_version": PAIRED_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "config_hash": config.config_hash,
        "oracle_schema_version": ORACLE_SCHEMA_VERSION,
        "dataset": dataset_identity,
        "questions_total": len(records),
        "oracle": {
            "evaluation_mode": "oracle",
            "metrics": summarize_rows(oracle_rows),
        },
        "predicted": {
            "evaluation_mode": "predicted",
            "metrics": summarize_rows(predicted_rows),
        },
        "paired": paired,
        "errors": [],
    }
    guard_errors = validate_paired_artifact(artifact)
    if guard_errors:
        raise RuntimeError("paired publication guard failed: " + "; ".join(guard_errors))
    # Finite check on latency metrics
    for mode in ("oracle", "predicted"):
        for key, value in artifact[mode]["metrics"].items():
            if value is None:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"invalid {mode} metric {key}")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=str(_project_root / "benchmarks" / "qa-dataset" / "oracle_v1.jsonl"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    records_path = Path(args.records)
    records = _read_jsonl(records_path)
    if args.limit:
        records = records[: args.limit]
    dataset_identity = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "file_sha256": sha256_file(records_path),
        "record_count": len(records),
        "sources": {str(records_path.as_posix()): sha256_file(records_path)},
    }
    from benchmarks.run_benchmark import build_benchmark_graph

    graph, _ = build_benchmark_graph()
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    artifact = run_paired_benchmark(
        records, graph, source_sha=source_sha, dataset_identity=dataset_identity
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(canonical_json({"status": "VALID", "questions": len(records), "sha256": digest, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
