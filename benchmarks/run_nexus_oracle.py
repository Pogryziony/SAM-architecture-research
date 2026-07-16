"""Build and evaluate the frozen-input NEXUS oracle benchmark.

This benchmark bypasses entity resolution with registered gold entities.  It
therefore measures traversal, evidence, realization, verification, and
abstention independently of SAM/ER3.  Existing validation questions provide
answer cases; manually curated relation labels provide exact path cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import canonical_json, normalize_question, sha256_file, sha256_json
from benchmarks.scoring import compute_fact_score
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner


ORACLE_SCHEMA_VERSION = "nexus-oracle-v1"
_ABSTAIN_MARKERS = ("insufficient evidence", "cannot answer", "unable to determine")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_oracle_records(
    validation_questions: Sequence[dict[str, Any]],
    relation_labels: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create the registered oracle contract without reading a frozen test split."""
    records: list[dict[str, Any]] = []
    for row in validation_questions:
        records.append({
            "id": "answer_" + str(row["id"]),
            "question": str(row["question"]),
            "gold_answer": str(row["answer"]),
            "gold_entities": sorted({str(value) for value in row.get("entities", [])}),
            "gold_path": [],
            "path_required": False,
            "should_abstain": False,
            "category": str(row.get("category", row.get("question_type", "answer"))),
            "source_split": "validation",
        })
    for index, row in enumerate(relation_labels):
        source, target, relation = str(row["source"]), str(row["target"]), str(row["edge_type"])
        negative = bool(row.get("is_negative", False))
        records.append({
            "id": f"relation_{index:03d}_{hashlib.sha256(canonical_json(row).encode()).hexdigest()[:10]}",
            "question": f"Does {source} have the {relation} relation to {target}?",
            "gold_answer": (
                "Insufficient evidence for that relation."
                if negative else f"Yes. {source} {relation} {target}."
            ),
            "gold_entities": sorted({source, target}),
            "gold_path": [] if negative else [{"source": source, "relation": relation, "target": target}],
            "path_required": not negative,
            "should_abstain": negative,
            "category": "negative_relation" if negative else "relation",
            "source_split": "relation_gold",
        })
    errors = validate_oracle_records(records)
    if errors:
        raise ValueError("invalid oracle records: " + "; ".join(errors))
    return sorted(records, key=lambda item: item["id"])


def validate_oracle_records(records: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not records:
        return ["oracle dataset is empty"]
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for index, row in enumerate(records):
        prefix = f"record {index}"
        for field in ("id", "question", "gold_answer", "category", "source_split"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                errors.append(f"{prefix}: missing {field}")
        record_id = str(row.get("id", ""))
        if record_id in seen_ids:
            errors.append(f"{prefix}: duplicate id {record_id}")
        seen_ids.add(record_id)
        normalized = normalize_question(str(row.get("question", "")))
        if normalized in seen_questions:
            errors.append(f"{prefix}: duplicate normalized question")
        seen_questions.add(normalized)
        entities = row.get("gold_entities")
        if not isinstance(entities, list) or not entities or not all(isinstance(item, str) and item for item in entities):
            errors.append(f"{prefix}: invalid gold_entities")
        if not isinstance(row.get("should_abstain"), bool):
            errors.append(f"{prefix}: should_abstain must be boolean")
        if not isinstance(row.get("path_required"), bool):
            errors.append(f"{prefix}: path_required must be boolean")
        path = row.get("gold_path")
        if not isinstance(path, list):
            errors.append(f"{prefix}: gold_path must be a list")
        elif row.get("path_required") and not path:
            errors.append(f"{prefix}: required gold_path is empty")
        else:
            for step in path:
                if not isinstance(step, dict) or not all(step.get(key) for key in ("source", "relation", "target")):
                    errors.append(f"{prefix}: malformed gold_path step")
    return errors


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), re.UNICODE))


def token_f1(predicted: str, gold: str) -> float:
    pred, truth = _tokens(predicted), _tokens(gold)
    if not pred or not truth:
        return 0.0
    overlap = len(pred & truth)
    precision, recall = overlap / len(pred), overlap / len(truth)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _proof_edges(audit: dict[str, Any]) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()
    for step in audit.get("proof_steps", []):
        source = str(step.get("from_node", ""))
        target = str(step.get("to_node", ""))
        if step.get("reversed"):
            source, target = target, source
        edges.add((source, str(step.get("relation", "")), target))
    return edges


def _path_recall(gold_path: Sequence[dict[str, Any]], audit: dict[str, Any]) -> float | None:
    if not gold_path:
        return None
    predicted = _proof_edges(audit)
    gold = {(str(step["source"]), str(step["relation"]), str(step["target"])) for step in gold_path}
    return len(gold & predicted) / len(gold)


def _entity_coverage(entities: Sequence[str], audit: dict[str, Any]) -> float:
    proof_entities = {
        str(value)
        for step in audit.get("proof_steps", [])
        for value in (step.get("from_node"), step.get("to_node"))
        if value
    }
    if not entities:
        return 0.0
    return len(set(entities) & proof_entities) / len(set(entities))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def run_oracle_benchmark(
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
    pipeline = runner.run_oracle(records, source_sha=source_sha)
    if pipeline.errors:
        raise RuntimeError("oracle pipeline failed: " + "; ".join(pipeline.errors))

    per_question: list[dict[str, Any]] = []
    fact_values: list[float] = []
    token_values: list[float] = []
    path_values: list[float] = []
    entity_values: list[float] = []
    latencies: list[float] = []
    tp = fp = fn = 0
    for row, qr in zip(records, pipeline.per_question, strict=True):
        fact = compute_fact_score(qr.answer, row["gold_answer"])["fuzzy_accuracy"]
        tokens = token_f1(qr.answer, row["gold_answer"])
        path_recall = _path_recall(row["gold_path"], qr.reasoning_audit)
        entity_coverage = _entity_coverage(row["gold_entities"], qr.reasoning_audit)
        predicted_abstain = qr.reasoning_action == "abstain" or any(
            marker in qr.answer.casefold() for marker in _ABSTAIN_MARKERS
        )
        gold_abstain = row["should_abstain"]
        tp += int(predicted_abstain and gold_abstain)
        fp += int(predicted_abstain and not gold_abstain)
        fn += int(not predicted_abstain and gold_abstain)
        if fact is not None:
            fact_values.append(float(fact))
        token_values.append(tokens)
        if path_recall is not None:
            path_values.append(path_recall)
        entity_values.append(entity_coverage)
        latency = sum(qr.per_stage_latency_ms.values())
        latencies.append(latency)
        per_question.append({
            "question_id": row["id"],
            "question_hash": hashlib.sha256(row["question"].encode()).hexdigest(),
            "category": row["category"],
            "answer": qr.answer,
            "gold_answer": row["gold_answer"],
            "fact_accuracy": fact,
            "token_f1": round(tokens, 4),
            "gold_path_recall": None if path_recall is None else round(path_recall, 4),
            "gold_entity_coverage": round(entity_coverage, 4),
            "should_abstain": gold_abstain,
            "predicted_abstain": predicted_abstain,
            "reasoning_action": qr.reasoning_action,
            "proof_valid": qr.proof_valid,
            "provenance_coverage": qr.provenance_coverage,
            "verifier_passed": qr.verifier_passed,
            "failure_category": qr.failure_category,
            "latency_ms": round(latency, 3),
            "reasoning_audit": qr.reasoning_audit,
        })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    abstention_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    artifact = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "config_hash": config.config_hash,
        "evaluation_mode": pipeline.evaluation_mode,
        "dataset": dataset_identity,
        "questions_total": len(records),
        "metrics": {
            "fact_accuracy_mean": round(sum(fact_values) / len(fact_values), 4) if fact_values else None,
            "token_f1_mean": round(sum(token_values) / len(token_values), 4),
            "gold_path_recall_mean": round(sum(path_values) / len(path_values), 4) if path_values else None,
            "gold_entity_coverage_mean": round(sum(entity_values) / len(entity_values), 4),
            "proof_valid_rate": round(sum(item["proof_valid"] for item in per_question) / len(per_question), 4),
            "provenance_coverage_mean": round(sum(item["provenance_coverage"] for item in per_question) / len(per_question), 4),
            "abstention_precision": round(precision, 4),
            "abstention_recall": round(recall, 4),
            "abstention_f1": round(abstention_f1, 4),
            "latency_p50_ms": round(_percentile(latencies, 0.50), 3),
            "latency_p95_ms": round(_percentile(latencies, 0.95), 3),
            "latency_p99_ms": round(_percentile(latencies, 0.99), 3),
        },
        "per_question": per_question,
        "errors": [],
    }
    guard_errors = validate_oracle_artifact(artifact)
    if guard_errors:
        raise RuntimeError("oracle publication guard failed: " + "; ".join(guard_errors))
    return artifact


def validate_oracle_artifact(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema_version") != ORACLE_SCHEMA_VERSION:
        errors.append("invalid schema version")
    if artifact.get("evaluation_mode") != "oracle":
        errors.append("evaluation_mode must be oracle")
    if not artifact.get("source_sha") or not artifact.get("config_hash"):
        errors.append("missing source/config hash")
    total = artifact.get("questions_total")
    rows = artifact.get("per_question")
    if not isinstance(total, int) or total <= 0:
        errors.append("questions_total must be positive")
    if not isinstance(rows, list) or len(rows) != total:
        errors.append("per_question count mismatch")
    elif any(not row.get("question_id") or not row.get("reasoning_audit") for row in rows):
        errors.append("incomplete per-question audit")
    dataset = artifact.get("dataset", {})
    if not dataset.get("sha256") or not dataset.get("sources"):
        errors.append("missing dataset identity")
    metrics = artifact.get("metrics", {})
    for key, value in metrics.items():
        if value is None and key in {"fact_accuracy_mean", "gold_path_recall_mean"}:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"invalid metric {key}")
        elif not key.startswith("latency_") and not 0.0 <= float(value) <= 1.0:
            errors.append(f"out-of-range metric {key}")
    if artifact.get("errors"):
        errors.append("artifact contains pipeline errors")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="stack/encoder/data/val.jsonl")
    parser.add_argument("--relations", default="benchmarks/qa-dataset/relation_gold.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if Path(args.questions).name.casefold() == "test.jsonl":
        raise ValueError("the consumed frozen test split is forbidden")
    question_rows = _read_jsonl(Path(args.questions))
    if args.limit:
        question_rows = question_rows[: args.limit]
    relation_rows = _read_jsonl(Path(args.relations))
    records = build_oracle_records(question_rows, relation_rows)
    dataset_identity = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "sha256": sha256_json(records),
        "sources": {
            args.questions: sha256_file(Path(args.questions)),
            args.relations: sha256_file(Path(args.relations)),
        },
    }
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, _ = build_benchmark_graph()
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    artifact = run_oracle_benchmark(
        records, graph, source_sha=source_sha, dataset_identity=dataset_identity
    )
    output = Path(args.output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(canonical_json({"status": "VALID", "questions": len(records), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
