"""Paired oracle versus predicted reporting on the same frozen records.

Oracle mode uses gold entry entities; predicted mode uses the stack ER path.
Default predicted resolver is lexical∪ER3 (`union`) with question-gated handoff
over the frozen Entity Ranker V3 checkpoint (no new training). Use
`--predicted-resolver er3` or `lexical` for historical baselines.
Publication requires both modes, ER decomposition metrics, and paired deltas.
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
from nexus.reasoning.model_interface import (
    DummyModel,
    SynthesizingModel,
    get_available_model,
)


PAIRED_SCHEMA_VERSION = "nexus-oracle-vs-predicted-v2"
DEFAULT_ER3_DIR = "models/encoder/entity_ranker_v3_20260711T081545Z"
_ABSTAIN_MARKERS = ("insufficient evidence", "cannot answer", "unable to determine")
_REALIZER_BACKENDS = (
    "synth",
    "pointer_copy",
    "grounded_v1",
    "deterministic_render",
    "l1_acceptance",
)


def _realizer_overrides(realizer_backend: str) -> dict[str, Any]:
    """Map CLI realizer backend to ProductionNEXUSConfig kwargs."""
    if realizer_backend == "synth":
        return {"realizer_backend": "synth"}
    if realizer_backend == "pointer_copy":
        cfg = ProductionNEXUSConfig.pointer_copy()
        return {
            "realizer_backend": cfg.realizer_backend,
            "require_structured_provenance": cfg.require_structured_provenance,
        }
    if realizer_backend == "deterministic_render":
        cfg = ProductionNEXUSConfig.deterministic_render()
        return {
            "realizer_backend": cfg.realizer_backend,
            "require_structured_provenance": cfg.require_structured_provenance,
        }
    if realizer_backend == "l1_acceptance":
        cfg = ProductionNEXUSConfig.l1_acceptance()
        return {
            "realizer_backend": cfg.realizer_backend,
            "realizer_model_dir": cfg.realizer_model_dir,
            "realizer_config_path": cfg.realizer_config_path,
            "realizer_checkpoint_sha256": cfg.realizer_checkpoint_sha256,
            "require_structured_provenance": cfg.require_structured_provenance,
        }
    if realizer_backend == "grounded_v1":
        cfg = ProductionNEXUSConfig.grounded()
        return {
            "realizer_backend": cfg.realizer_backend,
            "realizer_model_dir": cfg.realizer_model_dir,
            "realizer_config_path": cfg.realizer_config_path,
            "realizer_checkpoint_sha256": cfg.realizer_checkpoint_sha256,
            "require_structured_provenance": cfg.require_structured_provenance,
        }
    raise ValueError(f"unsupported realizer backend: {realizer_backend}")


def _select_model(model_name: str) -> Any:
    if model_name == "dummy":
        return DummyModel()
    if model_name == "synth":
        return SynthesizingModel()
    if model_name == "auto":
        return get_available_model()
    raise ValueError(f"unsupported model: {model_name}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _is_abstain(answer: str, reasoning_action: str) -> bool:
    return reasoning_action == "abstain" or any(
        marker in answer.casefold() for marker in _ABSTAIN_MARKERS
    )


def set_recall(gold: Sequence[str], predicted: Sequence[str]) -> float:
    """Fraction of gold IDs present in predicted IDs."""
    truth = {str(item) for item in gold if item}
    if not truth:
        return 0.0
    pred = {str(item) for item in predicted if item}
    return len(truth & pred) / len(truth)


def _candidate_ids(qr: Any) -> list[str]:
    candidates = getattr(qr, "resolution_candidates", None) or []
    ids: list[str] = []
    for item in candidates:
        if isinstance(item, dict) and item.get("entity_id"):
            ids.append(str(item["entity_id"]))
        else:
            entity_id = getattr(item, "entity_id", None)
            if entity_id:
                ids.append(str(entity_id))
    if ids:
        return ids
    return list(getattr(qr, "predicted_entities", None) or [])


def score_mode_rows(
    records: Sequence[dict[str, Any]],
    per_question: Sequence[Any],
) -> list[dict[str, Any]]:
    """Score pipeline rows against frozen oracle gold, including ER layers."""
    rows: list[dict[str, Any]] = []
    for record, qr in zip(records, per_question, strict=True):
        fact = compute_fact_score(qr.answer, record["gold_answer"])["fuzzy_accuracy"]
        tokens = token_f1(qr.answer, record["gold_answer"])
        path_recall = _path_recall(record["gold_path"], qr.reasoning_audit)
        entity_coverage = _entity_coverage(record["gold_entities"], qr.reasoning_audit)
        entry_ids = list(getattr(qr, "selected_entry_nodes", None) or getattr(qr, "predicted_entities", None) or [])
        pool_ids = _candidate_ids(qr)
        entry_recall = set_recall(record["gold_entities"], entry_ids)
        pool_recall = set_recall(record["gold_entities"], pool_ids)
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
            "entry_recall": round(entry_recall, 4),
            "pool_recall": round(pool_recall, 4),
            "selected_entry_nodes": entry_ids,
            "entity_resolution_method": getattr(qr, "entity_resolution_method", ""),
            "should_abstain": record["should_abstain"],
            "predicted_abstain": predicted_abstain,
            "reasoning_action": qr.reasoning_action,
            "proof_valid": qr.proof_valid,
            "provenance_coverage": qr.provenance_coverage,
            "latency_ms": round(latency, 3),
        })
    return rows


def _max_identical_entry_pack(rows: Sequence[dict[str, Any]]) -> int:
    from collections import Counter

    packs = [
        tuple(row.get("selected_entry_nodes") or [])
        for row in rows
    ]
    if not packs:
        return 0
    return int(Counter(packs).most_common(1)[0][1])


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question metrics with explicit denominators.

    Means that drop unscorable rows (``None``) always publish ``*_n_scored``
    alongside ``questions_total`` so a reduced scorable subset cannot hide.
    """
    n_total = len(rows)
    fact_values = [float(r["fact_accuracy"]) for r in rows if r["fact_accuracy"] is not None]
    token_values = [float(r["token_f1"]) for r in rows]
    path_values = [float(r["gold_path_recall"]) for r in rows if r["gold_path_recall"] is not None]
    entity_values = [float(r["gold_entity_coverage"]) for r in rows]
    entry_values = [float(r.get("entry_recall", 0.0)) for r in rows]
    pool_values = [float(r.get("pool_recall", 0.0)) for r in rows]
    latencies = [float(r["latency_ms"]) for r in rows]
    tp = sum(1 for r in rows if r["predicted_abstain"] and r["should_abstain"])
    fp = sum(1 for r in rows if r["predicted_abstain"] and not r["should_abstain"])
    fn = sum(1 for r in rows if not r["predicted_abstain"] and r["should_abstain"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    abstention_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "questions_total": n_total,
        "fact_accuracy_mean": round(sum(fact_values) / len(fact_values), 4) if fact_values else None,
        "fact_accuracy_n_scored": len(fact_values),
        "fact_accuracy_n_unscorable": n_total - len(fact_values),
        "token_f1_mean": round(sum(token_values) / len(token_values), 4) if token_values else 0.0,
        "token_f1_n_scored": len(token_values),
        "gold_path_recall_mean": round(sum(path_values) / len(path_values), 4) if path_values else None,
        "gold_path_recall_n_scored": len(path_values),
        "gold_path_recall_n_unscorable": n_total - len(path_values),
        "gold_entity_coverage_mean": round(sum(entity_values) / len(entity_values), 4) if entity_values else 0.0,
        "gold_entity_coverage_n_scored": len(entity_values),
        "entry_recall_mean": round(sum(entry_values) / len(entry_values), 4) if entry_values else 0.0,
        "entry_recall_n_scored": len(entry_values),
        "pool_recall_mean": round(sum(pool_values) / len(pool_values), 4) if pool_values else 0.0,
        "pool_recall_n_scored": len(pool_values),
        "max_identical_entry_pack": _max_identical_entry_pack(rows),
        "proof_valid_rate": round(sum(1 for r in rows if r["proof_valid"]) / len(rows), 4) if rows else 0.0,
        "proof_valid_n_scored": n_total,
        "provenance_coverage_mean": round(
            sum(float(r["provenance_coverage"]) for r in rows) / len(rows), 4
        ) if rows else 0.0,
        "provenance_coverage_n_scored": n_total,
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(abstention_f1, 4),
        "abstention_tp": tp,
        "abstention_fp": fp,
        "abstention_fn": fn,
        "latency_p50_ms": round(_percentile(latencies, 0.50), 3),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 3),
        "latency_n_scored": len(latencies),
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
                "entry_recall": round(
                    float(oracle.get("entry_recall", 0.0)) - float(predicted.get("entry_recall", 0.0)),
                    4,
                ),
                "pool_recall": round(
                    float(oracle.get("pool_recall", 0.0)) - float(predicted.get("pool_recall", 0.0)),
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
        metrics = block.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"{mode} metrics missing")
        else:
            for key in ("entry_recall_mean", "pool_recall_mean"):
                if key not in metrics:
                    errors.append(f"{mode} missing {key}")
    if not artifact.get("predicted_resolver"):
        errors.append("missing predicted_resolver identity")
    paired = artifact.get("paired")
    if not isinstance(paired, list) or not paired:
        errors.append("paired rows missing")
    elif any(not row.get("question_id") or "delta" not in row for row in paired):
        errors.append("incomplete paired rows")
    dataset = artifact.get("dataset", {})
    if not dataset.get("sha256") and not dataset.get("file_sha256"):
        errors.append("missing dataset identity")
    return errors


def build_predicted_runner(
    graph: InMemoryGraphStore,
    *,
    predicted_resolver: str,
    er3_dir: str,
    model: Any | None = None,
    union_top_k: int = 12,
    union_anchor_k: int = 8,
    realizer_backend: str = "synth",
) -> tuple[NEXUSRunner, dict[str, Any]]:
    """Construct the predicted-mode runner and resolver identity metadata."""
    realizer_kwargs = _realizer_overrides(realizer_backend)
    if predicted_resolver == "lexical":
        config = ProductionNEXUSConfig.lexical_only(**realizer_kwargs)
        runner = NEXUSRunner(graph, config, model=model)
        identity = {
            "name": "lexical",
            "config_hash": config.config_hash,
            "entity_ranker_v3_enabled": False,
            "realizer_backend": realizer_backend,
        }
        return runner, identity

    from stack.pipeline.resolver import ER3Resolver, LexicalResolver, UnionResolver

    if predicted_resolver == "er3":
        config = ProductionNEXUSConfig.with_entity_ranker_v3(er3_dir, **realizer_kwargs)
        resolver = ER3Resolver.from_directory(er3_dir, graph)
        runner = NEXUSRunner(graph, config, model=model, entity_resolver=resolver)
        identity = {
            "name": "entity_ranker_v3",
            "model_dir": er3_dir,
            "config_hash": config.config_hash,
            "entity_ranker_v3_enabled": True,
            "max_entry_nodes": config.max_entry_nodes,
            "realizer_backend": realizer_backend,
        }
        return runner, identity

    if predicted_resolver != "union":
        raise ValueError(f"unsupported predicted resolver: {predicted_resolver}")

    # Union handoff needs a slightly larger entry budget than raw ER3 top-10,
    # but still far below Stage 1D pool caps so hub noise stays bounded.
    config = ProductionNEXUSConfig.with_entity_ranker_v3(
        er3_dir, max_entry_nodes=max(12, int(union_top_k)), **realizer_kwargs
    )
    er3 = ER3Resolver.from_directory(er3_dir, graph)
    resolver = UnionResolver(
        er3,
        LexicalResolver(config=config),
        top_k=int(union_top_k),
        anchor_k=int(union_anchor_k),
    )
    runner = NEXUSRunner(graph, config, model=model, entity_resolver=resolver)
    identity = {
        "name": "union_lexical_er3",
        "model_dir": er3_dir,
        "config_hash": config.config_hash,
        "entity_ranker_v3_enabled": True,
        "max_entry_nodes": config.max_entry_nodes,
        "union_top_k": int(union_top_k),
        "union_anchor_k": int(union_anchor_k),
        "realizer_backend": realizer_backend,
    }
    return runner, identity


def run_paired_benchmark(
    records: list[dict[str, Any]],
    graph: InMemoryGraphStore,
    *,
    source_sha: str,
    dataset_identity: dict[str, Any],
    predicted_resolver: str = "union",
    er3_dir: str = DEFAULT_ER3_DIR,
    model: Any | None = None,
    union_top_k: int = 12,
    union_anchor_k: int = 8,
    realizer_backend: str = "synth",
    model_name: str = "auto",
) -> dict[str, Any]:
    errors = validate_oracle_records(records)
    if errors:
        raise ValueError("invalid oracle records: " + "; ".join(errors))

    realizer_kwargs = _realizer_overrides(realizer_backend)
    # Oracle arm intentionally ignores ER — gold entities isolate graph/reasoning.
    oracle_config = ProductionNEXUSConfig.lexical_only(**realizer_kwargs)
    oracle_runner = NEXUSRunner(graph, oracle_config, model=model)
    oracle_pipeline = oracle_runner.run_oracle(records, source_sha=source_sha)
    if oracle_pipeline.errors:
        raise RuntimeError("oracle pipeline failed: " + "; ".join(oracle_pipeline.errors))

    predicted_runner, predicted_identity = build_predicted_runner(
        graph,
        predicted_resolver=predicted_resolver,
        er3_dir=er3_dir,
        model=model,
        union_top_k=union_top_k,
        union_anchor_k=union_anchor_k,
        realizer_backend=realizer_backend,
    )
    predicted_pipeline = predicted_runner.run(records, source_sha=source_sha)
    if predicted_pipeline.errors:
        raise RuntimeError("predicted pipeline failed: " + "; ".join(predicted_pipeline.errors))

    oracle_rows = score_mode_rows(records, oracle_pipeline.per_question)
    predicted_rows = score_mode_rows(records, predicted_pipeline.per_question)
    paired = pair_rows(oracle_rows, predicted_rows)
    model_label = type(model).__name__ if model is not None else model_name
    artifact = {
        "schema_version": PAIRED_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "oracle_config_hash": oracle_config.config_hash,
        "config_hash": predicted_identity["config_hash"],
        "predicted_resolver": predicted_identity,
        "realizer_backend": realizer_backend,
        "model_name": model_label,
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--predicted-resolver",
        choices=("union", "er3", "lexical"),
        default="union",
        help="Predicted-arm resolver (default: lexical∪ER3 with question-gated handoff).",
    )
    parser.add_argument("--er3-dir", default=DEFAULT_ER3_DIR)
    parser.add_argument("--union-top-k", type=int, default=12)
    parser.add_argument(
        "--union-anchor-k",
        type=int,
        default=8,
        help="ER3 quality anchors kept before diversifying ungrounded handoff.",
    )
    parser.add_argument(
        "--dummy-model",
        action="store_true",
        help="Force DummyModel for deterministic offline publication.",
    )
    parser.add_argument(
        "--model",
        choices=("dummy", "synth", "auto"),
        default="auto",
        help="Surface model for non-grounded intents (default: auto-detect).",
    )
    parser.add_argument(
        "--realizer-backend",
        choices=_REALIZER_BACKENDS,
        default="l1_acceptance",
        help=(
            "Realization backend for both arms. Default l1_acceptance is the "
            "L1 hybrid (path render + node-fact copy + comparison); use "
            "deterministic_render for pure proof render; CI smoke should pass "
            "synth with --dummy-model."
        ),
    )
    args = parser.parse_args(argv)

    records_path = Path(args.records)
    records = _read_jsonl(records_path)
    if args.limit:
        records = records[: args.limit]
    dataset_identity = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "file_sha256": sha256_file(records_path),
        "record_count": len(records),
        "full_dataset": args.limit is None,
        "sources": {str(records_path.as_posix()): sha256_file(records_path)},
    }
    from benchmarks.run_benchmark import build_benchmark_graph

    graph, _ = build_benchmark_graph()
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    model_name = "dummy" if args.dummy_model else args.model
    model = _select_model(model_name)
    artifact = run_paired_benchmark(
        records,
        graph,
        source_sha=source_sha,
        dataset_identity=dataset_identity,
        predicted_resolver=args.predicted_resolver,
        er3_dir=args.er3_dir,
        model=model,
        union_top_k=args.union_top_k,
        union_anchor_k=args.union_anchor_k,
        realizer_backend=args.realizer_backend,
        model_name=model_name,
    )
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        canonical_json(
            {
                "status": "VALID",
                "questions": len(records),
                "predicted_resolver": args.predicted_resolver,
                "realizer_backend": args.realizer_backend,
                "model_name": type(model).__name__,
                "sha256": digest,
                "output": str(output),
                "predicted_entry_recall_mean": artifact["predicted"]["metrics"]["entry_recall_mean"],
                "predicted_pool_recall_mean": artifact["predicted"]["metrics"]["pool_recall_mean"],
                "predicted_fact_accuracy_mean": artifact["predicted"]["metrics"]["fact_accuracy_mean"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
