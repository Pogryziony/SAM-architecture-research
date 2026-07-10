"""Single-read, frozen Stage 1C final evaluator.

The evaluator deliberately accepts the already-loaded frozen test records.  The
CLI is the only place that opens ``test.jsonl`` and it opens it once.  Validation
artifacts and the frozen feature-logistic selection are read before that open.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.run_benchmark import build_benchmark_graph
from nexus.utils.canonical_labels import canonicalize_intent
from stack.encoder.c2_c3 import _node_features, _dot
from stack.encoder.intent_rules import get_rule_classifier
from stack.encoder.trivial_baseline import candidate_pool, rank_candidates

K = 10
FROZEN_SPLIT = "stack/encoder/data/test.jsonl"
FROZEN_SPLIT_ID = "stage1.0:test.jsonl:34278d5"
CALIBRATION_ARTIFACT = "benchmarks/results/baseline_val_20260710T172932Z.json"
SELECTION_ARTIFACT = "benchmarks/results/stage1c_full_selection_log.json"
MODEL_CONFIG = "models/encoder/stage1c/config.json"
MODEL_WEIGHTS = "models/encoder/stage1c/weights.json"
PARAPHRASE_SPLIT = "benchmarks/qa-dataset/paraphrase_30.jsonl"


def _check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= K:
        raise ValueError(f"k must be an integer in [1, {K}]")
    return k


def rank_feature_logistic(
    question: str, candidates: Iterable[str], graph: Any,
    ranker: Mapping[str, Any], k: int = K,
) -> list[str]:
    """Rank and cap candidates in the ranking path; K>10 is never permitted."""
    _check_k(k)
    weights = ranker.get("weights")
    if ranker.get("kind") != "feature_logistic" or not isinstance(weights, list):
        raise ValueError("final evaluator requires the frozen feature_logistic ranker")
    unique = list(dict.fromkeys(str(node) for node in candidates))
    scored = [(_dot(weights, [1.0, *_node_features(question, node, graph)]), node) for node in unique]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [node for _score, node in scored[:k]]


def validate_final_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema/provenance errors; this is intentionally independent of test data."""
    required = {"meta", "winner", "baseline", "metrics", "gates", "decision", "question_details"}
    errors = [f"missing top-level field: {key}" for key in sorted(required - set(artifact))]
    meta = artifact.get("meta", {})
    metrics = artifact.get("metrics", {})
    gates = artifact.get("gates", {})
    if not isinstance(meta, dict) or meta.get("question_count", 0) <= 0:
        errors.append("invalid meta question_count")
    if meta.get("validated_ids_match") is not True:
        errors.append("frozen split IDs were not validated")
    if artifact.get("winner") != "feature_logistic":
        errors.append("winner is not feature_logistic")
    if meta.get("k") != K:
        errors.append("K is not 10")
    for key in ("recall@1", "recall@5", "recall@10", "precision@10", "intent_accuracy", "resolution_rate", "latency_p50_ms", "rss_delta_mb"):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"missing numeric metric: {key}")
    required_gates = {"primary_recall", "control_validation", "intent_accuracy", "paraphrase_drop", "resolution_rate", "latency_p50", "rss_delta"}
    if not required_gates <= set(gates):
        errors.append("missing required gate")
    if len(artifact.get("question_details", [])) != meta.get("question_count"):
        errors.append("question detail count mismatch")
    if artifact.get("decision") not in {"HONEST PASS", "HONEST FAIL"}:
        errors.append("invalid mechanical decision")
    derived = all(isinstance(value, dict) and value.get("passed") is True for value in gates.values())
    if (artifact.get("decision") == "HONEST PASS") != derived:
        errors.append("decision is not mechanically derived from gates")
    return errors


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


def _metrics(rows: Sequence[Mapping[str, Any]], graph: Any, ranker: Mapping[str, Any], k: int = K) -> tuple[dict[str, float], list[dict[str, Any]]]:
    _check_k(k)
    hits = {1: 0, 5: 0, 10: 0}
    predicted = {1: 0, 5: 0, 10: 0}
    gold_total = sum(len(set(row["entities"])) for row in rows)
    resolved = 0
    intent_correct = 0
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    classifier = get_rule_classifier()
    for row in rows:
        started = time.perf_counter()
        pool = candidate_pool(str(row["question"]), graph)
        ids = [str(item["node_id"]) for item in pool]
        ranked = rank_feature_logistic(str(row["question"]), ids, graph, ranker, k)
        gold = set(str(x) for x in row["entities"])
        if ranked:
            resolved += 1
        for cut in (1, 5, 10):
            selected = set(ranked[:cut])
            hits[cut] += len(selected & gold)
            predicted[cut] += len(selected)
        gt_intent = canonicalize_intent(str(row.get("intent", row.get("question_type", ""))))
        pred_intent = canonicalize_intent(classifier.classify(str(row["question"])) or "factual_lookup")
        intent_correct += int(gt_intent == pred_intent)
        latencies.append((time.perf_counter() - started) * 1000)
        details.append({
            "id": row["id"], "question": row["question"], "gold_entities": sorted(gold),
            "ranked_entities_at_k": ranked, "candidate_count": len(ids),
            "hits_at_1": len(set(ranked[:1]) & gold), "hits_at_5": len(set(ranked[:5]) & gold),
            "hits_at_10": len(set(ranked[:10]) & gold), "gt_intent": gt_intent,
            "pred_intent": pred_intent, "intent_match": gt_intent == pred_intent,
            "latency_ms": latencies[-1],
        })
    count = len(rows)
    metrics = {
        "recall@1": hits[1] / gold_total if gold_total else 0.0,
        "recall@5": hits[5] / gold_total if gold_total else 0.0,
        "recall@10": hits[10] / gold_total if gold_total else 0.0,
        "precision@10": hits[10] / predicted[10] if predicted[10] else 0.0,
        "resolution_rate": resolved / count if count else 0.0,
        "intent_accuracy": intent_correct / count if count else 0.0,
        "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "latency_p90_ms": sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * .9))] if latencies else 0.0,
        "gold_entities": gold_total, "questions": count,
    }
    return metrics, details


def _baseline_metrics(rows: Sequence[Mapping[str, Any]], graph: Any) -> dict[str, float]:
    # Baseline uses the same candidate pool and the same in-path K cap.
    hits = {1: 0, 5: 0, 10: 0}; predicted10 = 0; resolved = 0
    gold_total = sum(len(set(row["entities"])) for row in rows)
    for row in rows:
        ranked = rank_candidates(candidate_pool(str(row["question"]), graph), graph, K)
        gold = set(str(x) for x in row["entities"])
        resolved += int(bool(ranked)); predicted10 += len(ranked)
        for cut in hits:
            hits[cut] += len(set(ranked[:cut]) & gold)
    return {f"recall@{cut}": hits[cut] / gold_total if gold_total else 0.0 for cut in hits} | {
        "precision@10": hits[10] / predicted10 if predicted10 else 0.0,
        "resolution_rate": resolved / len(rows) if rows else 0.0,
        "questions": len(rows), "gold_entities": gold_total,
    }


def _paraphrase_metrics(paraphrases: Sequence[Mapping[str, Any]], graph: Any, ranker: Mapping[str, Any]) -> dict[str, float]:
    original_hits = para_hits = original_gold = para_gold = 0
    pairs = 0
    for row in paraphrases:
        original = rank_feature_logistic(str(row["original_question"]), [x["node_id"] for x in candidate_pool(str(row["original_question"]), graph)], graph, ranker, K)
        para = rank_feature_logistic(str(row["paraphrase"]), [x["node_id"] for x in candidate_pool(str(row["paraphrase"]), graph)], graph, ranker, K)
        gold = set(str(x) for x in row["gt_entities"])
        original_hits += len(set(original) & gold); para_hits += len(set(para) & gold); original_gold += len(gold); para_gold += len(gold); pairs += 1
    original_recall = original_hits / original_gold if original_gold else 0.0
    para_recall = para_hits / para_gold if para_gold else 0.0
    return {"pairs": pairs, "original_recall@10": original_recall, "paraphrase_recall@10": para_recall, "drop_pp": (original_recall - para_recall) * 100}


def evaluate_once(root: str | Path = ".", output: str | Path | None = None) -> Path:
    root = Path(root)
    selection = _load_json(root / SELECTION_ARTIFACT)
    config = _load_json(root / MODEL_CONFIG)
    weights = _load_json(root / MODEL_WEIGHTS)
    validation_baseline = _load_json(root / CALIBRATION_ARTIFACT)
    if config.get("winner") != "feature_logistic" or selection.get("selection", {}).get("winner") != "feature_logistic":
        raise ValueError("selected validation winner is not feature_logistic")
    if selection.get("test_split_read") is not False or config.get("validation_only") is not True:
        raise ValueError("selection artifacts are not validation-only")
    ranker = weights
    graph, graph_meta = build_benchmark_graph()
    # The sole test split read. Every downstream test metric consumes this list.
    test_path = root / FROZEN_SPLIT
    test_bytes = test_path.read_bytes()
    rows = [json.loads(line) for line in test_bytes.splitlines() if line.strip()]
    frozen_ids = [str(row["id"]) for row in rows]
    baseline = _baseline_metrics(rows, graph)
    winner, details = _metrics(rows, graph, ranker, K)
    paraphrases = _load_jsonl(root / PARAPHRASE_SPLIT)
    paraphrase = _paraphrase_metrics(paraphrases, graph, ranker)
    rss = _rss_mb()
    selection_metrics = selection["selection"]
    winner_val = float(selection_metrics["winner_recall@10"])
    baseline_val = float(validation_baseline["metrics"]["recall@10"])
    gates = {
        "primary_recall": {"passed": winner["recall@10"] >= 0.65, "value": winner["recall@10"], "threshold": 0.65},
        "control_validation": {"passed": baseline_val <= winner_val - 0.15, "baseline_validation_recall@10": baseline_val, "winner_validation_recall@10": winner_val, "required_gap_pp": 15.0},
        "intent_accuracy": {"passed": winner["intent_accuracy"] >= 0.85, "value": winner["intent_accuracy"], "threshold": 0.85},
        "paraphrase_drop": {"passed": abs(paraphrase["drop_pp"]) < 10, "value_pp": paraphrase["drop_pp"], "threshold_pp": 10},
        "resolution_rate": {"passed": winner["resolution_rate"] >= baseline.get("resolution_rate", 0.0), "value": winner["resolution_rate"], "baseline": baseline.get("resolution_rate", 0.0)},
        "latency_p50": {"passed": winner["latency_p50_ms"] <= 50, "value_ms": winner["latency_p50_ms"], "threshold_ms": 50},
        "rss_delta": {"passed": rss <= 150, "value_mb": rss, "threshold_mb": 150},
    }
    all_pass = all(item["passed"] for item in gates.values())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "artifact": "stage1c_final", "winner": "feature_logistic", "all_pass": all_pass,
        "decision": "HONEST PASS" if all_pass else "HONEST FAIL",
        "meta": {"phase": "Stage 1C C4", "timestamp_utc": timestamp, "k": K, "question_count": len(rows),
                 "frozen_split": FROZEN_SPLIT, "frozen_split_identifier": FROZEN_SPLIT_ID,
                 "frozen_split_sha256": hashlib.sha256(test_bytes).hexdigest(),
                 "evaluation_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                 "validated_ids_match": len(rows) == 225 and len(frozen_ids) == len(set(frozen_ids)),
                 "validation_only_selection": True, "graph": graph_meta,
                 "calibration_artifact": CALIBRATION_ARTIFACT, "selection_artifact": SELECTION_ARTIFACT,
                 "model_config": MODEL_CONFIG, "model_weights": MODEL_WEIGHTS},
        "calibration_selection": {"validation_baseline": validation_baseline, "selection": selection_metrics},
        "winner": "feature_logistic", "metrics": {**winner, "rss_delta_mb": rss},
        "baseline": {"test": baseline, "validation": validation_baseline["metrics"]},
        "paraphrase": paraphrase, "gates": gates, "question_details": details,
    }
    errors = validate_final_artifact(artifact)
    if errors:
        raise ValueError("final artifact schema invalid: " + "; ".join(errors))
    destination = root / (output or f"benchmarks/results/stage1c_final_{timestamp}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite historical artifact: {destination}")
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-read Stage 1C final evaluator")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(evaluate_once(args.root))


if __name__ == "__main__":
    main()
