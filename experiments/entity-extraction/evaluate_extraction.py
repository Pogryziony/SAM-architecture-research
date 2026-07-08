"""
Entity extraction evaluation script for NEXUS Phase 1->2 gate.

Loads a labeled set of text snippets with ground-truth entities and relations,
runs the entity_extractor on each snippet, and computes precision/recall/F1.

Usage:
    python experiments/entity-extraction/evaluate_extraction.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from nexus.ingestion.entity_extractor import extract_from_markdown

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELED_SET = _project_root / "experiments" / "entity-extraction" / "labeled_set.jsonl"
RESULTS_FILE = _project_root / "experiments" / "entity-extraction" / "evaluation_results.json"

# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy_match(name_a: str, name_b: str, threshold: float = 0.85) -> bool:
    """Return True if two entity names are similar enough (case-insensitive)."""
    a_lower = name_a.strip().lower()
    b_lower = name_b.strip().lower()
    if a_lower == b_lower:
        return True
    if a_lower in b_lower or b_lower in a_lower:
        return True
    return SequenceMatcher(None, a_lower, b_lower).ratio() >= threshold


def _entity_in_set(entity: dict, candidate_set: list[dict], fuzzy: bool) -> bool:
    """Check if *entity* matches any entry in *candidate_set* by name."""
    name = entity["name"]
    for cand in candidate_set:
        if fuzzy:
            if _fuzzy_match(name, cand["name"]):
                return True
        else:
            if name.strip().lower() == cand["name"].strip().lower():
                return True
    return False


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_entity_metrics(
    predicted: list[dict], ground_truth: list[dict], fuzzy: bool
) -> dict[str, float]:
    """Compute precision, recall, F1 for a single example."""
    tp = sum(1 for gt in ground_truth if _entity_in_set(gt, predicted, fuzzy))
    n_pred = len(predicted)
    n_gt = len(ground_truth)

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "pred_count": n_pred, "gt_count": n_gt}


def compute_relation_metrics(
    predicted_rels: list[dict], ground_truth_rels: list[dict]
) -> dict[str, float]:
    """Compute precision, recall, F1 for relation triples (source, target, type)."""
    def _key(rel: dict) -> tuple[str, str, str]:
        return (
            rel["source"].strip().lower(),
            rel["target"].strip().lower(),
            rel["type"].strip().lower(),
        )

    gt_keys = {_key(r) for r in ground_truth_rels}
    pred_keys = {_key(r) for r in predicted_rels}

    tp = len(gt_keys & pred_keys)
    n_pred = len(pred_keys)
    n_gt = len(gt_keys)

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "pred_count": n_pred, "gt_count": n_gt}


def macro_average(metrics_list: list[dict]) -> dict[str, float]:
    """Average metric values across examples."""
    if not metrics_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        k: sum(m[k] for m in metrics_list) / len(metrics_list)
        for k in ("precision", "recall", "f1")
    }


def micro_average(entity_metrics: list[dict]) -> dict[str, float]:
    """Compute micro-averaged metrics from per-example TP/FP/FN counts."""
    total_tp = sum(m["tp"] for m in entity_metrics)
    total_pred = sum(m["pred_count"] for m in entity_metrics)
    total_gt = sum(m["gt_count"] for m in entity_metrics)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate() -> None:
    # 1. Load labeled data
    labeled = []
    with open(LABELED_SET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labeled.append(json.loads(line))

    print(f"Loaded {len(labeled)} labeled examples from {LABELED_SET}")

    # 2. Run extraction
    results_exact = []  # per-example results with exact matching
    results_fuzzy = []  # per-example results with fuzzy matching

    for entry in labeled:
        text = entry["text"]
        source = entry["source"]
        gt_entities = entry.get("entities", [])
        gt_relations = entry.get("relations", [])

        # Run the extractor
        predicted = extract_from_markdown(text, source)

        # The extractor doesn't extract relations, so predicted_rels = []
        predicted_rels: list[dict] = []

        # Compute metrics
        exact_em = compute_entity_metrics(predicted, gt_entities, fuzzy=False)
        fuzzy_em = compute_entity_metrics(predicted, gt_entities, fuzzy=True)
        rm = compute_relation_metrics(predicted_rels, gt_relations)

        result = {
            "id": entry["id"],
            "source": entry["source"],
            "difficulty": entry.get("difficulty", "unknown"),
            "text": text[:120] + "..." if len(text) > 120 else text,
            "gt_entity_count": len(gt_entities),
            "pred_entity_count": len(predicted),
            "gt_relation_count": len(gt_relations),
            "pred_relation_count": len(predicted_rels),
            "entities_exact": exact_em,
            "entities_fuzzy": fuzzy_em,
            "relations": rm,
            "predicted_entities": predicted,
        }
        results_exact.append(result)
        results_fuzzy.append(result)  # same objects; we use the nested fields

    # 3. Aggregate metrics
    # Exact entity matching
    exact_all = [r["entities_exact"] for r in results_exact]
    exact_macro = macro_average(exact_all)
    exact_micro = micro_average(exact_all)

    # Fuzzy entity matching
    fuzzy_all = [r["entities_fuzzy"] for r in results_fuzzy]
    fuzzy_macro = macro_average(fuzzy_all)
    fuzzy_micro = micro_average(fuzzy_all)

    # Relation metrics
    rel_all = [r["relations"] for r in results_exact]
    rel_macro = macro_average(rel_all)
    rel_micro = micro_average(rel_all)

    # 4. Per-difficulty breakdown
    difficulty_metrics = defaultdict(list)
    for r in results_exact:
        difficulty_metrics[r["difficulty"]].append(r["entities_exact"])

    difficulty_summary = {}
    for diff, metrics_list in sorted(difficulty_metrics.items()):
        difficulty_summary[diff] = {
            "count": len(metrics_list),
            "macro": macro_average(metrics_list),
            "micro": micro_average(metrics_list),
        }

    # 5. Per-document-type breakdown
    def _doc_type(source: str) -> str:
        if "/docs/" in source:
            return "docs"
        if "/experiments/" in source:
            return "experiments"
        return "other"

    doctype_metrics = defaultdict(list)
    for r in results_exact:
        doctype_metrics[_doc_type(r["source"])].append(r["entities_exact"])

    doctype_summary = {}
    for dt, metrics_list in sorted(doctype_metrics.items()):
        doctype_summary[dt] = {
            "count": len(metrics_list),
            "macro": macro_average(metrics_list),
            "micro": micro_average(metrics_list),
        }

    # 6. Build output
    output = {
        "total_examples": len(labeled),
        "entity_extraction": {
            "exact_match": {
                "macro": exact_macro,
                "micro": exact_micro,
            },
            "fuzzy_match": {
                "macro": fuzzy_macro,
                "micro": fuzzy_micro,
            },
        },
        "relation_extraction": {
            "macro": rel_macro,
            "micro": rel_micro,
        },
        "per_difficulty": difficulty_summary,
        "per_document_type": doctype_summary,
        "per_example": results_exact,
    }

    # 7. Gate check
    gate_metric = output["entity_extraction"]["fuzzy_match"]["micro"]["f1"]
    gate_passed = gate_metric >= 0.80
    output["gate_check"] = {
        "metric": "entity_extraction.fuzzy_match.micro.f1",
        "value": round(gate_metric, 4),
        "threshold": 0.80,
        "passed": gate_passed,
    }

    # 8. Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to {RESULTS_FILE}")

    # 9. Print summary table
    print("\n" + "=" * 72)
    print("  ENTITY EXTRACTION EVALUATION -- Phase 1->2 Gate Check")
    print("=" * 72)

    print(f"\n{'Metric':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 62)
    print(f"{'Entity (exact, macro)':<30} {exact_macro['precision']:>10.4f} {exact_macro['recall']:>10.4f} {exact_macro['f1']:>10.4f}")
    print(f"{'Entity (exact, micro)':<30} {exact_micro['precision']:>10.4f} {exact_micro['recall']:>10.4f} {exact_micro['f1']:>10.4f}")
    print(f"{'Entity (fuzzy, macro)':<30} {fuzzy_macro['precision']:>10.4f} {fuzzy_macro['recall']:>10.4f} {fuzzy_macro['f1']:>10.4f}")
    print(f"{'Entity (fuzzy, micro)':<30} {fuzzy_micro['precision']:>10.4f} {fuzzy_micro['recall']:>10.4f} {fuzzy_micro['f1']:>10.4f}")
    print(f"{'Relation (macro)':<30} {rel_macro['precision']:>10.4f} {rel_macro['recall']:>10.4f} {rel_macro['f1']:>10.4f}")
    print(f"{'Relation (micro)':<30} {rel_micro['precision']:>10.4f} {rel_micro['recall']:>10.4f} {rel_micro['f1']:>10.4f}")

    print(f"\n{'Difficulty':<20} {'Count':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 58)
    for diff in sorted(difficulty_summary):
        d = difficulty_summary[diff]
        m = d["macro"]
        print(f"{diff:<20} {d['count']:>6} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print(f"\n{'Doc Type':<20} {'Count':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 58)
    for dt in sorted(doctype_summary):
        d = doctype_summary[dt]
        m = d["macro"]
        print(f"{dt:<20} {d['count']:>6} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # Gate verdict
    print("\n" + "-" * 72)
    print(f"  GATE CHECK: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"  Entity fuzzy micro-F1: {gate_metric:.4f}  (threshold: 0.80)")
    if not gate_passed:
        print(f"  Gap to threshold: {0.80 - gate_metric:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    evaluate()
