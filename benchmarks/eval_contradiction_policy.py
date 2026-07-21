"""Stage 5 contradiction F1 + readiness calibration campaign.

Prereg: EXPERIMENT_CONTRADICTION_POLICY_V1.md (thresholds opened for development
gold ``contradiction_gold_v1.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.reasoning.conflict_policy import (
    Conflict,
    apply_conflict_policy,
    classify_graph_conflicts,
)

PREREGISTRATION_ID = "contradiction-policy-v1"
DEFAULT_GOLD = (
    _project_root / "benchmarks" / "qa-dataset" / "contradiction_gold_v1.jsonl"
)
CLASS_F1_MIN = 0.90
POLICY_ACCURACY_MIN = 0.90
UNCONDITIONAL_LEAK_MAX = 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _predict_class(record: dict[str, Any]) -> str:
    cls = str(record.get("conflict_class") or "none")
    if cls == "none":
        return "none"
    conflicts = classify_graph_conflicts(
        contradicts=[
            (record["source"], record["relation"], record["target"])
        ]
        if cls == "contradiction"
        else (),
        replaces=[
            (record["source"], record["relation"], record["target"])
        ]
        if cls == "supersession"
        else (),
        validity_mismatches=[
            (record["source"], record["relation"], record["target"])
        ]
        if cls == "validity_mismatch"
        else (),
        source_disagreements=[
            (record["source"], record["relation"], record["target"])
        ]
        if cls == "source_disagreement"
        else (),
    )
    if not conflicts:
        return "none"
    return conflicts[0].conflict_class.value


def _apply_record(record: dict[str, Any]) -> dict[str, Any]:
    cls = str(record.get("conflict_class") or "none")
    conflicts: list[Conflict] = []
    if cls != "none":
        raw = classify_graph_conflicts(
            contradicts=[
                (record["source"], record["relation"], record["target"])
            ]
            if cls == "contradiction"
            else (),
            replaces=[
                (record["source"], record["relation"], record["target"])
            ]
            if cls == "supersession"
            else (),
            validity_mismatches=[
                (record["source"], record["relation"], record["target"])
            ]
            if cls == "validity_mismatch"
            else (),
            source_disagreements=[
                (record["source"], record["relation"], record["target"])
            ]
            if cls == "source_disagreement"
            else (),
        )
        resolved = bool(record.get("resolved"))
        conflicts = [
            Conflict(
                conflict_class=item.conflict_class,
                source=item.source,
                relation=item.relation,
                target=item.target,
                resolved=resolved,
                note=item.note,
            )
            for item in raw
        ]
    decision = apply_conflict_policy(
        conflicts,
        base_recommendation=str(record.get("base_recommendation") or "answer"),
    )
    return {
        "id": record["id"],
        "gold_class": cls,
        "pred_class": _predict_class(record),
        "expected_recommendation": record.get("expected_recommendation"),
        "pred_recommendation": decision.recommendation,
        "allow_unconditional_answer": decision.allow_unconditional_answer,
        "should_allow_unconditional_answer": bool(
            record.get("should_allow_unconditional_answer")
        ),
    }


def _macro_f1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({row["gold_class"] for row in rows} | {row["pred_class"] for row in rows})
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for row in rows if row["gold_class"] == label and row["pred_class"] == label)
        fp = sum(1 for row in rows if row["gold_class"] != label and row["pred_class"] == label)
        fn = sum(1 for row in rows if row["gold_class"] == label and row["pred_class"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": float(sum(1 for row in rows if row["gold_class"] == label)),
        }
        if per_class[label]["support"] > 0:
            f1_values.append(f1)
    return {
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "per_class": per_class,
    }


def _brier_ece(records: list[dict[str, Any]], *, bins: int = 5) -> dict[str, Any]:
    scored = [
        item
        for item in records
        if "readiness_score" in item and "should_abstain" in item
    ]
    if not scored:
        return {"n": 0, "brier": None, "ece": None}
    probs = []
    outcomes = []
    for item in scored:
        # Map readiness to P(answer) = readiness; outcome 1 if not abstain.
        p = max(0.0, min(1.0, float(item["readiness_score"])))
        y = 0.0 if bool(item["should_abstain"]) else 1.0
        probs.append(p)
        outcomes.append(y)
    brier = sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)
    # Expected calibration error over equal-width bins.
    bucket_totals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for p, y in zip(probs, outcomes):
        idx = min(bins - 1, int(p * bins))
        bucket_totals[idx].append((p, y))
    ece = 0.0
    for items in bucket_totals.values():
        conf = sum(p for p, _ in items) / len(items)
        acc = sum(y for _, y in items) / len(items)
        ece += (len(items) / len(probs)) * abs(conf - acc)
    return {
        "n": len(probs),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_apply_record(record) for record in records]
    class_metrics = _macro_f1(rows)
    policy_correct = sum(
        1 for row in rows if row["pred_recommendation"] == row["expected_recommendation"]
    )
    policy_accuracy = policy_correct / len(rows) if rows else 0.0
    unconditional_leaks = sum(
        1
        for row in rows
        if row["allow_unconditional_answer"]
        and not row["should_allow_unconditional_answer"]
    )
    calibration = _brier_ece(records)
    errors: list[str] = []
    if class_metrics["macro_f1"] < CLASS_F1_MIN:
        errors.append(
            f"class macro_f1 {class_metrics['macro_f1']} < {CLASS_F1_MIN}"
        )
    if policy_accuracy < POLICY_ACCURACY_MIN:
        errors.append(f"policy_accuracy {policy_accuracy:.4f} < {POLICY_ACCURACY_MIN}")
    if unconditional_leaks > UNCONDITIONAL_LEAK_MAX:
        errors.append(
            f"unconditional_leaks {unconditional_leaks} > {UNCONDITIONAL_LEAK_MAX}"
        )
    return {
        "status": "PASS" if not errors else "FAIL",
        "preregistration_id": PREREGISTRATION_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "classification": class_metrics,
        "policy": {
            "accuracy": round(policy_accuracy, 4),
            "unconditional_leaks": unconditional_leaks,
        },
        "calibration": calibration,
        "errors": errors,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    records = _read_jsonl(args.gold)
    report = evaluate(records)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    summary = {
        "status": report["status"],
        "macro_f1": report["classification"]["macro_f1"],
        "policy_accuracy": report["policy"]["accuracy"],
        "brier": report["calibration"].get("brier"),
        "ece": report["calibration"].get("ece"),
        "errors": report["errors"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
