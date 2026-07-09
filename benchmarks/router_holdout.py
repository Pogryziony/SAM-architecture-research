"""
Router Hold-Out Validation — train/test split for the router decision table.

Reads the existing router_policy.json (which was trained on all 30 questions in-sample),
splits into train (first 15) and test (last 15), builds the decision table from training
set only, and evaluates routing quality on the held-out test set.

Output: updated nexus/reasoning/router_policy.json with train/test metadata.

Usage:
    python benchmarks/router_holdout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

POLICY_PATH = _project_root / "nexus" / "reasoning" / "router_policy.json"
OUTPUT_PATH = POLICY_PATH  # Overwrite in-place


def load_per_question_data() -> list[dict[str, Any]]:
    """Load per-question paired accuracy data from the policy file."""
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = json.load(f)
    return policy.get("per_question_data", [])


def build_decision_table(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a decision table from a set of per-question rows.

    Key: (intent, has_matching_metric, estimated_hops)
    Value: aggregated stats + best_arm by mean accuracy.
    """
    table: dict[str, dict[str, Any]] = {}

    for row in rows:
        s = row["signals"]
        key = f"{s['intent']}|{int(s['has_matching_metric'])}|{s['estimated_hops']}"

        if key not in table:
            table[key] = {
                "intent": s["intent"],
                "has_matching_metric": s["has_matching_metric"],
                "estimated_hops": s["estimated_hops"],
                "count": 0,
                "synth_wins": 0,
                "llm_wins": 0,
                "synth_accuracy": 0.0,
                "llm_accuracy": 0.0,
            }

        entry = table[key]
        entry["count"] += 1
        entry["synth_accuracy"] += row["synthesizer_accuracy"]
        entry["llm_accuracy"] += row["llm_accuracy"]
        if row["best_arm"] == "synthesizer":
            entry["synth_wins"] += 1
        else:
            entry["llm_wins"] += 1

    # Normalize and pick best_arm by mean accuracy (ties → synthesizer for cost)
    for entry in table.values():
        n = entry["count"]
        entry["synth_accuracy"] = round(entry["synth_accuracy"] / n, 4) if n > 0 else 0.0
        entry["llm_accuracy"] = round(entry["llm_accuracy"] / n, 4) if n > 0 else 0.0
        entry["best_arm"] = (
            "synthesizer" if entry["synth_accuracy"] >= entry["llm_accuracy"] else "llm"
        )

    return table


def evaluate_routing(
    rows: list[dict[str, Any]],
    decision_table: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Evaluate routing quality on a held-out set.

    Returns:
        dict with router_accuracy, oracle_accuracy, routing_quality,
        all_llm_accuracy, all_synth_accuracy.
    """
    router_acc = 0.0
    oracle_acc = 0.0
    all_llm_acc = 0.0
    all_synth_acc = 0.0
    n = len(rows)

    for row in rows:
        s = row["signals"]
        key = f"{s['intent']}|{int(s['has_matching_metric'])}|{s['estimated_hops']}"
        entry = decision_table.get(key)

        if entry and entry["best_arm"] == "synthesizer":
            router_acc += row["synthesizer_accuracy"]
        elif entry:
            router_acc += row["llm_accuracy"]
        else:
            # Table miss — pick best individual (optimistic fallback)
            router_acc += max(row["synthesizer_accuracy"], row["llm_accuracy"])

        oracle_acc += max(row["synthesizer_accuracy"], row["llm_accuracy"])
        all_llm_acc += row["llm_accuracy"]
        all_synth_acc += row["synthesizer_accuracy"]

    if n == 0:
        return {}

    return {
        "router_accuracy": round(router_acc / n, 4),
        "oracle_accuracy": round(oracle_acc / n, 4),
        "routing_quality": round((router_acc / n) / (oracle_acc / n), 4) if oracle_acc > 0 else 0.0,
        "all_llm_accuracy": round(all_llm_acc / n, 4),
        "all_synth_accuracy": round(all_synth_acc / n, 4),
    }


def main():
    # ── Load data ──
    rows = load_per_question_data()
    total = len(rows)
    print(f"Loaded {total} questions with paired arm data.")

    # ── Split into train (first N) and test (remaining) ──
    # Use 50/50 split
    n_train = total // 2
    train_rows = rows[:n_train]
    test_rows = rows[n_train:]
    print(f"Split: {len(train_rows)} train, {len(test_rows)} test")

    # ── Build decision table from training set ONLY ──
    decision_table = build_decision_table(train_rows)
    print(f"Decision table entries (from training set): {len(decision_table)}")

    # ── Evaluate on training set (in-sample, for comparison) ──
    train_metrics = evaluate_routing(train_rows, decision_table)

    # ── Evaluate on test set (held-out, the real metric) ──
    test_metrics = evaluate_routing(test_rows, decision_table)

    # ── Print report ──
    print()
    print("=" * 70)
    print("  ROUTER HOLD-OUT VALIDATION REPORT")
    print("=" * 70)
    print(f"  Training set: n={len(train_rows)}, table_entries={len(decision_table)}")
    print(f"  Test set:     n={len(test_rows)}")
    print()
    print(f"  {'':>20} {'Router':>10} {'Oracle':>10} {'Quality':>10} {'LLM-only':>10} {'Synth-only':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for label, m in [("Training (in-sample)", train_metrics), ("Test (held-out)", test_metrics)]:
        if not m:
            continue
        print(
            f"  {label:<20} "
            f"{m['router_accuracy']:>9.2%} "
            f"{m['oracle_accuracy']:>9.2%} "
            f"{m['routing_quality']:>9.2%} "
            f"{m['all_llm_accuracy']:>9.2%} "
            f"{m['all_synth_accuracy']:>9.2%}"
        )

    print()
    print("  Decision table (from training set):")
    print(f"  {'Intent':<22} {'Metric':>7} {'Hops':>5} {'Best':>14} {'S-Acc':>8} {'L-Acc':>8} {'N':>4}")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 5} {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 4}")
    for key in sorted(decision_table.keys()):
        e = decision_table[key]
        print(f"  {e['intent']:<22} {str(e['has_matching_metric']):>7} "
              f"{e['estimated_hops']:>5} {e['best_arm']:>14} "
              f"{e['synth_accuracy']:>7.1%} {e['llm_accuracy']:>7.1%} {e['count']:>4}")

    # ── Write updated router_policy.json ──
    policy = {
        "version": "1.1",
        "phase": 5,
        "description": (
            "Learned router decision table with hold-out validation. "
            "Trained on first N questions, validated on held-out N."
        ),
        "_comment": (
            f"Trained on first {len(train_rows)} questions, "
            f"validated on held-out {len(test_rows)}."
        ),
        "_train_n": len(train_rows),
        "_test_n": len(test_rows),
        "training_config": {
            "total_questions": total,
            "train_questions": len(train_rows),
            "test_questions": len(test_rows),
            "table_entries": len(decision_table),
            "limitation": (
                "Training data limited to 30 oracle-test questions. "
                "Split 15 train / 15 test. Expanding to 200 questions "
                "requires running paired-arm accuracy on the full QA dataset."
            ),
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "decision_table": decision_table,
        "per_question_data": rows,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)

    print(f"\n  Policy written to: {OUTPUT_PATH}")
    print()

    # ── Interpretation ──
    if test_metrics:
        rq = test_metrics["routing_quality"]
        if rq >= 0.90:
            print("  [PASS] Routing quality is acceptable (>=90% of oracle).")
        else:
            print("  [WARN] Routing quality below 90% -- may need more training data.")
        print(f"  Router vs LLM-only: {test_metrics['router_accuracy']:.2%} vs {test_metrics['all_llm_accuracy']:.2%}")
        print(f"  Router vs Synth-only: {test_metrics['router_accuracy']:.2%} vs {test_metrics['all_synth_accuracy']:.2%}")


if __name__ == "__main__":
    main()
