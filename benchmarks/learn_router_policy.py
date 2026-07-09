"""
Learn Router Policy — trains a decision table from per-question paired accuracy data.

For each question, runs BOTH arms (synthesizer and LLM), extracts routing signals,
and builds a lookup table: (intent, has_matching_metric, estimated_hops) → best_arm.

Output: nexus/reasoning/router_policy.json — version-controlled decision table.

Usage:
    python benchmarks/learn_router_policy.py --limit 30 --output nexus/reasoning/router_policy.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import (
    SynthesizingModel,
    FallbackModel,
    get_available_model,
)
from nexus.reasoning.verifier import Verifier
from nexus.reasoning.router import Router
from nexus.query.parser import parse_question

from benchmarks.run_benchmark import (
    build_benchmark_graph,
    load_questions,
    compute_key_fact_score,
)

# ── Signal extraction ──


def _extract_signals(
    question: str,
    evidence_pack: dict[str, Any],
    path_count: int,
    intent: str,
) -> dict[str, Any]:
    """Extract routing-relevant signals from evidence and question."""
    signals = evidence_pack.get("confidence_signals", {})

    # has_matching_metric: does evidence have a number matching the question's asked metric?
    numeric_match = signals.get("numeric_match", 0.0)
    has_matching_metric = numeric_match > 0.0

    # estimated_hops: 1, 2, or 3+
    estimated_hops = Router._estimate_hops(question)
    if estimated_hops < 1:
        estimated_hops = 1

    return {
        "intent": intent,
        "has_matching_metric": has_matching_metric,
        "estimated_hops": estimated_hops,
        "path_count": path_count,
        "numeric_match": round(numeric_match, 2),
        "has_key_finding": signals.get("has_key_finding", 0.0),
    }


def _run_arm(
    question_text: str,
    graph: InMemoryGraphStore,
    model,
    verifier: Verifier,
    arm_name: str,
) -> dict[str, Any]:
    """Run one arm (synth or LLM) and return accuracy + evidence signals."""
    t0 = time.perf_counter()
    try:
        result = answer_question(
            question=question_text,
            graph=graph,
            model=model,
            verifier=verifier,
        )
    except Exception as exc:
        return {
            "answer": f"[ERROR] {exc}",
            "accuracy": 0.0,
            "latency_s": time.perf_counter() - t0,
            "error": str(exc),
            "evidence_pack": {},
            "path_count": 0,
            "parsed_intent": "factual_lookup",
        }

    elapsed = time.perf_counter() - t0
    evidence_pack = result.get("evidence_pack", {})
    path_count = result.get("path_count", 0)
    parsed = result.get("parsed_query")
    intent = parsed.intent if parsed else "factual_lookup"

    return {
        "answer": result.get("answer", ""),
        "accuracy": None,  # filled in later
        "latency_s": round(elapsed, 4),
        "error": None,
        "evidence_pack": evidence_pack,
        "path_count": path_count,
        "parsed_intent": intent,
    }


def learn_policy(
    limit: int = 30,
    output_path: str = "nexus/reasoning/router_policy.json",
) -> None:
    """Run paired arms on each question and learn the decision table."""
    dataset_path = _project_root / "benchmarks" / "qa-dataset" / "questions.jsonl"
    if not dataset_path.exists():
        print(f"Error: QA dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading questions from: {dataset_path}")
    questions = load_questions(str(dataset_path), limit)
    total = len(questions)
    print(f"Loaded {total} questions (limit={limit})")

    # Build graph
    print("\nBuilding graph...")
    graph, graph_provenance = build_benchmark_graph()
    print(f"Graph ready: {graph_provenance['node_count']} nodes, "
          f"{graph_provenance['edge_count']} edges")

    # Initialize models
    print("\nInitializing models...")
    primary_model = get_available_model()
    llm_model = FallbackModel(primary_model)
    synth_model = SynthesizingModel()
    verifier = Verifier(hallucination_threshold=0.2)

    print(f"\nRunning paired-arm training on {total} questions...\n")

    # ── Collect per-question data ──
    rows: list[dict[str, Any]] = []

    for i, q in enumerate(questions, 1):
        qtext = q["question"]
        qid = q.get("id", f"q{str(i).zfill(3)}")
        ground_truth = q.get("answer", "")
        marker = f"[{i}/{total}]"

        print(f"{marker} {qid}: {qtext[:70]}...")

        # Run synthesizer arm
        synth_result = _run_arm(qtext, graph, synth_model, verifier, "synth")
        synth_accuracy = compute_key_fact_score(synth_result["answer"], ground_truth)
        synth_acc = synth_accuracy.get("fuzzy_accuracy", 0.0) if isinstance(synth_accuracy, dict) else 0.0
        synth_result["accuracy"] = round(synth_acc, 4) if synth_acc is not None else 0.0

        # Run LLM arm
        llm_result = _run_arm(qtext, graph, llm_model, verifier, "llm")
        llm_accuracy = compute_key_fact_score(llm_result["answer"], ground_truth)
        llm_acc = llm_accuracy.get("fuzzy_accuracy", 0.0) if isinstance(llm_accuracy, dict) else 0.0
        llm_result["accuracy"] = round(llm_acc, 4) if llm_acc is not None else 0.0

        # Extract signals from evidence (use synth result, same traversal)
        signals = _extract_signals(
            qtext,
            synth_result["evidence_pack"],
            synth_result["path_count"],
            synth_result["parsed_intent"],
        )

        # Determine best arm
        synth_a = synth_result["accuracy"]
        llm_a = llm_result["accuracy"]
        if synth_a >= llm_a:
            best_arm = "synthesizer"
        else:
            best_arm = "llm"

        print(f"  synth={synth_a:.2%}  llm={llm_a:.2%}  best={best_arm}  "
              f"intent={signals['intent']}  metric={signals['has_matching_metric']}  "
              f"hops={signals['estimated_hops']}")

        rows.append({
            "question_id": qid,
            "question": qtext,
            "ground_truth": ground_truth,
            "signals": signals,
            "synthesizer_accurate": synth_a > 0.0,
            "synthesizer_accuracy": synth_a,
            "llm_accurate": llm_a > 0.0,
            "llm_accuracy": llm_a,
            "best_arm": best_arm,
        })

    # ── Build decision table ──
    # Key: (intent, has_matching_metric, estimated_hops)
    # Value: aggregated stats + best_arm
    decision_table: dict[str, dict[str, Any]] = {}

    for row in rows:
        s = row["signals"]
        key = f"{s['intent']}|{int(s['has_matching_metric'])}|{s['estimated_hops']}"

        if key not in decision_table:
            decision_table[key] = {
                "intent": s["intent"],
                "has_matching_metric": s["has_matching_metric"],
                "estimated_hops": s["estimated_hops"],
                "count": 0,
                "synth_wins": 0,
                "llm_wins": 0,
                "synth_accuracy": 0.0,
                "llm_accuracy": 0.0,
            }

        entry = decision_table[key]
        entry["count"] += 1
        entry["synth_accuracy"] += row["synthesizer_accuracy"]
        entry["llm_accuracy"] += row["llm_accuracy"]
        if row["best_arm"] == "synthesizer":
            entry["synth_wins"] += 1
        else:
            entry["llm_wins"] += 1

    # Normalize accuracies and pick best_arm by mean accuracy (not win count)
    # Mean accuracy is what matters for expected routing quality.
    for key, entry in decision_table.items():
        n = entry["count"]
        entry["synth_accuracy"] = round(entry["synth_accuracy"] / n, 4) if n > 0 else 0.0
        entry["llm_accuracy"] = round(entry["llm_accuracy"] / n, 4) if n > 0 else 0.0
        # Use mean accuracy to determine best arm (ties go to synthesizer for cost)
        if entry["synth_accuracy"] >= entry["llm_accuracy"]:
            entry["best_arm"] = "synthesizer"
        else:
            entry["best_arm"] = "llm"

    # ── Compute aggregate metrics ──
    router_accuracy = 0.0
    oracle_accuracy = 0.0
    for row in rows:
        # Router accuracy: if we pick the best_arm from the table
        s = row["signals"]
        key = f"{s['intent']}|{int(s['has_matching_metric'])}|{s['estimated_hops']}"
        entry = decision_table.get(key)
        if entry and entry["best_arm"] == "synthesizer":
            router_accuracy += row["synthesizer_accuracy"]
        elif entry:
            router_accuracy += row["llm_accuracy"]
        else:
            # Fall back to best individual
            router_accuracy += max(row["synthesizer_accuracy"], row["llm_accuracy"])

        # Oracle accuracy: always pick the best arm
        oracle_accuracy += max(row["synthesizer_accuracy"], row["llm_accuracy"])

    router_accuracy /= len(rows) if rows else 1
    oracle_accuracy /= len(rows) if rows else 1
    routing_quality = router_accuracy / oracle_accuracy if oracle_accuracy > 0 else 0.0

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("  LEARNED ROUTER POLICY")
    print("=" * 70)
    print(f"  Questions trained: {len(rows)}")
    print(f"  Table entries: {len(decision_table)}")
    print(f"  Router accuracy: {router_accuracy:.2%}")
    print(f"  Oracle accuracy: {oracle_accuracy:.2%}")
    print(f"  Routing quality: {routing_quality:.2%}")
    print()

    # Print the table
    print("  Decision table:")
    print(f"  {'Intent':<22} {'Metric':>7} {'Hops':>5} {'Best':>14} {'S-Acc':>8} {'L-Acc':>8} {'N':>4}")
    print(f"  {'-'*22} {'-'*7} {'-'*5} {'-'*14} {'-'*8} {'-'*8} {'-'*4}")
    for key in sorted(decision_table.keys()):
        e = decision_table[key]
        print(f"  {e['intent']:<22} {str(e['has_matching_metric']):>7} "
              f"{e['estimated_hops']:>5} {e['best_arm']:>14} "
              f"{e['synth_accuracy']:>7.1%} {e['llm_accuracy']:>7.1%} {e['count']:>4}")

    print()

    # ── Write decision table ──
    output_path_abs = _project_root / output_path
    output_path_abs.parent.mkdir(parents=True, exist_ok=True)

    policy = {
        "version": "1.0",
        "phase": 5,
        "description": "Learned router decision table mapping (intent, has_matching_metric, estimated_hops) → best_arm",
        "training_config": {
            "limit": limit,
            "questions": len(rows),
            "table_entries": len(decision_table),
        },
        "metrics": {
            "router_accuracy": round(router_accuracy, 4),
            "oracle_accuracy": round(oracle_accuracy, 4),
            "routing_quality": round(routing_quality, 4),
        },
        "decision_table": decision_table,
        "per_question_data": rows,
    }

    with open(output_path_abs, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)

    print(f"Policy written to: {output_path_abs}")
    print(f"\nDone. Router accuracy: {router_accuracy:.2%} | "
          f"Oracle: {oracle_accuracy:.2%} | "
          f"Quality: {routing_quality:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn router decision table from per-question paired accuracy data"
    )
    parser.add_argument(
        "--limit", type=int, default=30,
        help="Number of questions to train on (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, default="nexus/reasoning/router_policy.json",
        help="Output path for the decision table (default: nexus/reasoning/router_policy.json)"
    )
    args = parser.parse_args()
    learn_policy(limit=args.limit, output_path=args.output)
