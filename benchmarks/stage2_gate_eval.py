"""
Stage 2.3 Gate Evaluation — Realization L1

Runs 30 questions through the NEXUS pipeline with the enhanced
SynthesizingModel and measures:
  - Naturalness score (vs baseline, target +5pt)
  - Relevance (target >= 77%)
  - Hallucination rate (target <= current synth)
  - Accuracy (target >= current -2pp)

Usage:
    python benchmarks/stage2_gate_eval.py --limit 30
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import SynthesizingModel
from nexus.reasoning.verifier import Verifier
from benchmarks.run_benchmark import (
    build_benchmark_graph, load_questions, compute_key_fact_score,
)
from benchmarks.naturalness_eval import score_naturalness
from benchmarks.relevance_judge import RelevanceJudge


def run_gate_evaluation(limit: int = 30) -> dict[str, Any]:
    """Run gate evaluation for Stage 2.3."""
    dataset_path = _project_root / "benchmarks" / "qa-dataset" / "questions.jsonl"
    questions = load_questions(str(dataset_path), limit)
    total = len(questions)

    # Build graph
    graph, graph_provenance = build_benchmark_graph()

    # Models: enhanced SynthesizingModel + verifier
    synth_model = SynthesizingModel()
    verifier = Verifier(hallucination_threshold=0.2)
    relevance_judge = RelevanceJudge()
    import re as _re

    results: list[dict[str, Any]] = []
    naturalness_scores: list[float] = []
    accuracy_scores: list[float] = []
    hallucination_rates: list[float] = []
    relevance_results: list[str] = []

    for i, q in enumerate(questions, 1):
        qtext = q["question"]
        qid = q.get("id", f"q{i:03d}")
        ground_truth = q.get("answer", "")

        try:
            t0 = time.perf_counter()
            result = answer_question(
                question=qtext,
                graph=graph,
                model=synth_model,
                verifier=verifier,
            )
            latency = time.perf_counter() - t0

            answer = result.get("answer", "")
            evidence_pack = result.get("evidence_pack", {})

            # Extract edge types from paths section in evidence
            edge_types: list[str] = []
            paths = result.get("prompt_text", "")
            edge_pattern = _re.compile(r'--\[(\w+)\]')
            for m in edge_pattern.finditer(paths):
                etype = m.group(1)
                if etype not in edge_types:
                    edge_types.append(etype)

            # Extract facts for naturalness scoring
            facts: list[str] = []
            for nf in evidence_pack.get("node_facts", []):
                text = nf.get("text", "") if isinstance(nf, dict) else str(nf)
                if text.strip():
                    facts.append(text.strip())
            for f in evidence_pack.get("facts", []):
                if isinstance(f, str) and f.strip():
                    facts.append(f.strip())

            # Compute metrics
            acc = compute_key_fact_score(answer, ground_truth)
            if isinstance(acc, dict):
                accuracy = acc.get("fuzzy_accuracy")
                if accuracy is None:
                    accuracy = acc.get("exact_accuracy", 0.0)
                if accuracy is None:
                    accuracy = 0.0
            else:
                accuracy = acc if acc is not None else 0.0

            nat = score_naturalness(answer, facts, edge_types if edge_types else None)
            nat_score = nat["total"]

            q_type = q.get("question_type", "factual")
            rel = relevance_judge.judge(qtext, answer, q_type)
            rel_status = rel.get("verdict", "no") if isinstance(rel, dict) else "no"

            verif = result.get("verification")
            if verif is not None:
                hall_rate = verif.hallucination_rate
            else:
                hall_rate = 0.0

            naturalness_scores.append(nat_score)
            accuracy_scores.append(accuracy)
            hallucination_rates.append(hall_rate)
            relevance_results.append(rel_status)

            results.append({
                "question_id": qid,
                "question": qtext,
                "answer": answer,
                "accuracy": accuracy,
                "naturalness": nat_score,
                "relevance": rel_status,
                "hallucination_rate": hall_rate,
                "latency_s": round(latency, 3),
            })

            marker = f"[{i}/{total}]"
            safe_acc = accuracy if accuracy is not None else 0.0
            safe_nat = nat_score if nat_score is not None else 0.0
            print(
                f"  {marker} {qid}: acc={safe_acc:.3f} nat={safe_nat:.1f} "
                f"rel={rel_status} hall={hall_rate:.2f}"
            )

        except Exception as exc:
            print(f"  [{i}/{total}] {qid}: ERROR - {exc}")
            results.append({
                "question_id": qid,
                "question": qtext,
                "error": str(exc),
            })
            continue  # Skip metrics for error-ed questions

    # ── Aggregate metrics ──
    # Filter out None values
    valid_nat = [s for s in naturalness_scores if s is not None]
    valid_acc = [s for s in accuracy_scores if s is not None]
    valid_hall = [s for s in hallucination_rates if s is not None]
    valid_rel = [r for r in relevance_results if r in ("yes", "partial", "no")]

    n_nat = len(valid_nat)
    n_acc = len(valid_acc)
    n_rel = len(valid_rel)

    mean_nat = sum(valid_nat) / n_nat if n_nat > 0 else 0.0
    mean_acc = sum(valid_acc) / n_acc if n_acc > 0 else 0.0
    mean_hall = sum(valid_hall) / len(valid_hall) if valid_hall else 0.0

    yes_count = sum(1 for r in valid_rel if r == "yes")
    partial_count = sum(1 for r in valid_rel if r == "partial")
    no_count = sum(1 for r in valid_rel if r == "no")
    relevance_pct = round((yes_count + partial_count * 0.5) / n_rel * 100, 1) if n_rel > 0 else 0.0

    # ── Baseline comparison ──
    # Baseline values from OLD SynthesizingModel (measured on same 30 questions):
    #   Mean hallucination: 0.4114
    #   Mean accuracy: 0.1694
    #   Naturalness baseline: 35.0 (from test_naturalness_eval.py)
    baseline_nat = 35.0
    baseline_hall = 0.4114
    baseline_acc = 0.1694

    return {
        "n_questions": len(questions),
        "n_scored": n_acc,
        "naturalness": {
            "mean": round(mean_nat, 1),
            "baseline": baseline_nat,
            "delta": round(mean_nat - baseline_nat, 1),
            "gate_threshold": 5.0,
        },
        "accuracy": {
            "mean": round(mean_acc, 4),
            "scores": accuracy_scores,
            "baseline": baseline_acc,
            "gate_threshold_pp": -0.02,
        },
        "relevance": {
            "yes": yes_count,
            "partial": partial_count,
            "no": no_count,
            "pct": relevance_pct,
            "gate_threshold": 77.0,
        },
        "hallucination": {
            "mean": round(mean_hall, 4),
            "rates": hallucination_rates,
            "baseline": baseline_hall,
        },
        "results": results,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2.3 Gate Evaluation")
    parser.add_argument("--limit", type=int, default=30, help="Number of questions")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"Stage 2.3 Gate Evaluation — limit={args.limit}")
    print("=" * 60)

    summary = run_gate_evaluation(limit=args.limit)

    print("\n" + "=" * 60)
    print("GATE RESULTS")
    print("=" * 60)

    nat = summary["naturalness"]
    acc = summary["accuracy"]
    rel = summary["relevance"]
    hall = summary["hallucination"]

    print(f"  Naturalness: {nat['mean']:.1f} (baseline: {nat['baseline']}, "
          f"delta: +{nat['delta']:.1f}, gate: >= +5.0)")
    nat_pass = nat["delta"] >= 5.0
    print(f"    {'-> PASS' if nat_pass else '-> FAIL'}")

    print(f"  Relevance:   {rel['pct']:.1f}% "
          f"(yes={rel['yes']}, partial={rel['partial']}, no={rel['no']}, "
          f"gate: >= 77.0%)")
    rel_pass = rel["pct"] >= 77.0
    print(f"    -> {'PASS' if rel_pass else 'FAIL'}")

    print(f"  Hallucination: {hall['mean']:.4f} "
          f"(baseline: {hall['baseline']:.4f}, gate: <= baseline)")
    hall_pass = hall["mean"] <= hall["baseline"]
    print(f"    -> {'PASS' if hall_pass else 'FAIL'}")

    print(f"  Accuracy:    {acc['mean']:.4f} "
          f"(baseline: {acc['baseline']:.4f}, gate: >= {acc['baseline'] - 0.02:.4f})")
    acc_pass = acc["mean"] >= (acc["baseline"] - 0.02)
    print(f"    -> {'PASS' if acc_pass else 'FAIL'}")
    # Accuracy gate: no worse than -2pp vs current synth
    # Current synth baseline accuracy ~0.4-0.5 based on Stage 1 results
    acc_baseline = 0.40
    acc_pass = acc["mean"] >= (acc_baseline - 0.02)
    print(f"    -> {'PASS' if acc_pass else 'FAIL'}")

    all_pass = nat_pass and rel_pass and hall_pass and acc_pass
    print(f"\n  OVERALL: {'ALL GATES PASSED' if all_pass else 'GATE FAILURE'}")

    if not all_pass:
        print("\n  Writing STAGE2_NEGATIVE.md...")
        negative_path = _project_root / "STAGE2_NEGATIVE.md"
        failures = []
        if not nat_pass:
            failures.append(f"Naturalness delta +{nat['delta']:.1f} < +5.0")
        if not rel_pass:
            failures.append(f"Relevance {rel['pct']:.1f}% < 77.0%")
        if not hall_pass:
            failures.append(f"Hallucination {hall['mean']:.4f} > baseline {hall['baseline']:.4f}")
        if not acc_pass:
            failures.append(f"Accuracy {acc['mean']:.4f} < baseline {acc['baseline']:.4f} - 0.02")

        with open(negative_path, "w", encoding="utf-8") as f:
            f.write("# STAGE2_NEGATIVE.md — Gate Failure Report\n\n")
            f.write(f"**Date**: Automated gate evaluation\n")
            f.write(f"**Stage**: 2.3 — Realization L1 Gate\n\n")
            f.write("## Failed Gates\n\n")
            for failure in failures:
                f.write(f"- {failure}\n")
            f.write("\n## Metrics\n\n")
            f.write(f"- Naturalness: {nat['mean']:.1f} (delta +{nat['delta']:.1f})\n")
            f.write(f"- Relevance: {rel['pct']:.1f}%\n")
            f.write(f"- Hallucination: {hall['mean']:.4f}\n")
            f.write(f"- Accuracy: {acc['mean']:.4f}\n")
            f.write(f"\n## Per-Question Results\n\n")
            for r in summary["results"]:
                f.write(f"- {r.get('question_id', '?')}: "
                        f"nat={r.get('naturalness','?')}, "
                        f"acc={r.get('accuracy','?')}, "
                        f"rel={r.get('relevance','?')}\n")
            f.write(f"\n**Verdict**: STOP. Naturalness improvement insufficient.\n")
        print(f"  Written: {negative_path}")
        sys.exit(1)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _project_root / "benchmarks" / "results" / "stage2_gate_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
