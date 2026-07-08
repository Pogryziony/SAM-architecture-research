"""
NEXUS QA Benchmark Harness (P3 -- Phase 3/4).

Compares the full NEXUS pipeline against an LLM-only baseline on the QA dataset.
Key metrics: hallucination rate, answer rate, latency, verification pass rate.

Usage:
    python benchmarks/run_benchmark.py --limit 50
    python benchmarks/run_benchmark.py --limit 100 --output benchmarks/results.json
"""

from __future__ import annotations

import argparse
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
from nexus.reasoning.model_interface import DummyModel, ModelInterface
from nexus.reasoning.verifier import Verifier, VerificationResult


# ---- LLM-only baseline model ----

class ClosedBookModel(ModelInterface):
    """
    LLM-only baseline: answers WITHOUT any external evidence.
    
    This represents a small reasoning model operating in closed-book mode --
    it has no access to a knowledge graph or retrieved documents.
    The expected behavior is "I don't know" type responses, which means
    high answer honesty but zero factual answers.
    """
    
    def generate(self, prompt: str) -> str:
        return (
            "I don't have enough specific information to answer this question "
            "accurately. Without access to the relevant documents, experiment "
            "reports, or knowledge sources, I cannot provide a factual answer."
        )


# ---- Question loader ----

def load_questions(jsonl_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load questions from the JSONL dataset, optionally limited to N questions."""
    questions: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit and limit > 0:
        questions = questions[:limit]
    return questions


# ---- Pipeline runners ----

def run_nexus_pipeline(
    question_text: str,
    graph: InMemoryGraphStore,
    model: ModelInterface,
    verifier: Verifier,
) -> dict[str, Any]:
    """
    Run the full NEXUS pipeline and return timing + metrics.
    
    Returns a dict with:
        - answer: the generated answer text
        - passed: whether verification passed
        - hallucination_rate: float 0.0-1.0
        - supported_count: number of supported claims
        - unsupported_count: number of unsupported claims
        - path_count: number of traversal paths found
        - is_insufficient: whether the answer says "Insufficient evidence"
        - latency_s: total wall-clock seconds
        - error: error message if pipeline crashed (None otherwise)
    """
    t0 = time.perf_counter()
    try:
        result = answer_question(question_text, graph, model=model, verifier=verifier)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "answer": f"[ERROR] {exc}",
            "passed": False,
            "hallucination_rate": 1.0,
            "supported_count": 0,
            "unsupported_count": 0,
            "path_count": 0,
            "is_insufficient": False,
            "latency_s": round(elapsed, 4),
            "error": str(exc),
        }
    elapsed = time.perf_counter() - t0

    verif: VerificationResult | None = result.get("verification")
    answer = result.get("answer", "")

    is_insufficient = "insufficient evidence" in answer.lower()

    if verif is not None:
        passed = verif.passed
        hall_rate = verif.hallucination_rate
        supported = verif.supported_count
        unsupported = len(verif.unsupported_claims)
    else:
        passed = True
        hall_rate = 0.0
        supported = 0
        unsupported = 0

    return {
        "answer": answer,
        "passed": passed,
        "hallucination_rate": round(hall_rate, 4),
        "supported_count": supported,
        "unsupported_count": unsupported,
        "path_count": result.get("path_count", 0),
        "is_insufficient": is_insufficient,
        "latency_s": round(elapsed, 4),
        "error": None,
    }


def run_baseline(
    question_text: str,
    model: ModelInterface,
) -> dict[str, Any]:
    """
    Run the LLM-only (closed-book) baseline.
    
    The model receives only the question text -- no evidence, no graph.
    Expected behavior: honest "I don't know" for all questions.
    """
    t0 = time.perf_counter()
    try:
        prompt = (
            "Answer the following question without any external knowledge or "
            "retrieval. If you don't know, say so honestly.\n\n"
            f"QUESTION: {question_text}\n\nANSWER:"
        )
        answer = model.generate(prompt)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "answer": f"[ERROR] {exc}",
            "latency_s": round(elapsed, 4),
            "error": str(exc),
        }
    elapsed = time.perf_counter() - t0
    return {
        "answer": answer,
        "latency_s": round(elapsed, 4),
        "error": None,
    }


# ---- Metrics computation ----

def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from benchmark results."""
    total = len(results)
    nexus_errors = sum(1 for r in results if r["nexus"].get("error"))
    baseline_errors = sum(1 for r in results if r["baseline"].get("error"))

    nexus_answered = sum(1 for r in results if not r["nexus"]["is_insufficient"] and not r["nexus"].get("error"))
    nexus_insufficient = sum(1 for r in results if r["nexus"]["is_insufficient"])
    nexus_passed = sum(1 for r in results if r["nexus"]["passed"] and not r["nexus"].get("error"))
    nexus_hall_rates = [r["nexus"]["hallucination_rate"] for r in results if not r["nexus"].get("error")]
    nexus_latencies = [r["nexus"]["latency_s"] for r in results if not r["nexus"].get("error")]
    nexus_paths = [r["nexus"]["path_count"] for r in results if not r["nexus"].get("error")]

    baseline_latencies = [r["baseline"]["latency_s"] for r in results if not r["baseline"].get("error")]

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "total_questions": total,
        "nexus_errors": nexus_errors,
        "baseline_errors": baseline_errors,
        "nexus": {
            "answered": nexus_answered,
            "insufficient_evidence": nexus_insufficient,
            "answer_rate": round(nexus_answered / total, 4) if total > 0 else 0.0,
            "verification_passed": nexus_passed,
            "verification_pass_rate": round(nexus_passed / total, 4) if total > 0 else 0.0,
            "avg_hallucination_rate": round(avg(nexus_hall_rates), 4),
            "min_hallucination_rate": round(min(nexus_hall_rates), 4) if nexus_hall_rates else 0.0,
            "max_hallucination_rate": round(max(nexus_hall_rates), 4) if nexus_hall_rates else 0.0,
            "avg_latency_s": round(avg(nexus_latencies), 4),
            "avg_paths_found": round(avg(nexus_paths), 2),
        },
        "baseline": {
            "avg_latency_s": round(avg(baseline_latencies), 4),
        },
    }


# ---- Display ----

def print_comparison(summary: dict[str, Any]):
    """Print a human-readable comparison table."""
    n = summary["nexus"]
    b = summary["baseline"]
    total = summary["total_questions"]

    print()
    print("=" * 72)
    print("  NEXUS QA Benchmark -- Results Summary")
    print("=" * 72)
    print(f"  Questions benchmarked:  {total}")
    print(f"  NEXUS errors:           {summary['nexus_errors']}")
    print(f"  Baseline errors:        {summary['baseline_errors']}")
    print()
    print("  -- Comparison: NEXUS vs LLM-only Baseline --")
    print(f"  {'Metric':<35} {'NEXUS':>12} {'Baseline':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    
    # Answer rate
    nexus_answer_rate = n["answer_rate"]
    nexus_ans_str = f"{nexus_answer_rate:.1%} ({n['answered']}/{total})"
    print(f"  {'Answer rate':<35} {nexus_ans_str:>12} {'N/A (closed-book)':>12}")
    
    # Insufficient evidence
    ins_str = f"{n['insufficient_evidence']}/{total}"
    print(f"  {'Insufficient evidence':<35} {ins_str:>12} {'N/A':>12}")
    
    # Hallucination rate (key metric)
    hall_str = f"{n['avg_hallucination_rate']:.2%}"
    print(f"  {'Avg hallucination rate':<35} {hall_str:>12} {'N/A':>12}")
    
    # Verification pass rate
    ver_p_str = f"{n['verification_pass_rate']:.1%} ({n['verification_passed']}/{total})"
    print(f"  {'Verification pass rate':<35} {ver_p_str:>12} {'N/A':>12}")
    
    # Latency
    n_lat = f"{n['avg_latency_s']:.3f}s"
    b_lat = f"{b['avg_latency_s']:.3f}s"
    print(f"  {'Avg latency':<35} {n_lat:>12} {b_lat:>12}")
    
    # Paths
    n_paths = f"{n['avg_paths_found']:.1f}"
    print(f"  {'Avg paths found':<35} {n_paths:>12} {'N/A':>12}")
    
    print()
    print("  Key insight: NEXUS hallucination rate should be significantly lower")
    print("  than any retrieval-free baseline because every claim is verified")
    print("  against structured evidence from the knowledge graph.")
    print("=" * 72)
    print()


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS QA Benchmark Harness -- compare NEXUS vs LLM-only baseline"
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Number of questions to benchmark (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/results.json",
        help="Output file for results (default: benchmarks/results.json)"
    )
    parser.add_argument(
        "--no-populate", action="store_true",
        help="Skip graph population (use existing populated graph -- for debugging)"
    )
    args = parser.parse_args()

    # Resolve paths
    dataset_path = _project_root / "benchmarks" / "qa-dataset" / "questions.jsonl"
    output_path = _project_root / args.output

    if not dataset_path.exists():
        print(f"Error: QA dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading questions from: {dataset_path}")
    questions = load_questions(str(dataset_path), args.limit)
    total = len(questions)
    print(f"Loaded {total} questions (limit={args.limit})")

    # Populate graph
    print("\nPopulating knowledge graph...")
    graph = InMemoryGraphStore()
    if not args.no_populate:
        from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR
        try:
            graph = populate_graph(EXPERIMENTS_DIR, graph)
        except FileNotFoundError:
            print("  Warning: sam-lm/experiments not found, trying ingest_docs fallback...")
            from nexus.ingestion.ingest_docs import ingest_directory
            docs_dir = _project_root / "docs"
            if docs_dir.exists():
                ingest_directory(docs_dir, graph)
            sam_dir = _project_root / "sam-lm" / "docs"
            if sam_dir.exists():
                ingest_directory(sam_dir, graph)
            exp_dir = _project_root / "sam-lm" / "experiments"
            if exp_dir.exists():
                ingest_directory(exp_dir, graph)
    print(f"Graph ready: {graph.node_count} nodes, {graph.edge_count} edges")

    # Initialize models
    nexus_model = DummyModel()
    verifier = Verifier(hallucination_threshold=0.2)
    baseline_model = ClosedBookModel()

    print(f"\nRunning benchmark on {total} questions...\n")

    results: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        qtext = q["question"]
        qid = q.get("id", f"q{str(i).zfill(3)}")
        
        # Progress
        marker = f"[{i}/{total}]"
        
        # Run NEXUS pipeline
        nexus_result = run_nexus_pipeline(qtext, graph, nexus_model, verifier)
        
        # Run LLM-only baseline
        baseline_result = run_baseline(qtext, baseline_model)
        
        # Status indicator
        if nexus_result["error"]:
            status = "ERR"
        elif nexus_result["is_insufficient"]:
            status = "INS"
        elif nexus_result["passed"]:
            status = "PASS"
        else:
            status = f"HALL({nexus_result['hallucination_rate']:.0%})"
        
        print(f"  {marker} {qid}: {status} | paths={nexus_result['path_count']} | "
              f"nexus={nexus_result['latency_s']:.3f}s | baseline={baseline_result['latency_s']:.3f}s")
        
        results.append({
            "question_id": qid,
            "question": qtext,
            "ground_truth": q.get("answer", ""),
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            "hops": q.get("hops", 1),
            "nexus": nexus_result,
            "baseline": baseline_result,
        })

    # Compute summary
    summary = compute_summary(results)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "limit": args.limit,
            "graph_nodes": graph.node_count,
            "graph_edges": graph.edge_count,
        },
        "summary": summary,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    print_comparison(summary)


if __name__ == "__main__":
    main()
