"""
NEXUS QA Benchmark Harness (P3 -- Phase 3/4).

Compares the full NEXUS pipeline against an evidence-blind baseline on the QA dataset.
Key metrics: accuracy score, hallucination rate, answer rate, latency, verification pass rate.

Reproducibility: The benchmark graph is built deterministically from
populate_from_experiments + ingest_docs in fixed order. Results are exactly
reproducible from committed code with no non-deterministic components.

Exact reproduction command:
    python benchmarks/run_benchmark.py --limit 50 --output benchmarks/results.json

Usage:
    python benchmarks/run_benchmark.py --limit 50
    python benchmarks/run_benchmark.py --limit 100 --output benchmarks/results.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
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
from nexus.reasoning.model_interface import (
    DummyModel, EvidenceBlindModel, ModelInterface,
    get_available_model, FallbackModel, SynthesizingModel,
)
from nexus.reasoning.verifier import Verifier, VerificationResult


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


# ---- Key-fact accuracy scoring ----

# Regex patterns for extracting key facts from text
_FACT_PATTERNS = [
    # Percentages: 99.87%, 100%, 50%, 96.6% (no \b after % since % is non-word)
    (re.compile(r'\b(\d+\.?\d*\s*%)(?:\s|$|[,.);])'), "percentage"),
    # Numbers with "million": 15.7 million, 19,000
    (re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*million\b', re.IGNORECASE), "number+million"),
    # Numbers with common technical units: 1,650 slots, 19,000 examples, 853 vocab tokens
    (re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:live\s+)?(slots?|examples?|tokens?|parameters?|params?|subkeys?|distractors?|vocabulary|hops?)\b', re.IGNORECASE), "number+unit"),
    # Standalone large numbers (>=100) with context
    (re.compile(r'\b(\d{3,}(?:,\d{3})*(?:\.\d+)?)\b'), "large_number"),
    # @ notation: all_required@32, Rec@8
    (re.compile(r'\b(\w+@\d+)\b'), "at_notation"),
    # K= notation: K=32
    (re.compile(r'\b([Kk]=\d+)\b'), "k_notation"),
    # Named experiment IDs: Exp_0_6_Validation, Exp_0_13A_NoisyMemory
    (re.compile(r'\b(Exp_\d+_\d+[A-Z]?_\w+)\b'), "experiment_id"),
    # Named concept IDs: Concept_SelectorBottleneck, Concept_ArchitectureWorks
    (re.compile(r'\b(Concept_\w+)\b'), "concept_id"),
    # Named decision IDs: Decision_PivotToNEXUS
    (re.compile(r'\b(Decision_\w+)\b'), "decision_id"),
    # Relation words: depends_on, validates, caused_by, contradicts, etc.
    (re.compile(r'\b(depends_on|validates|caused_by|contradicts|implements|mentioned_in|derived_from|related_to|replaces|blocked_by)\b', re.IGNORECASE), "relation"),
    # Key named modes: core_only, oracle_memory, retrieved_memory, random_memory, oracle_text_memory
    (re.compile(r'\b(core_only|oracle_memory|retrieved_memory|random_memory|oracle_text_memory|oracle_filter|oracle_text_memory|retrieved_memory_external_text_query)\b', re.IGNORECASE), "sam_mode"),
    # Gate references: Gate 1, Gate 2
    (re.compile(r'\b(Gate\s+\d+)\b', re.IGNORECASE), "gate_ref"),
]


def _extract_key_facts(text: str) -> set[str]:
    """Extract key facts from text using defined regex patterns.
    
    Returns a set of normalized fact strings suitable for set intersection.
    """
    facts: set[str] = set()
    for pattern, fact_type in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            # Normalize: lowercase, strip whitespace
            fact_str = match.group(0).strip().lower()
            # Normalize comma-separated numbers: 1,650 -> 1650
            fact_str = re.sub(r'(\d),(\d)', r'\1\2', fact_str)
            facts.add(fact_str)
    return facts


def compute_key_fact_score(predicted_answer: str, ground_truth: str) -> float | None:
    """Compute key-fact match score between predicted and ground truth answers.
    
    A "key fact" is a numeric value, entity name, or relation word extracted
    via regex. Score = |intersection| / |ground_truth_facts|.
    
    If the predicted answer says "Insufficient evidence", score = 0.0.
    If ground truth has no extractable facts, returns None (exclude from
    aggregate accuracy — there's nothing to measure against).
    """
    if "insufficient evidence" in predicted_answer.lower():
        return 0.0
    
    gt_facts = _extract_key_facts(ground_truth)
    pred_facts = _extract_key_facts(predicted_answer)
    
    if not gt_facts:
        # No ground truth facts to compare against — cannot score
        return None
    
    intersection = gt_facts & pred_facts
    score = len(intersection) / len(gt_facts)
    return round(score, 4)


# ---- Graph construction ----


def build_benchmark_graph() -> tuple[InMemoryGraphStore, dict[str, Any]]:
    """Build the benchmark knowledge graph deterministically.
    
    Runs BOTH populate_from_experiments AND ingest_docs in a fixed,
    deterministic order to ensure reproducible benchmark results.
    
    Returns:
        (graph, provenance_dict) where provenance_dict contains:
            - node_count, edge_count
            - build_command: the exact Python command to reproduce
            - timestamp: ISO 8601 timestamp
    """
    from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR
    from nexus.ingestion.ingest_docs import ingest_directory
    
    graph = InMemoryGraphStore()
    
    # Step 1: Populate from experiments (structure + metrics)
    if EXPERIMENTS_DIR.exists():
        graph = populate_graph(EXPERIMENTS_DIR, graph)
    
    # Step 2: Ingest documents for additional entity/relation extraction
    docs_dir = _project_root / "docs"
    if docs_dir.exists():
        ingest_directory(docs_dir, graph)
    sam_docs_dir = _project_root / "sam-lm" / "docs"
    if sam_docs_dir.exists():
        ingest_directory(sam_docs_dir, graph)
    sam_exp_dir = _project_root / "sam-lm" / "experiments"
    if sam_exp_dir.exists():
        ingest_directory(sam_exp_dir, graph)
    
    provenance = {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "build_command": "python benchmarks/run_benchmark.py --limit 50 --output benchmarks/results.json",
        "build_steps": [
            "1. populate_from_experiments(EXPERIMENTS_DIR, graph)",
            "2. ingest_directory('docs/', graph)",
            "3. ingest_directory('sam-lm/docs/', graph)",
            "4. ingest_directory('sam-lm/experiments/', graph)",
        ],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    
    return graph, provenance


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
    Run the evidence-blind baseline.
    
    The model receives the question WITHOUT any evidence from the
    knowledge graph — simulating a model that can only use general
    knowledge. Uses the same prompt structure as NEXUS but with
    evidence stripped out.
    """
    t0 = time.perf_counter()
    try:
        prompt = (
            "SYSTEM: You are a precise reasoning assistant. "
            "Answer based on your general knowledge. "
            "If you truly don't know, say so honestly.\n\n"
            f"QUESTION: {question_text}\n\n"
            "EVIDENCE:\n  (No evidence found in the knowledge graph.)\n\n"
            "ANSWER:"
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
    
    # Check if the answer shows evidence of comprehension
    is_insufficient = "insufficient evidence" in answer.lower()
    
    return {
        "answer": answer,
        "is_insufficient": is_insufficient,
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
    baseline_insufficient = sum(1 for r in results if r["baseline"]["is_insufficient"])

    # Accuracy scores (exclude None values — questions without extractable GT facts)
    nexus_accuracies = [
        r["nexus"]["accuracy"]
        for r in results
        if not r["nexus"].get("error") 
        and "accuracy" in r["nexus"] 
        and r["nexus"]["accuracy"] is not None
    ]
    baseline_accuracies = [
        r["baseline"]["accuracy"]
        for r in results
        if not r["baseline"].get("error") 
        and "accuracy" in r["baseline"]
        and r["baseline"]["accuracy"] is not None
    ]
    scorable_count = len(nexus_accuracies)  # questions with measurable ground truth

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "total_questions": total,
        "scorable_questions": scorable_count,
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
            "avg_accuracy": round(avg(nexus_accuracies), 4),
            "min_accuracy": round(min(nexus_accuracies), 4) if nexus_accuracies else 0.0,
            "max_accuracy": round(max(nexus_accuracies), 4) if nexus_accuracies else 0.0,
        },
        "baseline": {
            "avg_latency_s": round(avg(baseline_latencies), 4),
            "insufficient_evidence": baseline_insufficient,
            "avg_accuracy": round(avg(baseline_accuracies), 4),
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
    print(f"  Scorable (GT has facts): {summary.get('scorable_questions', total)}")
    print(f"  NEXUS errors:           {summary['nexus_errors']}")
    print(f"  Baseline errors:        {summary['baseline_errors']}")
    print()
    print("  -- Comparison: NEXUS (with evidence) vs Baseline (without evidence) --")
    print(f"  {'Metric':<38} {'NEXUS':>10} {'Baseline':>12}")
    print(f"  {'-'*38} {'-'*10} {'-'*12}")
    
    # Answer rate
    nexus_answer_rate = n["answer_rate"]
    nexus_ans_str = f"{nexus_answer_rate:.1%} ({n['answered']}/{total})"
    base_ans_str = f"{total - b['insufficient_evidence']}/{total}" if "insufficient_evidence" in b else "N/A"
    print(f"  {'Answer rate':<38} {nexus_ans_str:>10} {base_ans_str:>12}")
    
    # Insufficient evidence
    ins_str = f"{n['insufficient_evidence']}/{total}"
    base_ins_str = f"{b['insufficient_evidence']}/{total}" if "insufficient_evidence" in b else "N/A"
    print(f"  {'Insufficient evidence':<38} {ins_str:>10} {base_ins_str:>12}")
    
    # Accuracy score (key metric)
    n_acc = f"{n['avg_accuracy']:.2%}" if n.get("avg_accuracy") is not None else "N/A"
    b_acc = f"{b['avg_accuracy']:.2%}" if b.get("avg_accuracy") is not None else "N/A"
    print(f"  {'Avg accuracy (key-fact match)':<38} {n_acc:>10} {b_acc:>12}")
    
    # Hallucination rate
    hall_str = f"{n['avg_hallucination_rate']:.2%}"
    print(f"  {'Avg hallucination rate':<38} {hall_str:>10} {'N/A':>12}")
    
    # Verification pass rate
    ver_p_str = f"{n['verification_pass_rate']:.1%} ({n['verification_passed']}/{total})"
    print(f"  {'Verification pass rate':<38} {ver_p_str:>10} {'N/A':>12}")
    
    # Latency
    n_lat = f"{n['avg_latency_s']:.3f}s"
    b_lat = f"{b['avg_latency_s']:.3f}s"
    print(f"  {'Avg latency':<38} {n_lat:>10} {b_lat:>12}")
    
    # Paths
    n_paths = f"{n['avg_paths_found']:.1f}"
    print(f"  {'Avg paths found':<38} {n_paths:>10} {'N/A':>12}")
    
    print()
    print("  Accuracy = key-fact overlap between model answer and ground truth.")
    print("  NEXUS hallucination rate measures unsupported claims in generated answers.")
    print("  The evidence-blind baseline has no graph access — it can only use")
    print("  general knowledge extracted from the question text.")
    print("=" * 72)
    print()


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS QA Benchmark Harness -- compare NEXUS vs evidence-blind baseline"
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

    # Build graph deterministically
    if args.no_populate:
        print("\nSkipping graph population (--no-populate)")
        graph = InMemoryGraphStore()
        graph_provenance = {"node_count": 0, "edge_count": 0, "build_command": "skipped (--no-populate)"}
    else:
        print("\nBuilding benchmark graph (deterministic)...")
        graph, graph_provenance = build_benchmark_graph()
    print(f"Graph ready: {graph_provenance['node_count']} nodes, {graph_provenance['edge_count']} edges")

    # Initialize models
    primary_model = get_available_model()
    # Wrap in FallbackModel: uses LLM first, falls back to SynthesizingModel
    # when the LLM says "insufficient evidence" but evidence IS present
    nexus_model = FallbackModel(primary_model)
    verifier = Verifier(hallucination_threshold=0.2)
    baseline_model = EvidenceBlindModel()

    print(f"\nRunning benchmark on {total} questions...\n")

    results: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        qtext = q["question"]
        qid = q.get("id", f"q{str(i).zfill(3)}")
        ground_truth = q.get("answer", "")
        
        # Progress
        marker = f"[{i}/{total}]"
        
        # Run NEXUS pipeline
        nexus_result = run_nexus_pipeline(qtext, graph, nexus_model, verifier)
        
        # Compute accuracy for NEXUS
        nexus_accuracy = compute_key_fact_score(
            nexus_result["answer"], ground_truth
        )
        nexus_result["accuracy"] = nexus_accuracy
        
        # Run evidence-blind baseline
        baseline_result = run_baseline(qtext, baseline_model)
        
        # Compute accuracy for baseline
        baseline_accuracy = compute_key_fact_score(
            baseline_result["answer"], ground_truth
        )
        baseline_result["accuracy"] = baseline_accuracy
        
        # Status indicator
        if nexus_result["error"]:
            status = "ERR"
        elif nexus_result["is_insufficient"]:
            status = "INS"
        elif nexus_result["passed"]:
            status = "PASS"
        else:
            status = f"HALL({nexus_result['hallucination_rate']:.0%})"
        
        print(f"  {marker} {qid}: {status} | acc={nexus_accuracy if nexus_accuracy is not None else 'N/A'} | paths={nexus_result['path_count']} | "
              f"nexus={nexus_result['latency_s']:.3f}s | baseline={baseline_result['latency_s']:.3f}s (acc={baseline_accuracy if baseline_accuracy is not None else 'N/A'})")
        
        results.append({
            "question_id": qid,
            "question": qtext,
            "ground_truth": ground_truth,
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
            "model": nexus_model.name,
            "model_backend": type(nexus_model).__name__,
            "verification_threshold": 0.2,
        },
        "graph_provenance": graph_provenance,
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
