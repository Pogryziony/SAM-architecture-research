"""
NEXUS External Corpus Benchmark — generalization test.

Runs the full NEXUS pipeline on an external corpus
with zero hand-curated aliases. Measures whether rule-based entity/relation
extraction generalizes to unfamiliar domains.

Usage:
    python benchmarks/run_external_benchmark.py --corpus <path/to/docs>
    python benchmarks/run_external_benchmark.py --corpus <path/to/docs> --limit 5 --verbose
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.ingestion.ingest_generic import ingest_generic, print_stats
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import (
    DummyModel, get_available_model, FallbackModel,
)
from nexus.reasoning.verifier import Verifier
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG

# Import scoring functions from the main benchmark
from benchmarks.run_benchmark import (
    load_questions, compute_key_fact_score, run_nexus_pipeline,
    run_baseline, compute_summary, print_comparison,
    _count_tokens,
)


def build_external_graph(corpus_dirs: list[Path], verbose: bool = False) -> InMemoryGraphStore:
    """Build a knowledge graph from one or more external corpus directories."""
    graph = InMemoryGraphStore()
    total_nodes = 0
    total_edges = 0

    for corpus_dir in corpus_dirs:
        print(f"\nBuilding graph from: {corpus_dir}")
        t0 = time.perf_counter()

        nodes, edges = ingest_generic(
            corpus_dir, graph,
            patterns=["**/*.md", "**/*.txt"],
            verbose=verbose,
        )
        total_nodes += nodes
        total_edges += edges

        elapsed = time.perf_counter() - t0
        print(f"  -> {nodes} nodes added, {edges} edges added in {elapsed:.2f}s")

    print(f"\nTotal graph: {total_nodes} nodes, {total_edges} edges")
    return graph


def run_external_benchmark(
    corpus_dirs: list[Path],
    qa_path: Path,
    limit: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full external corpus benchmark pipeline."""

    # ── Build graph ──
    graph = build_external_graph(corpus_dirs, verbose=False)

    if verbose:
        print_stats(graph)

    # ── Load questions ──
    questions = load_questions(str(qa_path), limit=limit)
    print(f"\nLoaded {len(questions)} questions from {qa_path}")

    # ── Setup model ──
    config = DEFAULT_CONFIG
    print(f"\nModel: FallbackModel (DummyModel primary + SynthesizingModel fallback)")
    primary_model = get_available_model()
    model = FallbackModel(primary_model)
    verifier = Verifier(hallucination_threshold=config.hallucination_threshold)

    # ── Run pipeline ──
    results: list[dict[str, Any]] = []
    nexus_answers = 0
    entity_res_hits = 0

    for i, q in enumerate(questions):
        qid = q["id"]
        question_text = q["question"]
        ground_truth = q["answer"]

        if verbose:
            print(f"\n[{i+1}/{len(questions)}] {qid}: {question_text[:80]}...")

        # NEXUS pipeline
        nexus_result = run_nexus_pipeline(question_text, graph, model, verifier)

        # Track entity resolution: did we find any entities?
        er_hit = False
        entity_ids = nexus_result.get("parsed_entity_ids", [])
        if entity_ids:
            er_hit = True
            entity_res_hits += 1

        # Accuracy scoring
        accuracy_data = compute_key_fact_score(
            nexus_result["answer"], ground_truth, use_fuzzy=True
        )

        # Conciseness
        answer_tokens = _count_tokens(nexus_result["answer"])
        gt_tokens = _count_tokens(ground_truth)
        conciseness_ratio = round(answer_tokens / max(gt_tokens, 1), 2)
        too_verbose = conciseness_ratio > 3.0

        nexus_result["accuracy"] = accuracy_data["fuzzy_accuracy"]
        nexus_result["exact_accuracy"] = accuracy_data["exact_accuracy"]
        nexus_result["scoring_detail"] = accuracy_data["scoring_detail"]
        nexus_result["entity_resolution_hit"] = er_hit
        nexus_result["conciseness"] = {
            "answer_tokens": answer_tokens,
            "gt_tokens": gt_tokens,
            "ratio": conciseness_ratio,
            "too_verbose": too_verbose,
        }

        # Baseline
        baseline_result = run_baseline(question_text, model)
        baseline_acc = compute_key_fact_score(
            baseline_result["answer"], ground_truth, use_fuzzy=True
        )
        baseline_result["accuracy"] = baseline_acc["fuzzy_accuracy"]
        baseline_result["exact_accuracy"] = baseline_acc["exact_accuracy"]

        if nexus_result.get("is_insufficient", False):
            if verbose:
                print(f"  NEXUS: INSUFFICIENT EVIDENCE (entities: {entity_ids})")
        else:
            nexus_answers += 1
            if verbose:
                acc_str = f"{nexus_result['accuracy']:.0%}" if nexus_result['accuracy'] is not None else "N/A"
                print(f"  NEXUS: acc={acc_str}, paths={nexus_result['path_count']}, hall={nexus_result['hallucination_rate']:.2f}")

        results.append({
            "question_id": qid,
            "question": question_text,
            "ground_truth": ground_truth,
            "question_type": q.get("question_type", "factual"),
            "difficulty": q.get("difficulty", "easy"),
            "hops": q.get("hops", 1),
            "nexus": nexus_result,
            "baseline": baseline_result,
        })

    # ── Compute summary ──
    summary = compute_summary(results)

    # Add external-corpus-specific fields
    entity_res_rate = round(entity_res_hits / len(questions), 4) if questions else 0.0
    summary["corpus"] = {
        "paths": [str(d) for d in corpus_dirs],
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "entity_resolution_hits": entity_res_hits,
        "entity_resolution_total": len(questions),
        "entity_resolution_rate": entity_res_rate,
    }
    summary["generalization_assessment"] = _assess_generalization(summary)

    return {"results": results, "summary": summary, "graph": graph}


def _assess_generalization(summary: dict[str, Any]) -> str:
    """Assess whether NEXUS generalizes to the external corpus."""
    n = summary["nexus"]
    accuracy = n.get("avg_accuracy", 0.0) or 0.0
    answer_rate = n.get("answer_rate", 0.0) or 0.0
    entity_res_rate = summary.get("corpus", {}).get("entity_resolution_rate", 0.0)

    if accuracy > 0.25:
        return (
            f"PASS (generalization confirmed): "
            f"accuracy={accuracy:.1%}, answer_rate={answer_rate:.1%}, "
            f"entity_resolution={entity_res_rate:.1%}. "
            f"Rule-based extraction works on unfamiliar domains without hand-curated aliases."
        )
    else:
        return (
            f"BELOW THRESHOLD (25%): "
            f"accuracy={accuracy:.1%}, answer_rate={answer_rate:.1%}, "
            f"entity_resolution={entity_res_rate:.1%}. "
            f"This reveals the Phase 5 problem early: extraction quality degrades "
            f"on unfamiliar domains. Entity resolution is the bottleneck."
        )


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS External Corpus Benchmark — generalization test"
    )
    parser.add_argument(
        "--corpus", nargs="+", required=True,
        help="Path(s) to external corpus directory"
    )
    parser.add_argument(
        "--qa",
        default=None,
        help="Path to QA questions JSONL file"
    )
    parser.add_argument(
        "--limit", type=int, default=15,
        help="Max questions to benchmark"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    corpus_dirs = [Path(d) for d in args.corpus]
    for d in corpus_dirs:
        if not d.exists():
            print(f"Error: corpus directory not found: {d}")
            sys.exit(1)

    qa_path = Path(args.qa) if args.qa else None
    if not qa_path or not qa_path.exists():
        print(f"Error: QA file required. Use --qa <path/to/questions.jsonl>")
        sys.exit(1)

    print("=" * 72)
    print("  NEXUS External Corpus Benchmark — Generalization Test")
    print("=" * 72)
    for d in corpus_dirs:
        print(f"  Corpus: {d}")
    print(f"  QA file: {qa_path}")
    print(f"  Limit: {args.limit}")
    print()

    result = run_external_benchmark(
        corpus_dirs, qa_path,
        limit=args.limit,
        verbose=args.verbose,
    )

    # ── Print summary ──
    print_comparison(result["summary"])

    # ── Print generalization assessment ──
    assessment = result["summary"].get("generalization_assessment", "N/A")
    print(f"\n  Generalization Assessment:")
    print(f"  {assessment}")

    # ── Save results ──
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _project_root / "benchmarks" / f"external_results_{timestamp}.json"

    serializable = {
        "results": result["results"],
        "summary": {
            k: v for k, v in result["summary"].items()
        },
        "meta": {
            "corpus": [str(d) for d in corpus_dirs],
            "qa_file": str(qa_path),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
