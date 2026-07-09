"""
Oracle Evidence Test - Phase 1 Decision Gate

Purpose: Determine if evidence quality is the bottleneck.

Approach:
1. Run 30 questions through NEXUS pipeline (normal mode)
2. Extract the evidence pack that was built
3. Inject ground-truth fact as one extra evidence line
4. Rerun both generators (LLM + SynthesizingModel) with oracle evidence
5. Compare accuracy: baseline vs oracle

If accuracy jumps to 60-80%, evidence is the bottleneck → proceed to Phase 2
If accuracy stays ~30%, generators are limiting → pivot strategy

Also measures evidence_recall: what fraction of ground-truth facts are present in evidence pack.
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
    FallbackModel, SynthesizingModel, get_available_model,
)
from nexus.reasoning.verifier import Verifier

from benchmarks.run_benchmark import (
    build_benchmark_graph, load_questions, compute_key_fact_score, _extract_key_facts,
)


def _count_tokens(text: str) -> int:
    """Simple word-count token estimation."""
    return len(text.split())


def _flatten_evidence_pack(pack: dict[str, Any]) -> str:
    """Flatten evidence_pack dict to a single text string for fact extraction.

    Produces a human-readable representation suitable for prompt injection
    and key-fact matching against ground truth.
    """
    if not pack:
        return ""
    parts: list[str] = []

    # Node facts (curated key_findings/descriptions)
    for nf in pack.get("node_facts", []):
        text = nf.get("text", "") if isinstance(nf, dict) else str(nf)
        if text.strip():
            parts.append(text.strip())

    # Edge-based facts
    for f in pack.get("facts", []):
        if isinstance(f, str) and f.strip():
            parts.append(f.strip())

    # Neighbor key_findings
    for nf in pack.get("neighbor_facts", []):
        text = nf.get("text", "") if isinstance(nf, dict) else str(nf)
        if text.strip():
            parts.append(text.strip())

    # Numbers section
    for num_entry in pack.get("numbers", []):
        bits = []
        for k, v in num_entry.items():
            if k not in ("entity", "source") and v is not None:
                bits.append(f"{k}: {v}")
        if bits:
            parts.append("; ".join(bits))

    # Numbers by metric (grouped) - include more entries
    for metric_name, entries in pack.get("numbers_by_metric", {}).items():
        for entry in entries[:10]:
            entity = entry.get("entity", "")
            value = entry.get("value", "")
            if value:
                parts.append(f"{entity}: {metric_name} = {value}")

    return "\n".join(parts)


def run_oracle_evidence_test(
    limit: int = 30,
    output_path: str = "benchmarks/results/oracle_evidence_test.json",
) -> None:
    """
    Run oracle evidence test on 30 questions.
    
    For each question:
    1. Run normal NEXUS pipeline (get evidence pack)
    2. Extract ground-truth fact from answer
    3. Inject GT fact into evidence pack
    4. Rerun both generators (LLM + SynthesizingModel)
    5. Measure if accuracy improves (and by how much)
    6. Compute evidence_recall: GT facts present in original evidence
    """
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
    print(f"Graph ready: {graph_provenance['node_count']} nodes, {graph_provenance['edge_count']} edges")

    # Initialize models
    primary_model = get_available_model()
    nexus_model = FallbackModel(primary_model)
    verifier = Verifier(hallucination_threshold=0.2)
    synth_model = SynthesizingModel()

    print(f"\nRunning oracle evidence test on {total} questions...\n")

    results: list[dict[str, Any]] = []
    
    for i, q in enumerate(questions, 1):
        qtext = q["question"]
        qid = q.get("id", f"q{str(i).zfill(3)}")
        ground_truth = q.get("answer", "")
        
        marker = f"[{i}/{total}]"
        
        # Step 1: Run normal NEXUS pipeline (get baseline evidence)
        print(f"{marker} Processing: {qtext[:60]}...")
        
        try:
            t0 = time.perf_counter()
            baseline_result = answer_question(
                question=qtext,
                graph=graph,
                model=nexus_model,
                verifier=verifier,
            )
            baseline_latency = time.perf_counter() - t0
            
            baseline_answer = baseline_result.get("answer", "")
            baseline_path_count = baseline_result.get("path_count", 0)
            evidence_pack = baseline_result.get("evidence_pack", {})
            baseline_evidence = _flatten_evidence_pack(evidence_pack)
            baseline_accuracy = compute_key_fact_score(baseline_answer, ground_truth)
            
        except Exception as exc:
            results.append({
                "question_id": qid,
                "question": qtext,
                "error": str(exc),
                "error_stage": "baseline_run",
            })
            continue
        
        # Step 2: Extract GT fact from ground truth
        gt_facts = _extract_key_facts(ground_truth)
        if not gt_facts:
            results.append({
                "question_id": qid,
                "question": qtext,
                "ground_truth": ground_truth,
                "baseline_answer": baseline_answer,
                "baseline_accuracy": baseline_accuracy.get("fuzzy_accuracy") if isinstance(baseline_accuracy, dict) else baseline_accuracy,
                "oracle_answer_llm": None,
                "oracle_accuracy_llm": None,
                "oracle_answer_synth": None,
                "oracle_accuracy_synth": None,
                "evidence_recall": None,
                "note": "Ground truth has no extractable facts",
            })
            continue
        
        # Step 3: Compute evidence_recall (GT facts present in baseline evidence)
        evidence_facts = _extract_key_facts(baseline_evidence)
        evidence_recall_numerator = len(gt_facts & evidence_facts)
        evidence_recall_denominator = len(gt_facts)
        evidence_recall = evidence_recall_numerator / evidence_recall_denominator if evidence_recall_denominator > 0 else 0.0
        
        # Step 4: Inject GT fact INTO the real evidence (not replacing it)
        # Pick one GT fact to inject (the first one for consistency)
        gt_fact_to_inject = sorted(gt_facts)[0] if gt_facts else ""
        if baseline_evidence:
            oracle_evidence = baseline_evidence + f"\n- (ORACLE INJECTED) {gt_fact_to_inject}"
        else:
            oracle_evidence = f"- (ORACLE INJECTED) {gt_fact_to_inject}"
        
        # Step 5a: Rerun LLM with oracle evidence
        # Use the EXACT baseline prompt; inject 1 GT fact line into the EVIDENCE section.
        try:
            t0 = time.perf_counter()
            baseline_prompt = baseline_result.get("prompt_text", "")
            gt_injection = f"\n  - (ORACLE GROUND TRUTH) {gt_fact_to_inject}"
            # Find the EVIDENCE section header and inject right after it
            evidence_marker = "\nEVIDENCE:"
            evidence_idx = baseline_prompt.find(evidence_marker)
            if evidence_idx != -1:
                # Insert after the "EVIDENCE:" line (find the next newline)
                next_nl = baseline_prompt.find("\n", evidence_idx + len(evidence_marker))
                if next_nl != -1:
                    oracle_prompt_llm = baseline_prompt[:next_nl] + gt_injection + baseline_prompt[next_nl:]
                else:
                    oracle_prompt_llm = baseline_prompt + gt_injection
            else:
                oracle_prompt_llm = baseline_prompt + gt_injection
            oracle_answer_llm = primary_model.generate(oracle_prompt_llm)
            oracle_latency_llm = time.perf_counter() - t0
            oracle_accuracy_llm = compute_key_fact_score(oracle_answer_llm, ground_truth)
        except Exception as exc:
            oracle_answer_llm = f"[ERROR] {exc}"
            oracle_accuracy_llm = None
            oracle_latency_llm = 0.0
        
        # Step 5b: Rerun SynthesizingModel with oracle evidence
        # Use the SAME baseline prompt with GT injected (identical to LLM oracle prompt).
        try:
            t0 = time.perf_counter()
            oracle_answer_synth = synth_model.generate(oracle_prompt_llm)
            oracle_latency_synth = time.perf_counter() - t0
            oracle_accuracy_synth = compute_key_fact_score(oracle_answer_synth, ground_truth)
        except Exception as exc:
            oracle_answer_synth = f"[ERROR] {exc}"
            oracle_accuracy_synth = None
            oracle_latency_synth = 0.0
        
        # Extract numeric accuracy scores
        baseline_acc_score = baseline_accuracy.get("fuzzy_accuracy") if isinstance(baseline_accuracy, dict) else baseline_accuracy
        oracle_acc_llm_score = oracle_accuracy_llm.get("fuzzy_accuracy") if isinstance(oracle_accuracy_llm, dict) else oracle_accuracy_llm
        oracle_acc_synth_score = oracle_accuracy_synth.get("fuzzy_accuracy") if isinstance(oracle_accuracy_synth, dict) else oracle_accuracy_synth
        
        # Compute improvement
        improvement_llm = (oracle_acc_llm_score or 0.0) - (baseline_acc_score or 0.0) if baseline_acc_score is not None else None
        improvement_synth = (oracle_acc_synth_score or 0.0) - (baseline_acc_score or 0.0) if baseline_acc_score is not None else None
        
        result = {
            "question_id": qid,
            "question": qtext,
            "ground_truth": ground_truth,
            "evidence_recall": round(evidence_recall, 4),
            "evidence_recall_num": evidence_recall_numerator,
            "evidence_recall_den": evidence_recall_denominator,
            "path_count": baseline_path_count,
            "evidence_tokens": _count_tokens(baseline_evidence),
            "baseline_answer": baseline_answer,
            "baseline_accuracy": round(baseline_acc_score, 4) if baseline_acc_score is not None else None,
            "baseline_latency_s": round(baseline_latency, 4),
            "oracle_answer_llm": oracle_answer_llm,
            "oracle_accuracy_llm": round(oracle_acc_llm_score, 4) if oracle_acc_llm_score is not None else None,
            "oracle_latency_llm_s": round(oracle_latency_llm, 4),
            "improvement_llm": round(improvement_llm, 4) if improvement_llm is not None else None,
            "oracle_answer_synth": oracle_answer_synth,
            "oracle_accuracy_synth": round(oracle_acc_synth_score, 4) if oracle_acc_synth_score is not None else None,
            "oracle_latency_synth_s": round(oracle_latency_synth, 4),
            "improvement_synth": round(improvement_synth, 4) if improvement_synth is not None else None,
        }
        results.append(result)
        
        # Print progress
        b_acc = f"{baseline_acc_score:.2%}" if baseline_acc_score is not None else "N/A"
        o_llm = f"{oracle_acc_llm_score:.2%}" if oracle_acc_llm_score is not None else "N/A"
        o_syn = f"{oracle_acc_synth_score:.2%}" if oracle_acc_synth_score is not None else "N/A"
        i_llm = f"{improvement_llm:+.0%}" if improvement_llm is not None else "N/A"
        i_syn = f"{improvement_synth:+.0%}" if improvement_synth is not None else "N/A"
        e_rec = f"{evidence_recall:.0%}" if evidence_recall is not None else "N/A"
        print(f"  baseline: {b_acc} | oracle LLM: {o_llm} (+{i_llm}) | synth: {o_syn} (+{i_syn}) | recall: {e_rec}")
    
    # Aggregate stats
    print("\n" + "="*80)
    print("ORACLE EVIDENCE TEST RESULTS")
    print("="*80)
    
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        print("No valid results to aggregate.")
        return
    
    # Extract metrics
    baseline_accs = [r["baseline_accuracy"] for r in valid_results if r.get("baseline_accuracy") is not None]
    oracle_llm_accs = [r["oracle_accuracy_llm"] for r in valid_results if r.get("oracle_accuracy_llm") is not None]
    oracle_synth_accs = [r["oracle_accuracy_synth"] for r in valid_results if r.get("oracle_accuracy_synth") is not None]
    improvements_llm = [r.get("improvement_llm") for r in valid_results if r.get("improvement_llm") is not None]
    improvements_synth = [r.get("improvement_synth") for r in valid_results if r.get("improvement_synth") is not None]
    evidence_recalls = [r.get("evidence_recall") for r in valid_results if r.get("evidence_recall") is not None]
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    path_counts = [r.get("path_count", 0) for r in valid_results]
    evidence_token_counts = [r.get("evidence_tokens", 0) for r in valid_results]
    
    summary = {
        "total_questions": total,
        "valid_results": len(valid_results),
        "errors": len([r for r in results if "error" in r]),
        "baseline_accuracy": {
            "mean": round(avg(baseline_accs), 4),
            "min": round(min(baseline_accs), 4) if baseline_accs else None,
            "max": round(max(baseline_accs), 4) if baseline_accs else None,
        },
        "oracle_accuracy_llm": {
            "mean": round(avg(oracle_llm_accs), 4),
            "min": round(min(oracle_llm_accs), 4) if oracle_llm_accs else None,
            "max": round(max(oracle_llm_accs), 4) if oracle_llm_accs else None,
        },
        "oracle_accuracy_synth": {
            "mean": round(avg(oracle_synth_accs), 4),
            "min": round(min(oracle_synth_accs), 4) if oracle_synth_accs else None,
            "max": round(max(oracle_synth_accs), 4) if oracle_synth_accs else None,
        },
        "improvement_llm": {
            "mean": round(avg(improvements_llm), 4),
            "min": round(min(improvements_llm), 4) if improvements_llm else None,
            "max": round(max(improvements_llm), 4) if improvements_llm else None,
        },
        "improvement_synth": {
            "mean": round(avg(improvements_synth), 4),
            "min": round(min(improvements_synth), 4) if improvements_synth else None,
            "max": round(max(improvements_synth), 4) if improvements_synth else None,
        },
        "evidence_recall": {
            "mean": round(avg(evidence_recalls), 4),
            "min": round(min(evidence_recalls), 4) if evidence_recalls else None,
            "max": round(max(evidence_recalls), 4) if evidence_recalls else None,
        },
        "path_count": {
            "mean": round(avg(path_counts), 1),
            "total_zero": sum(1 for p in path_counts if p == 0),
            "total_nonzero": sum(1 for p in path_counts if p > 0),
        },
        "evidence_tokens": {
            "mean": round(avg(evidence_token_counts), 1),
            "min": round(min(evidence_token_counts)) if evidence_token_counts else None,
            "max": round(max(evidence_token_counts)) if evidence_token_counts else None,
        },
        "oracle_ceiling": round(avg(oracle_llm_accs), 4) if oracle_llm_accs else None,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    
    # Print summary
    print(f"\nTotal questions: {summary['total_questions']}")
    print(f"Valid results: {summary['valid_results']}")
    print(f"Errors: {summary['errors']}")
    print()
    
    print("BASELINE (normal NEXUS evidence):")
    print(f"  Mean accuracy: {summary['baseline_accuracy']['mean']:.2%}")
    print(f"  Min / Max: {summary['baseline_accuracy']['min']:.2%} / {summary['baseline_accuracy']['max']:.2%}")
    print()
    
    print("ORACLE CEILING (LLM with injected GT fact):")
    print(f"  Mean accuracy: {summary['oracle_accuracy_llm']['mean']:.2%}")
    print(f"  Min / Max: {summary['oracle_accuracy_llm']['min']:.2%} / {summary['oracle_accuracy_llm']['max']:.2%}")
    print(f"  Improvement (delta): {summary['improvement_llm']['mean']:+.2%}")
    print()
    
    print("ORACLE CEILING (Synthesizer with injected GT fact):")
    print(f"  Mean accuracy: {summary['oracle_accuracy_synth']['mean']:.2%}")
    print(f"  Min / Max: {summary['oracle_accuracy_synth']['min']:.2%} / {summary['oracle_accuracy_synth']['max']:.2%}")
    print(f"  Improvement (delta): {summary['improvement_synth']['mean']:+.2%}")
    print()
    
    print("EVIDENCE RECALL (GT facts present in baseline evidence pack):")
    print(f"  Mean recall: {summary['evidence_recall']['mean']:.2%}")
    print(f"  Min / Max: {summary['evidence_recall']['min']:.2%} / {summary['evidence_recall']['max']:.2%}")
    print()

    print("GRAPH TRAVERSAL:")
    print(f"  Mean paths: {summary['path_count']['mean']:.1f}")
    print(f"  Questions with 0 paths: {summary['path_count']['total_zero']}")
    print(f"  Questions with paths: {summary['path_count']['total_nonzero']}")
    print()

    print("EVIDENCE SIZE:")
    print(f"  Mean tokens: {summary['evidence_tokens']['mean']:.0f}")
    print(f"  Min / Max: {summary['evidence_tokens']['min']} / {summary['evidence_tokens']['max']}")
    print()
    
    # Decision gate
    oracle_ceiling = summary['oracle_ceiling']
    if oracle_ceiling is not None and oracle_ceiling >= 0.40:
        print("DECISION GATE PASSED")
        print(f"  Oracle ceiling ({oracle_ceiling:.0%}) >= 40%")
        print(f"  Evidence is confirmed as the bottleneck.")
        print(f"  PROCEED to Phase 2: Structured Metric Ingestion")
    elif oracle_ceiling is not None and oracle_ceiling < 0.40:
        print("DECISION GATE FAILED")
        print(f"  Oracle ceiling ({oracle_ceiling:.0%}) < 40%")
        print(f"  Generators/scoring are the limiting factor.")
        print(f"  PIVOT to generator fine-tuning or model swap (out of scope)")
    else:
        print("DECISION GATE INCONCLUSIVE")
        print(f"  Could not compute oracle ceiling.")
    
    print("="*80)
    print()
    
    # Write results
    output_path = _project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_output = {
        "summary": summary,
        "questions": results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, default=str)
    
    print(f"Results written to: {output_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle Evidence Test - Phase 1 Decision Gate")
    parser.add_argument(
        "--limit", type=int, default=30,
        help="Number of questions to test (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/results/oracle_evidence_test.json",
        help="Output file for results"
    )
    args = parser.parse_args()
    
    run_oracle_evidence_test(limit=args.limit, output_path=args.output)
