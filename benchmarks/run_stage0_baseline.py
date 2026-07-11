"""Stage 0 — Honest NEXUS vs RAG baseline comparison.

Both arms receive identical questions, use the same scoring,
and produce complete per-question serialized results.
The generated artifact is self-validating via a publication guard.

Uses lexical-only NEXUS (no learned components).  Does not read
the consumed frozen split.

Usage:
    python benchmarks/run_stage0_baseline.py --limit 30 --output benchmarks/results/stage0_<TS>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner

from benchmarks.compare_arms import compare_paired
from benchmarks.scoring import compute_fact_score


def load_questions(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit and limit > 0:
        questions = questions[:limit]
    return questions


def run_rag_baseline(questions: list[dict], model_name: str = "qwen2.5:latest") -> list[dict]:
    """Run RAG baseline using the existing rag_baseline module."""
    from benchmarks.rag_baseline import initialize_rag_pipeline, run_rag_pipeline, initialize_nexus_model

    print("  Initializing RAG pipeline...")
    try:
        rag_state = initialize_rag_pipeline()
        model, model_name = initialize_nexus_model()
    except Exception as exc:
        print(f"  WARNING: RAG initialization failed: {exc}")
        return [{"rag_answer": "", "rag_error": f"init_failed:{exc}", "rag_latency_ms": 0} for _ in questions]

    results = []
    for i, q in enumerate(questions):
        question = q["question"]
        try:
            rag_result = run_rag_pipeline(question, rag_state, model)
            results.append({
                "rag_answer": rag_result.get("answer", ""),
                "rag_chunks": rag_result.get("retrieved_chunks", [])[:3],
                "rag_latency_ms": rag_result.get("rag_time_ms", 0),
                "rag_evidence": rag_result.get("evidence", {}),
            })
        except Exception as exc:
            results.append({
                "rag_answer": "",
                "rag_error": str(exc),
                "rag_latency_ms": 0,
            })
        if (i + 1) % 10 == 0:
            print(f"  RAG: {i + 1}/{len(questions)}")
    return results


def build_graph() -> InMemoryGraphStore:
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, _ = build_benchmark_graph()
    return graph


def run_stage0(
    questions: list[dict],
    output_path: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the honest Stage 0 NEXUS vs RAG baseline."""
    questions = questions[:limit] if limit else questions
    print(f"Stage 0 baseline: {len(questions)} questions")

    utc_now = datetime.now(timezone.utc)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    # ── Build graph ──
    print("Building graph...")
    graph = build_graph()
    graph_meta = {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
    }

    # ── NEXUS arm ──
    print("Running NEXUS arm (lexical-only)...")
    config = ProductionNEXUSConfig.lexical_only()
    nexus_runner = NEXUSRunner(graph, config)
    nexus_result = nexus_runner.run(questions, source_sha=source_sha)

    # ── RAG arm ──
    print("Running RAG arm...")
    rag_results = run_rag_baseline(questions)

    # ── Score both arms ──
    nexus_scores: list[float | None] = []
    rag_scores: list[float | None] = []
    per_question: list[dict] = []

    for i, q in enumerate(questions):
        ground_truth = q.get("answer", q.get("entities", ""))
        if isinstance(ground_truth, list):
            ground_truth = ", ".join(str(e) for e in ground_truth)

        nexus_ans = nexus_result.per_question[i].answer if i < len(nexus_result.per_question) else ""
        rag_ans = rag_results[i].get("rag_answer", "") if i < len(rag_results) else ""

        ns = compute_fact_score(nexus_ans or "", str(ground_truth))
        rs = compute_fact_score(rag_ans or "", str(ground_truth))

        nexus_scores.append(ns.get("fuzzy_accuracy", 0.0))
        rag_scores.append(rs.get("fuzzy_accuracy", 0.0))

        per_question.append({
            "question_id": q.get("id", str(i)),
            "question": q["question"][:200],
            "ground_truth": str(ground_truth)[:200],
            "nexus_answer": nexus_ans[:500],
            "rag_answer": rag_ans[:500],
            "nexus_accuracy": ns.get("fuzzy_accuracy", 0.0),
            "rag_accuracy": rs.get("fuzzy_accuracy", 0.0),
            "nexus_parsed_intent": nexus_result.per_question[i].parsed_intent if i < len(nexus_result.per_question) else "",
            "nexus_failure": nexus_result.per_question[i].failure_category if i < len(nexus_result.per_question) else "",
        })

    # ── Paired comparison ──
    comparison = compare_paired(nexus_scores, rag_scores, "NEXUS", "RAG")

    # ── Build artifact ──
    out_path = Path(output_path)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {out_path}")

    train_path = _project_root / "stack" / "encoder" / "data" / "train.jsonl"
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()

    artifact = {
        "stage": "stage0_baseline",
        "created_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "pipeline": "nexus_v1_lexical_only",
        "config_hash": config.config_hash,
        "question_source": "stack/encoder/data/train.jsonl",
        "question_source_sha256": train_hash,
        "questions_total": len(questions),
        "graph": graph_meta,
        "nexus": {
            "answered": len([s for s in nexus_scores if s is not None]),
            "mean_accuracy": round(sum(s for s in nexus_scores if s is not None) / max(1, len([s for s in nexus_scores if s is not None])), 4),
        },
        "rag": {
            "answered": len([s for s in rag_scores if s is not None]),
            "mean_accuracy": round(sum(s for s in rag_scores if s is not None) / max(1, len([s for s in rag_scores if s is not None])), 4),
        },
        "paired_comparison": comparison,
        "per_question": per_question,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.rename(out_path)

    print(f"\nArtifact: {out_path}")
    print(f"NEXUS accuracy: {artifact['nexus']['mean_accuracy']:.4f}")
    print(f"RAG accuracy: {artifact['rag']['mean_accuracy']:.4f}")
    print(f"Paired N: {comparison['paired_n']}")
    print(f"W/L/T (NEXUS vs RAG): {comparison['win_loss_tie']}")
    print(f"Sign test p: {comparison['sign_test_p']}")

    return artifact


def validate_artifact(artifact: dict) -> list[str]:
    """Publication guard — returns list of errors (empty = valid)."""
    errors = []

    per_q = artifact.get("per_question", [])

    # Both arms must have non-empty results
    nexus_answered = sum(
        1 for pq in per_q
        if pq.get("nexus_answer") and len(pq["nexus_answer"].strip()) >= 10
    )
    rag_answered = sum(
        1 for pq in per_q
        if pq.get("rag_answer") and len(pq["rag_answer"].strip()) >= 10
    )
    rag_errors = sum(1 for pq in per_q if pq.get("rag_error"))
    nexus_errors = sum(1 for pq in per_q if pq.get("nexus_error"))

    if nexus_answered == 0 and nexus_errors > 0:
        errors.append(f"NEXUS arm: 0 answered, {nexus_errors} errors")
    if rag_answered == 0 and rag_errors > 0:
        errors.append(f"RAG arm: 0 answered, {rag_errors} errors")
    if nexus_answered == 0 and nexus_errors == 0:
        errors.append("NEXUS arm has zero non-empty answers")
    if rag_answered == 0 and rag_errors == 0:
        errors.append("RAG arm has zero non-empty answers")

    # Same denominator
    if artifact["questions_total"] == 0:
        errors.append("Zero questions total")

    # Must have per_question data
    if not per_q:
        errors.append("Missing per_question data")

    # Must have source SHA
    if not artifact.get("source_sha"):
        errors.append("Missing source_sha")

    # Must have comparison with non-zero paired
    comp = artifact.get("paired_comparison", {})
    if not comp.get("paired_n"):
        errors.append("Paired comparison has zero paired questions")

    # Question ID match between arms
    nexus_ids = {pq.get("question_id") for pq in per_q if pq.get("nexus_answer")}
    rag_ids = {pq.get("question_id") for pq in per_q if pq.get("rag_answer")}
    if nexus_ids and rag_ids and nexus_ids != rag_ids:
        errors.append("NEXUS and RAG arms answered different question sets")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Stage 0 — Honest NEXUS vs RAG baseline")
    parser.add_argument("--limit", type=int, default=30, help="Number of questions")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--questions", default="stack/encoder/data/train.jsonl",
                        help="Question source (default: train.jsonl)")
    args = parser.parse_args()

    questions = load_questions(args.questions, args.limit)
    artifact = run_stage0(questions, args.output, args.limit)

    errors = validate_artifact(artifact)
    if errors:
        print("\n❌ PUBLICATION GUARD FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ Publication guard passed.")


if __name__ == "__main__":
    main()
