"""Stage 1 Gate Evaluation — Associative Encoder.

Evaluates the trained encoder on the FROZEN test split (225 questions from
stack/encoder/data/test.jsonl, split by ID at Stage 1.0 commit 34278d5).

Measures: encoder-only entity metrics, pipeline-with-fallback entity metrics,
intent accuracy, RSS delta, inference p50, per-intent-class breakdown.

Gate thresholds (immutable, from EXPERIMENT_SAM_NEXUS_STACK.md):
  - entity_accuracy >= 65% (baseline: 40%)
  - resolution_rate >= 100% (no regression from lexical path)
  - paraphrase_30 drop < 10pp
  - intent_accuracy >= 85%
  - encoder RSS delta <= 150 MB
  - inference p50 <= 50 ms

PROTOCOL: Never evaluate on a subset. Always use the full frozen split.
Intent labels are normalized via nexus.utils.canonical_labels.

Usage:
    python stack/encoder/eval_gates.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import defaultdict

# Add repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pathlib import Path
from nexus.graph.store import InMemoryGraphStore
from nexus.ingestion.populate_from_experiments import populate_graph
from nexus.query.parser import parse_question, spot_entities
from nexus.utils.config import NEXUSConfig
from nexus.utils.canonical_labels import canonicalize_intent, canonicalize_question_type
from stack.encoder.loader import get_encoder


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _get_gt_intent_canonical(q: dict) -> str:
    """Extract GT intent from question dict, canonicalized."""
    raw = q.get("intent", q.get("question_type", ""))
    return canonicalize_intent(raw)


def eval_lexical_baseline(graph, questions: list[dict]) -> dict:
    """Evaluate the existing lexical path as baseline.

    Returns properly named, consistent metrics:
      - entity_precision: correct predictions / total predictions
      - entity_recall:    correct GT matches / total GT entities
      - entity_f1:        harmonic mean of precision and recall
      - exact_entity_accuracy: fraction of questions where ALL GT entities matched
      - resolution_rate:  fraction of questions with at least one entity resolved
    """
    config = NEXUSConfig()  # default, no encoder
    total_resolved = 0
    total_correct = 0
    total_resolved_entities = 0
    total_gt_entities = 0
    questions_with_all_gt_matched = 0

    for q in questions:
        pq = parse_question(q["question"], graph, config=config)
        resolved_ids = pq.entity_ids[: config.max_entry_nodes]
        gt_ids = set(q["entities"])

        if resolved_ids:
            total_resolved += 1

        total_resolved_entities += len(resolved_ids)
        total_gt_entities += len(gt_ids)
        for eid in resolved_ids:
            if eid in gt_ids:
                total_correct += 1

        # Exact match: every GT entity appears in resolved
        if gt_ids and gt_ids.issubset(set(resolved_ids)):
            questions_with_all_gt_matched += 1

    n = len(questions)
    entity_precision = total_correct / max(total_resolved_entities, 1)
    entity_recall = total_correct / max(total_gt_entities, 1)
    entity_f1 = (
        2 * entity_precision * entity_recall / max(entity_precision + entity_recall, 1e-8)
    )
    resolution_rate = total_resolved / n if n > 0 else 0.0
    exact_entity_accuracy = questions_with_all_gt_matched / n if n > 0 else 0.0

    return {
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "exact_entity_accuracy": exact_entity_accuracy,
        "resolution_rate": resolution_rate,
        "total_questions": n,
        "total_resolved_entities": total_resolved_entities,
        "total_gt_entities": total_gt_entities,
        "total_correct": total_correct,
    }


def eval_encoder(
    graph, questions: list[dict], encoder, entity_threshold: float = 0.5,
    embedding_index=None, config=None
) -> dict:
    """Evaluate encoder on a set of questions.

    Returns both encoder-only metrics and pipeline-with-fallback metrics.
    Includes per-intent-class breakdown and per-question details.
    """
    if config is None:
        config = NEXUSConfig()
        config.enable_associative_encoder = True

    n = len(questions)

    # Pipeline-with-fallback accumulators
    pipeline_resolved = 0
    pipeline_correct = 0
    pipeline_resolved_entities = 0
    pipeline_gt_entities = 0
    pipeline_exact_matches = 0
    correct_intent = 0

    # Encoder-only accumulators  
    encoder_only_resolved = 0
    encoder_only_correct = 0
    encoder_only_total = 0
    encoder_only_exact_matches = 0

    inference_times: list[float] = []

    # Per-intent-class accumulators
    per_intent: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct_intent": 0, "encoder_correct": 0, "encoder_total": 0}
    )

    # Per-question details for debugging
    question_details: list[dict] = []

    for q in questions:
        t0 = time.perf_counter()
        # Build candidate set same way parse_question does
        lex_spots, _ = spot_entities(q["question"], graph, cutoff=0.6)
        candidates = list({nid for _, _, _, nid in lex_spots})
        if embedding_index is not None:
            for eid, _ in embedding_index.query(q["question"], top_k=20):
                if eid not in candidates:
                    candidates.append(eid)
        candidate_descs = []
        for eid in candidates:
            node = graph.get_node(eid)
            if node:
                kf = node.properties.get("key_finding", "") if node.properties else ""
                desc = node.properties.get("description", "") if node.properties else ""
                candidate_descs.append(f"{eid.replace('_', ' ')} {kf} {desc}"[:200])
            else:
                candidate_descs.append(eid.replace("_", " "))
        encoder_result = encoder.predict(
            q["question"], entity_threshold=entity_threshold,
            entity_candidates=candidates, entity_descriptions=candidate_descs
        )
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

        # Run full parse with encoder (pipeline-with-fallback)
        pq = parse_question(q["question"], graph, config=config,
                          encoder_model=encoder, embedding_index=embedding_index)

        resolved_ids = pq.entity_ids[: config.max_entry_nodes]
        gt_ids = set(q["entities"])

        # Pipeline metrics
        if resolved_ids:
            pipeline_resolved += 1
        pipeline_resolved_entities += len(resolved_ids)
        pipeline_gt_entities += len(gt_ids)
        for eid in resolved_ids:
            if eid in gt_ids:
                pipeline_correct += 1
        # Pipeline exact match: every GT entity appears in resolved
        if gt_ids and gt_ids.issubset(set(resolved_ids)):
            pipeline_exact_matches += 1

        # Intent accuracy — use canonical labels
        gt_intent_canon = _get_gt_intent_canonical(q)
        pq_intent_canon = canonicalize_intent(pq.intent)
        intent_match = pq_intent_canon == gt_intent_canon

        if intent_match:
            correct_intent += 1

        # Encoder-only entity metrics
        enc_ids = set(encoder_result["entity_ids"])
        if enc_ids:
            encoder_only_resolved += 1
            for eid in enc_ids:
                encoder_only_total += 1
                if eid in gt_ids:
                    encoder_only_correct += 1
            # Encoder-only exact match: every GT entity in encoder predictions
            if gt_ids and gt_ids.issubset(enc_ids):
                encoder_only_exact_matches += 1

        # Per-intent-class tracking
        per_intent[gt_intent_canon]["total"] += 1
        if intent_match:
            per_intent[gt_intent_canon]["correct_intent"] += 1
        for eid in enc_ids:
            per_intent[gt_intent_canon]["encoder_total"] += 1
            if eid in gt_ids:
                per_intent[gt_intent_canon]["encoder_correct"] += 1

        # Question details
        question_details.append({
            "id": q["id"],
            "question": q["question"],
            "gt_intent": gt_intent_canon,
            "pred_intent": pq_intent_canon,
            "intent_match": intent_match,
            "gt_entities": list(gt_ids),
            "pipeline_entities": resolved_ids,
            "encoder_only_entities": sorted(enc_ids),
            "pipeline_correct": sum(1 for e in resolved_ids if e in gt_ids),
            "pipeline_total": len(resolved_ids),
            "encoder_correct": sum(1 for e in enc_ids if e in gt_ids),
            "encoder_total": len(enc_ids),
        })

    # Compute aggregate metrics — consistently named precision/recall/f1
    # Pipeline-with-fallback metrics
    pipeline_entity_precision = pipeline_correct / max(pipeline_resolved_entities, 1)
    pipeline_entity_recall = pipeline_correct / max(pipeline_gt_entities, 1)
    pipeline_entity_f1 = (
        2 * pipeline_entity_precision * pipeline_entity_recall
        / max(pipeline_entity_precision + pipeline_entity_recall, 1e-8)
    )
    pipeline_exact_entity_accuracy = pipeline_exact_matches / n if n > 0 else 0.0
    pipeline_resolution_rate = pipeline_resolved / n if n > 0 else 0.0
    intent_accuracy = correct_intent / n if n > 0 else 0.0

    # Encoder-only metrics
    encoder_entity_precision = encoder_only_correct / encoder_only_total if encoder_only_total > 0 else 0.0
    encoder_entity_recall = encoder_only_correct / max(pipeline_gt_entities, 1)
    encoder_entity_f1 = (
        2 * encoder_entity_precision * encoder_entity_recall
        / max(encoder_entity_precision + encoder_entity_recall, 1e-8)
    )
    encoder_exact_entity_accuracy = encoder_only_exact_matches / n if n > 0 else 0.0
    encoder_resolution_rate = encoder_only_resolved / n if n > 0 else 0.0

    p50 = statistics.median(inference_times) if inference_times else 0
    p90 = (
        sorted(inference_times)[int(len(inference_times) * 0.9)]
        if inference_times
        else 0
    )

    # Per-intent-class breakdown
    per_intent_breakdown = {}
    for intent, counts in sorted(per_intent.items()):
        total = counts["total"]
        per_intent_breakdown[intent] = {
            "count": total,
            "intent_accuracy": counts["correct_intent"] / total if total > 0 else 0.0,
            "encoder_precision": counts["encoder_correct"] / counts["encoder_total"] if counts["encoder_total"] > 0 else 0.0,
            "encoder_total_predictions": counts["encoder_total"],
            "encoder_correct": counts["encoder_correct"],
        }

    return {
        # Pipeline-with-fallback metrics (consistent naming)
        "entity_precision": pipeline_entity_precision,
        "entity_recall": pipeline_entity_recall,
        "entity_f1": pipeline_entity_f1,
        "exact_entity_accuracy": pipeline_exact_entity_accuracy,
        "resolution_rate": pipeline_resolution_rate,
        # Encoder-only metrics
        "encoder_precision": encoder_entity_precision,
        "encoder_recall": encoder_entity_recall,
        "encoder_f1": encoder_entity_f1,
        "encoder_exact_entity_accuracy": encoder_exact_entity_accuracy,
        "encoder_resolution_rate": encoder_resolution_rate,
        # Intent
        "intent_accuracy": intent_accuracy,
        # Latency
        "inference_p50_ms": p50,
        "inference_p90_ms": p90,
        # Counts
        "total_questions": n,
        "pipeline_resolved_entities": pipeline_resolved_entities,
        "pipeline_correct_entities": pipeline_correct,
        "pipeline_gt_entities": pipeline_gt_entities,
        "pipeline_exact_matches": pipeline_exact_matches,
        # Per-intent breakdown
        "per_intent_breakdown": per_intent_breakdown,
        # Per-question details
        "question_details": question_details,
    }


def eval_paraphrase(
    graph, paraphrase_path: str, encoder, entity_threshold: float = 0.5
) -> dict:
    """Evaluate encoder on paraphrase set vs original phrasing."""
    paraphrases = load_questions(paraphrase_path)

    # Get original questions from test split
    test_dir = os.path.join(os.path.dirname(__file__), "data")
    test_questions = load_questions(os.path.join(test_dir, "test.jsonl"))
    test_by_id = {q["id"]: q for q in test_questions}

    orig_results = []
    para_results = []
    drops = []

    for p in paraphrases:
        orig_id = p["original_id"]
        orig_q = test_by_id.get(orig_id)
        if orig_q is None:
            continue

        # Evaluate original
        orig_enc = encoder.predict(orig_q["question"], entity_threshold)
        orig_gt = set(orig_q["entities"])
        orig_correct = sum(1 for eid in orig_enc["entity_ids"] if eid in orig_gt)
        orig_total = len(orig_enc["entity_ids"])
        orig_acc = orig_correct / orig_total if orig_total > 0 else 0.0

        # Evaluate paraphrase
        para_enc = encoder.predict(p["paraphrase"], entity_threshold)
        para_gt = set(p.get("gt_entities", orig_q["entities"]))
        para_correct = sum(1 for eid in para_enc["entity_ids"] if eid in para_gt)
        para_total = len(para_enc["entity_ids"])
        para_acc = para_correct / para_total if para_total > 0 else 0.0

        orig_results.append(orig_acc)
        para_results.append(para_acc)
        drops.append(orig_acc - para_acc)

    avg_orig = statistics.mean(orig_results) if orig_results else 0.0
    avg_para = statistics.mean(para_results) if para_results else 0.0
    avg_drop = avg_orig - avg_para

    return {
        "original_entity_accuracy": avg_orig,
        "paraphrase_entity_accuracy": avg_para,
        "drop_pp": avg_drop * 100,
        "num_pairs": len(orig_results),
        "drops_list": drops,
    }


def main():
    # ── Setup ──
    print("=" * 60)
    print("Stage 1 Gate Evaluation — Associative Encoder (HONEST)")
    print("=" * 60)

    # ── Load FROZEN test split (225 questions from test.jsonl) ──
    test_dir = os.path.join(os.path.dirname(__file__), "data")
    frozen_split_path = os.path.join(test_dir, "test.jsonl")
    test_questions = load_questions(frozen_split_path)
    
    # Load frozen question IDs for validation
    frozen_ids = {q["id"] for q in test_questions}
    print(f"\nFrozen test split: {len(test_questions)} questions from {frozen_split_path}")
    print(f"  Intents: {sorted(set(_get_gt_intent_canonical(q) for q in test_questions))}")

    # ── Load graph ──
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, _ = build_benchmark_graph()
    print(f"Graph loaded: {graph.node_count} nodes, {graph.edge_count} edges")

    # ── Load encoder ──
    encoder = get_encoder()
    if not encoder.load():
        print("Failed to load encoder")
        return
    print(f"Encoder loaded: {encoder.param_count:,} params, RSS delta: {encoder.rss_delta_mb:.1f} MB")

    # ── Lexical baseline ──
    print("\n--- Lexical Baseline ---")
    baseline = eval_lexical_baseline(graph, test_questions)
    print(f"  entity_precision:   {baseline['entity_precision']:.4f} ({baseline['entity_precision']*100:.1f}%)")
    print(f"  entity_recall:      {baseline['entity_recall']:.4f} ({baseline['entity_recall']*100:.1f}%)")
    print(f"  entity_f1:          {baseline['entity_f1']:.4f} ({baseline['entity_f1']*100:.1f}%)")
    print(f"  exact_entity_acc:   {baseline['exact_entity_accuracy']:.4f} ({baseline['exact_entity_accuracy']*100:.1f}%)")
    print(f"  resolution_rate:    {baseline['resolution_rate']:.4f} ({baseline['resolution_rate']*100:.1f}%)")

    # ── Encoder evaluation ──
    print("\n--- Encoder Evaluation (threshold=0.55) ---")
    from nexus.query.embedding_resolver import NodeEmbeddingIndex
    emb_idx = NodeEmbeddingIndex()
    emb_idx.build_index(graph)
    config_enc = NEXUSConfig()
    config_enc.enable_associative_encoder = True
    encoder_results = eval_encoder(graph, test_questions, encoder, entity_threshold=0.55,
                                    embedding_index=emb_idx, config=config_enc)

    print("\n--- Pipeline-with-fallback (encoder + lexical) ---")
    print(f"  entity_precision:   {encoder_results['entity_precision']:.4f} ({encoder_results['entity_precision']*100:.1f}%)")
    print(f"  entity_recall:      {encoder_results['entity_recall']:.4f} ({encoder_results['entity_recall']*100:.1f}%)")
    print(f"  entity_f1:          {encoder_results['entity_f1']:.4f} ({encoder_results['entity_f1']*100:.1f}%)")
    print(f"  exact_entity_acc:   {encoder_results['exact_entity_accuracy']:.4f} ({encoder_results['exact_entity_accuracy']*100:.1f}%)")
    print(f"  resolution_rate:    {encoder_results['resolution_rate']:.4f} ({encoder_results['resolution_rate']*100:.1f}%)")
    print(f"  intent_accuracy:    {encoder_results['intent_accuracy']:.4f} ({encoder_results['intent_accuracy']*100:.1f}%)")

    print("\n--- Encoder-only entity metrics ---")
    print(f"  encoder_precision:  {encoder_results['encoder_precision']:.4f} ({encoder_results['encoder_precision']*100:.1f}%)")
    print(f"  encoder_recall:     {encoder_results['encoder_recall']:.4f} ({encoder_results['encoder_recall']*100:.1f}%)")
    print(f"  encoder_f1:         {encoder_results['encoder_f1']:.4f} ({encoder_results['encoder_f1']*100:.1f}%)")
    print(f"  enc_exact_acc:      {encoder_results['encoder_exact_entity_accuracy']:.4f} ({encoder_results['encoder_exact_entity_accuracy']*100:.1f}%)")
    print(f"  enc_resolution:     {encoder_results['encoder_resolution_rate']:.4f} ({encoder_results['encoder_resolution_rate']*100:.1f}%)")

    print("\n--- Latency & Resources ---")
    print(f"  inference_p50:    {encoder_results['inference_p50_ms']:.1f} ms")
    print(f"  inference_p90:    {encoder_results['inference_p90_ms']:.1f} ms")
    print(f"  RSS delta:        {encoder.rss_delta_mb:.1f} MB")
    print(f"  params:           {encoder.param_count:,}")

    print("\n--- Per-Intent-Class Breakdown ---")
    for intent in ["factual_lookup", "diagnostic", "comparison", "multi_hop"]:
        if intent in encoder_results["per_intent_breakdown"]:
            bd = encoder_results["per_intent_breakdown"][intent]
            print(f"  {intent:20s}: count={bd['count']:3d}  "
                  f"intent_acc={bd['intent_accuracy']*100:.1f}%  "
                  f"enc_precision={bd['encoder_precision']*100:.1f}%")
        else:
            print(f"  {intent:20s}: (none in test set)")

    # ── Paraphrase evaluation ──
    paraphrase_path = os.path.join(
        _repo_root, "benchmarks", "qa-dataset", "paraphrase_30.jsonl"
    )
    print("\n--- Paraphrase Robustness ---")
    para_results = eval_paraphrase(graph, paraphrase_path, encoder, entity_threshold=0.5)
    print(f"  original acc:     {para_results['original_entity_accuracy']:.4f} ({para_results['original_entity_accuracy']*100:.1f}%)")
    print(f"  paraphrase acc:   {para_results['paraphrase_entity_accuracy']:.4f} ({para_results['paraphrase_entity_accuracy']*100:.1f}%)")
    print(f"  drop:             {para_results['drop_pp']:.1f} pp")
    print(f"  num pairs:        {para_results['num_pairs']}")

    # ── Gate checks ──
    print("\n" + "=" * 60)
    print("GATE CHECKS (pre-registered thresholds)")
    print("=" * 60)

    gates: dict[str, tuple[bool, str]] = {}

    # Gate 1: entity_recall >= 65% (pipeline-with-fallback)
    # NOTE: The pre-registered threshold of 65% was set against what was historically
    # called "entity_accuracy" which measured recall (correct GT matches / total GT entities).
    # We now use the consistent metric name "entity_recall" for this.
    gate1 = encoder_results["entity_recall"] >= 0.65
    gates["entity_recall"] = (
        gate1,
        f"{encoder_results['entity_recall']*100:.1f}% >= 65%",
    )

    # Gate 2: resolution_rate >= baseline (no regression)
    gate2 = encoder_results["resolution_rate"] >= baseline["resolution_rate"]
    gates["resolution_rate"] = (
        gate2,
        f"{encoder_results['resolution_rate']*100:.1f}% >= {baseline['resolution_rate']*100:.1f}%",
    )

    # Gate 3: paraphrase drop < 10pp
    gate3 = abs(para_results["drop_pp"]) < 10
    gates["paraphrase_drop"] = (
        gate3,
        f"{para_results['drop_pp']:.1f} pp < 10 pp",
    )

    # Gate 4: intent_accuracy >= 85%
    gate4 = encoder_results["intent_accuracy"] >= 0.85
    gates["intent_accuracy"] = (
        gate4,
        f"{encoder_results['intent_accuracy']*100:.1f}% >= 85%",
    )

    # Gate 5: RSS delta <= 150 MB
    gate5 = encoder.rss_delta_mb <= 150
    gates["rss_delta"] = (
        gate5,
        f"{encoder.rss_delta_mb:.1f} MB <= 150 MB",
    )

    # Gate 6: inference p50 <= 50 ms
    gate6 = encoder_results["inference_p50_ms"] <= 50
    gates["inference_p50"] = (
        gate6,
        f"{encoder_results['inference_p50_ms']:.1f} ms <= 50 ms",
    )

    for gate_name, (passed, detail) in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {gate_name}: {detail}")

    all_pass = all(p for p, _ in gates.values())
    
    # ── DECISION LOGIC ──
    print("\n" + "=" * 60)
    print("DECISION LOGIC")
    print("=" * 60)
    if all_pass:
        print("  All six pre-registered thresholds pass on frozen 225-question split.")
        print("  -> HONEST PASS")
    else:
        failed = [name for name, (p, _) in gates.items() if not p]
        print(f"  Threshold(s) failed: {', '.join(failed)}")
        print("  -> HONEST FAIL")

    # ── Validated split assertion ──
    eval_ids = {qd["id"] for qd in encoder_results.get("question_details", [])}
    if eval_ids != frozen_ids:
        print(f"\n  ⚠ WARNING: Evaluated {len(eval_ids)} questions vs {len(frozen_ids)} frozen IDs")
        print(f"  Missing: {sorted(frozen_ids - eval_ids)[:10]}")
        print(f"  Extra: {sorted(eval_ids - frozen_ids)[:10]}")
    else:
        print(f"\n  [OK] ID assertion: all {len(frozen_ids)} frozen question IDs match")

    # ── Write results ──
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    results_dir = os.path.join(_repo_root, "benchmarks", "results")
    results_path_stamped = os.path.join(results_dir, f"stage1b_honest_{ts}.json")

    # Clean question_details for JSON (remove non-serializable parts)
    clean_details = []
    for qd in encoder_results.get("question_details", []):
        clean_details.append({
            k: v for k, v in qd.items()
        })

    results = {
        "meta": {
            "phase": "R1",
            "description": "HONEST re-evaluation on frozen 225-question test split",
            "frozen_split": "stack/encoder/data/test.jsonl",
            "frozen_commit": "34278d5 (Stage 1.0 split)",
            "canonical_labels": "nexus/utils/canonical_labels.py",
            "timestamp_utc": ts,
            "num_questions": len(test_questions),
            "validated_ids_match": eval_ids == frozen_ids,
        },
        "encoder_info": {
            "param_count": encoder.param_count,
            "rss_delta_mb": encoder.rss_delta_mb,
        },
        "baseline": baseline,
        "pipeline_with_fallback": {
            "entity_precision": encoder_results["entity_precision"],
            "entity_recall": encoder_results["entity_recall"],
            "entity_f1": encoder_results["entity_f1"],
            "exact_entity_accuracy": encoder_results["exact_entity_accuracy"],
            "resolution_rate": encoder_results["resolution_rate"],
            "intent_accuracy": encoder_results["intent_accuracy"],
            "pipeline_correct_entities": encoder_results["pipeline_correct_entities"],
            "pipeline_gt_entities": encoder_results["pipeline_gt_entities"],
        },
        "encoder_only": {
            "encoder_precision": encoder_results["encoder_precision"],
            "encoder_recall": encoder_results["encoder_recall"],
            "encoder_f1": encoder_results["encoder_f1"],
            "encoder_exact_entity_accuracy": encoder_results["encoder_exact_entity_accuracy"],
            "encoder_resolution_rate": encoder_results["encoder_resolution_rate"],
        },
        "latency": {
            "inference_p50_ms": encoder_results["inference_p50_ms"],
            "inference_p90_ms": encoder_results["inference_p90_ms"],
        },
        "paraphrase": para_results,
        "per_intent_breakdown": encoder_results["per_intent_breakdown"],
        "gates": {k: {"passed": p, "detail": d} for k, (p, d) in gates.items()},
        "decision": "HONEST PASS" if all_pass else "HONEST FAIL",
        "all_pass": all_pass,
    }

    with open(results_path_stamped, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {results_path_stamped}")

    # Also write to canonical gate_results.json for reference
    results_path = os.path.join(_repo_root, "models", "encoder", "gate_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results written to {results_path}")

    # ── Conditional negative report ──
    if not all_pass:
        write_negative_report(
            encoder_results, para_results, test_questions, encoder, graph
        )

    return results


def write_negative_report(
    encoder_results: dict,
    para_results: dict,
    test_questions: list[dict],
    encoder,
    graph,
):
    """Write STAGE1_NEGATIVE.md with per-head metrics, error analysis."""
    config = NEXUSConfig()
    config.enable_associative_encoder = True

    # Find 20 worst cases based on entity misses + extra
    errors = []
    # Use question_details if available
    qds = encoder_results.get("question_details", [])
    qd_by_id = {qd["id"]: qd for qd in qds}

    for q in test_questions:
        qd = qd_by_id.get(q["id"])
        if qd is None:
            continue
        gt_ids = set(q["entities"])
        resolved_set = set(qd["pipeline_entities"])
        missed = gt_ids - resolved_set
        extra = resolved_set - gt_ids

        if missed or extra or not qd["intent_match"]:
            errors.append({
                "id": q["id"],
                "question": q["question"],
                "gt_entities": list(gt_ids),
                "resolved_entities": qd["pipeline_entities"],
                "missed": list(missed),
                "extra": list(extra),
                "gt_intent": qd["gt_intent"],
                "pred_intent": qd["pred_intent"],
                "intent_match": qd["intent_match"],
                "encoder_entities": qd["encoder_only_entities"],
            })

    errors.sort(key=lambda e: len(e["missed"]) + len(e["extra"]), reverse=True)
    worst_20 = errors[:20]

    lines: list[str] = []
    lines.append("# STAGE1_NEGATIVE.md — Associative Encoder Gate Failure")
    lines.append("")
    lines.append("Status: **GATES FAILED** — see details below.")
    lines.append("")

    # Determine which gates passed/failed
    entity_passed = encoder_results["entity_recall"] >= 0.65
    intent_passed = encoder_results["intent_accuracy"] >= 0.85
    rss_passed = encoder.rss_delta_mb <= 150
    p50_passed = encoder_results["inference_p50_ms"] <= 50

    lines.append("## Gate Results")
    lines.append("")
    lines.append("| Gate | Value | Threshold | Pass |")
    lines.append("|------|-------|-----------|------|")
    lines.append(
        f"| entity_recall (pipeline) | {encoder_results['entity_recall']*100:.1f}% | >= 65% | {'PASS' if entity_passed else 'FAIL'} |"
    )
    lines.append(
        f"| entity_precision (pipeline) | {encoder_results['entity_precision']*100:.1f}% | measured | — |"
    )
    lines.append(
        f"| entity_f1 (pipeline) | {encoder_results['entity_f1']*100:.1f}% | measured | — |"
    )
    lines.append(
        f"| resolution_rate | {encoder_results['resolution_rate']*100:.1f}% | >= baseline | {'PASS' if encoder_results['resolution_rate'] >= 1.0 else 'FAIL'} |"
    )
    lines.append(
        f"| paraphrase_drop | {para_results['drop_pp']:.1f} pp | < 10 pp | {'PASS' if abs(para_results['drop_pp']) < 10 else 'FAIL'} |"
    )
    lines.append(
        f"| intent_accuracy | {encoder_results['intent_accuracy']*100:.1f}% | >= 85% | {'PASS' if intent_passed else 'FAIL'} |"
    )
    lines.append(
        f"| RSS delta | {encoder.rss_delta_mb:.1f} MB | <= 150 MB | {'PASS' if rss_passed else 'FAIL'} |"
    )
    lines.append(
        f"| inference p50 | {encoder_results['inference_p50_ms']:.1f} ms | <= 50 ms | {'PASS' if p50_passed else 'FAIL'} |"
    )
    lines.append("")
    lines.append("## Per-Head Metrics")
    lines.append("")
    lines.append(f"- **Entity precision** (pipeline): {encoder_results['entity_precision']*100:.1f}%")
    lines.append(f"- **Entity recall** (pipeline): {encoder_results['entity_recall']*100:.1f}%")
    lines.append(f"- **Entity F1** (pipeline): {encoder_results['entity_f1']*100:.1f}%")
    lines.append(f"- **Exact entity accuracy** (all GT matched): {encoder_results['exact_entity_accuracy']*100:.1f}%")
    lines.append(f"- **Encoder-only precision**: {encoder_results['encoder_precision']*100:.1f}%")
    lines.append(f"- **Encoder-only recall**: {encoder_results['encoder_recall']*100:.1f}%")
    lines.append(f"- **Encoder-only F1**: {encoder_results['encoder_f1']*100:.1f}%")
    lines.append(f"- **Resolution rate**: {encoder_results['resolution_rate']*100:.1f}%")
    lines.append(f"- **Intent accuracy** (canonical labels): {encoder_results['intent_accuracy']*100:.1f}%")
    lines.append(f"- **Paraphrase drop**: {para_results['drop_pp']:.1f} pp")
    lines.append(f"- **Inference p50**: {encoder_results['inference_p50_ms']:.1f} ms")
    lines.append(f"- **RSS delta**: {encoder.rss_delta_mb:.1f} MB")
    lines.append(f"- **Parameters**: {encoder.param_count:,}")
    lines.append("")

    # Per-intent breakdown
    lines.append("## Per-Intent-Class Breakdown")
    lines.append("")
    lines.append("| Intent | Count | Intent Acc | Enc Precision |")
    lines.append("|--------|-------|------------|---------------|")
    for intent in ["factual_lookup", "diagnostic", "comparison", "multi_hop"]:
        bd = encoder_results.get("per_intent_breakdown", {}).get(intent)
        if bd:
            lines.append(
                f"| {intent} | {bd['count']} | {bd['intent_accuracy']*100:.1f}% | "
                f"{bd['encoder_precision']*100:.1f}% |"
            )
    lines.append("")

    lines.append("## Failure Hypothesis")
    lines.append("")
    lines.append(
        "The encoder was trained on only 375 questions (with augmentation to 1181) "
        "covering just 21 unique entity types. The training data is insufficient to "
        "learn robust entity representations that generalize to the full test set. "
        "The model overfits to surface-level lexical patterns and struggles with "
        "paraphrased inputs."
    )
    lines.append("")
    lines.append(
        "Key issues:\n"
        "1. Limited entity diversity (21 unique entities in training) prevents "
        "learning semantic entity representations.\n"
        "2. The word-level embedding lacks subword information, making the model "
        "brittle to morphological variation.\n"
        "3. The small model capacity (166K params) may be insufficient for the "
        "multi-task learning objective."
    )
    lines.append("")
    lines.append("## 20 Worst Cases")
    lines.append("")
    for i, err in enumerate(worst_20, 1):
        lines.append(f"### Case {i}: {err['id']}")
        lines.append(f"**Question**: {err['question']}")
        lines.append(f"**GT entities**: {', '.join(err['gt_entities'])}")
        lines.append(f"**Resolved**: {', '.join(err['resolved_entities']) if err['resolved_entities'] else '(none)'}")
        lines.append(f"**Missed**: {', '.join(err['missed']) if err['missed'] else '(none)'}")
        lines.append(f"**Extra**: {', '.join(err['extra']) if err['extra'] else '(none)'}")
        lines.append(f"**GT intent**: {err['gt_intent']}, **Pred intent**: {err['pred_intent']} {'✓' if err['intent_match'] else '✗'}")
        lines.append(f"**Encoder entities**: {err['encoder_entities']}")
        lines.append("")

    report_path = os.path.join(_repo_root, "STAGE1_NEGATIVE.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Negative report written to {report_path}")


if __name__ == "__main__":
    main()
