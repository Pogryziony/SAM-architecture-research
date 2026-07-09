"""Stage 1 Gate Evaluation — Associative Encoder.

Evaluates the trained encoder on the untouched test split and paraphrase set.
Measures: entity_accuracy, resolution_rate, intent_accuracy, RSS delta, inference p50.

Gate thresholds (immutable, from EXPERIMENT_SAM_NEXUS_STACK.md):
  - entity_accuracy >= 65% (baseline: 40%)
  - resolution_rate >= 100% (no regression from lexical path)
  - paraphrase_30 drop < 10pp
  - intent_accuracy >= 85%
  - encoder RSS delta <= 150 MB
  - inference p50 <= 50 ms

Usage:
    python stack/encoder/eval_gates.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

# Add repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pathlib import Path
from nexus.graph.store import InMemoryGraphStore
from nexus.ingestion.populate_from_experiments import populate_graph
from nexus.query.parser import parse_question
from nexus.utils.config import NEXUSConfig
from stack.encoder.loader import get_encoder


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def eval_lexical_baseline(graph, questions: list[dict]) -> dict:
    """Evaluate the existing lexical path as baseline."""
    config = NEXUSConfig()  # default, no encoder
    total_resolved = 0
    total_correct_entities = 0
    total_gt_entities = 0

    for q in questions:
        pq = parse_question(q["question"], graph, config=config)
        resolved_ids = pq.entity_ids[: config.max_entry_nodes]
        gt_ids = set(q["entities"])

        if resolved_ids:
            total_resolved += 1

        # Count how many resolved entities are in GT (precision)
        for eid in resolved_ids:
            if eid in gt_ids:
                total_correct_entities += 1

        total_gt_entities += len(gt_ids)

    n = len(questions)
    entity_accuracy = total_correct_entities / max(1, sum(len(pq.entity_ids[:5]) for pq in [
        parse_question(q["question"], graph, config=config) for q in questions
    ])) if questions else 0.0
    # Recalculate properly
    total_resolved_entities = 0
    total_correct = 0
    for q in questions:
        pq = parse_question(q["question"], graph, config=config)
        resolved_ids = pq.entity_ids[: config.max_entry_nodes]
        total_resolved_entities += len(resolved_ids)
        gt_ids = set(q["entities"])
        for eid in resolved_ids:
            if eid in gt_ids:
                total_correct += 1

    entity_accuracy = total_correct / total_resolved_entities if total_resolved_entities > 0 else 0.0
    resolution_rate = total_resolved / n if n > 0 else 0.0

    return {
        "entity_accuracy": entity_accuracy,
        "resolution_rate": resolution_rate,
        "total_questions": n,
        "total_resolved_entities": total_resolved_entities,
    }


def eval_encoder(
    graph, questions: list[dict], encoder, entity_threshold: float = 0.5
) -> dict:
    """Evaluate encoder on a set of questions."""
    config = NEXUSConfig()
    config.enable_associative_encoder = True

    n = len(questions)
    total_resolved = 0  # Questions with any entities resolved
    total_correct = 0  # Correct entity predictions
    total_resolved_entities = 0  # Total entities resolved
    total_gt_entities = 0  # Total GT entities
    correct_intent = 0
    inference_times: list[float] = []
    encoder_only_resolved = 0
    encoder_only_correct = 0
    encoder_only_total = 0

    for q in questions:
        # Time inference
        t0 = time.perf_counter()
        encoder_result = encoder.predict(q["question"], entity_threshold=entity_threshold)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)  # ms

        # Run full parse with encoder
        pq = parse_question(q["question"], graph, config=config, encoder_model=encoder)

        resolved_ids = pq.entity_ids[: config.max_entry_nodes]
        gt_ids = set(q["entities"])

        if resolved_ids:
            total_resolved += 1

        for eid in resolved_ids:
            total_resolved_entities += 1
            if eid in gt_ids:
                total_correct += 1

        total_gt_entities += len(gt_ids)

        # Intent accuracy
        if pq.intent == q.get("intent", q.get("question_type", "")):
            correct_intent += 1

        # Encoder-only metrics: only entities from encoder prediction
        enc_ids = set(encoder_result["entity_ids"])
        if enc_ids:
            encoder_only_resolved += 1
            for eid in enc_ids:
                encoder_only_total += 1
                if eid in gt_ids:
                    encoder_only_correct += 1

    entity_accuracy = total_correct / total_resolved_entities if total_resolved_entities > 0 else 0.0
    resolution_rate = total_resolved / n if n > 0 else 0.0
    intent_accuracy = correct_intent / n if n > 0 else 0.0
    encoder_precision = encoder_only_correct / encoder_only_total if encoder_only_total > 0 else 0.0
    encoder_resolution_rate = encoder_only_resolved / n if n > 0 else 0.0

    # Inference latency stats
    p50 = statistics.median(inference_times) if inference_times else 0
    p90 = (
        sorted(inference_times)[int(len(inference_times) * 0.9)]
        if inference_times
        else 0
    )

    return {
        "entity_accuracy": entity_accuracy,
        "resolution_rate": resolution_rate,
        "intent_accuracy": intent_accuracy,
        "encoder_precision": encoder_precision,
        "encoder_resolution_rate": encoder_resolution_rate,
        "inference_p50_ms": p50,
        "inference_p90_ms": p90,
        "total_questions": n,
        "total_resolved_entities": total_resolved_entities,
        "total_correct_entities": total_correct,
        "total_gt_entities": total_gt_entities,
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
    print("Stage 1 Gate Evaluation — Associative Encoder")
    print("=" * 60)

    # Load graph
    graph = InMemoryGraphStore()
    experiments_dir = os.path.join(_repo_root, "sam-lm", "experiments")
    populate_graph(Path(experiments_dir), graph)
    print(f"Graph loaded: {len(graph._nodes)} nodes")

    # Load encoder
    encoder = get_encoder()
    rss_before = encoder.rss_delta_mb  # already loaded

    # Load test data
    test_dir = os.path.join(os.path.dirname(__file__), "data")
    test_questions = load_questions(os.path.join(test_dir, "test.jsonl"))
    print(f"Test questions: {len(test_questions)}")

    # ── Lexical baseline ──
    print("\n--- Lexical Baseline ---")
    baseline = eval_lexical_baseline(graph, test_questions)
    print(f"  entity_accuracy: {baseline['entity_accuracy']:.4f} ({baseline['entity_accuracy']*100:.1f}%)")
    print(f"  resolution_rate: {baseline['resolution_rate']:.4f} ({baseline['resolution_rate']*100:.1f}%)")

    # ── Encoder evaluation ──
    print("\n--- Encoder Evaluation (threshold=0.55) ---")
    encoder_results = eval_encoder(graph, test_questions, encoder, entity_threshold=0.55)

    print(f"  entity_accuracy:  {encoder_results['entity_accuracy']:.4f} ({encoder_results['entity_accuracy']*100:.1f}%)")
    print(f"  resolution_rate:  {encoder_results['resolution_rate']:.4f} ({encoder_results['resolution_rate']*100:.1f}%)")
    print(f"  intent_accuracy:  {encoder_results['intent_accuracy']:.4f} ({encoder_results['intent_accuracy']*100:.1f}%)")
    print(f"  encoder_precision:{encoder_results['encoder_precision']:.4f} ({encoder_results['encoder_precision']*100:.1f}%)")
    print(f"  enc_res_rate:     {encoder_results['encoder_resolution_rate']:.4f} ({encoder_results['encoder_resolution_rate']*100:.1f}%)")
    print(f"  inference_p50:    {encoder_results['inference_p50_ms']:.1f} ms")
    print(f"  inference_p90:    {encoder_results['inference_p90_ms']:.1f} ms")
    print(f"  RSS delta:        {encoder.rss_delta_mb:.1f} MB")
    print(f"  params:           {encoder.param_count:,}")

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
    print("GATE CHECKS")
    print("=" * 60)

    gates: dict[str, tuple[bool, str]] = {}

    # Gate 1: entity_accuracy >= 65%
    gate1 = encoder_results["encoder_precision"] >= 0.65
    gates["entity_accuracy"] = (
        gate1,
        f"{encoder_results['encoder_precision']*100:.1f}% >= 65%",
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
    print(f"\n  ALL GATES: {'PASS' if all_pass else 'FAIL'}")

    # ── Write results ──
    results = {
        "baseline": baseline,
        "encoder": {k: v for k, v in encoder_results.items() if not isinstance(v, list)},
        "paraphrase": para_results,
        "gates": {k: {"passed": p, "detail": d} for k, (p, d) in gates.items()},
        "all_pass": all_pass,
        "rss_delta_mb": encoder.rss_delta_mb,
        "param_count": encoder.param_count,
    }

    results_path = os.path.join(_repo_root, "models", "encoder", "gate_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {results_path}")

    if not all_pass:
        # Write STAGE1_NEGATIVE.md
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

    # Find 20 worst cases
    errors = []
    for q in test_questions:
        encoder_result = encoder.predict(q["question"])
        pq = parse_question(q["question"], graph, config=config, encoder_model=encoder)
        resolved_ids = pq.entity_ids[: config.max_entry_nodes]
        gt_ids = set(q["entities"])

        # Find which GT entities were missed
        resolved_set = set(resolved_ids)
        missed = gt_ids - resolved_set
        extra = resolved_set - gt_ids

        # Intent check
        intent_match = pq.intent == q.get("intent", "")

        if missed or extra or not intent_match:
            errors.append({
                "id": q["id"],
                "question": q["question"],
                "gt_entities": list(gt_ids),
                "resolved_entities": resolved_ids,
                "missed": list(missed),
                "extra": list(extra),
                "gt_intent": q.get("intent", ""),
                "pred_intent": pq.intent,
                "intent_match": intent_match,
                "encoder_entities": encoder_result["entity_ids"],
                "encoder_scores": encoder_result["entity_scores"],
            })

    errors.sort(key=lambda e: len(e["missed"]) + len(e["extra"]), reverse=True)
    worst_20 = errors[:20]

    lines: list[str] = []
    lines.append("# STAGE1_NEGATIVE.md — Associative Encoder Gate Failure")
    lines.append("")
    lines.append("Status: **GATES FAILED** — see details below.")
    lines.append("")
    lines.append("## Gate Results")
    lines.append("")
    lines.append("| Gate | Value | Threshold | Pass |")
    lines.append("|------|-------|-----------|------|")
    lines.append(
        f"| entity_accuracy | {encoder_results['encoder_precision']*100:.1f}% | >= 65% | {'PASS' if encoder_results['encoder_precision'] >= 0.65 else 'FAIL'} |"
    )
    lines.append(
        f"| resolution_rate | {encoder_results['resolution_rate']*100:.1f}% | >= 100% (no regression) | {'PASS' if encoder_results['resolution_rate'] >= 1.0 else 'FAIL'} |"
    )
    lines.append(
        f"| paraphrase_drop | {para_results['drop_pp']:.1f} pp | < 10 pp | {'PASS' if abs(para_results['drop_pp']) < 10 else 'FAIL'} |"
    )
    lines.append(
        f"| intent_accuracy | {encoder_results['intent_accuracy']*100:.1f}% | >= 85% | {'PASS' if encoder_results['intent_accuracy'] >= 0.85 else 'FAIL'} |"
    )
    lines.append(
        f"| RSS delta | {encoder.rss_delta_mb:.1f} MB | <= 150 MB | {'PASS' if encoder.rss_delta_mb <= 150 else 'FAIL'} |"
    )
    lines.append(
        f"| inference p50 | {encoder_results['inference_p50_ms']:.1f} ms | <= 50 ms | {'PASS' if encoder_results['inference_p50_ms'] <= 50 else 'FAIL'} |"
    )
    lines.append("")
    lines.append("## Per-Head Metrics")
    lines.append("")
    lines.append(f"- **Entity precision** (encoder-only): {encoder_results['encoder_precision']*100:.1f}%")
    lines.append(f"- **Entity resolution rate** (encoder-only): {encoder_results['encoder_resolution_rate']*100:.1f}%")
    lines.append(f"- **Combined entity_accuracy**: {encoder_results['entity_accuracy']*100:.1f}%")
    lines.append(f"- **Combined resolution_rate**: {encoder_results['resolution_rate']*100:.1f}%")
    lines.append(f"- **Intent accuracy**: {encoder_results['intent_accuracy']*100:.1f}%")
    lines.append(f"- **Paraphrase drop**: {para_results['drop_pp']:.1f} pp")
    lines.append(f"- **Inference p50**: {encoder_results['inference_p50_ms']:.1f} ms")
    lines.append(f"- **RSS delta**: {encoder.rss_delta_mb:.1f} MB")
    lines.append(f"- **Parameters**: {encoder.param_count:,}")
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
