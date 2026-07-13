"""Stage 2 & 3 Revalidation — Realization and Dialogue on canonical NEXUS pipeline.

Runs Stage 2 (realization L1) and Stage 3 (dialogue) using the
canonical NEXUSRunner instead of the old ad-hoc paths.
Records complete per-question and per-dialogue outputs.

Does not reuse historical PASS labels.
Does not read the consumed frozen split.

Usage:
    python benchmarks/run_stage2_stage3.py --stage 2 --limit 30
    python benchmarks/run_stage2_stage3.py --stage 3
    python benchmarks/run_stage2_stage3.py --stage both --limit 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner
from benchmarks.realizer_contracts import sha256_json


# ═══════════════════════════════════════════════════════════════════════
# Gate thresholds (immutable)
# ═══════════════════════════════════════════════════════════════════════

STAGE2_GATES = {
    "naturalness_improvement": {"threshold": 5.0, "operator": ">="},
    "relevance": {"threshold": 0.77, "operator": ">="},
    "accuracy_no_worse_than_2pp_below_baseline": {"threshold": -0.02, "operator": ">="},
}

STAGE3_GATES = {
    "reference_resolution": {"threshold": 0.70, "operator": ">="},
    "single_turn_regression": {"threshold": 0.02, "operator": "<="},
    "dialogue_latency_p50_ms": {"threshold": 5.0, "operator": "<="},
}


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — Realization L1
# ═══════════════════════════════════════════════════════════════════════

def _canonicalize_stage2(value: Any, *, preserve_list: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonicalize_stage2(item, preserve_list=key in {"per_question", "case_order"})
            for key, item in value.items()
        }
    if isinstance(value, list):
        items = [_canonicalize_stage2(item) for item in value]
        if preserve_list:
            return items
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return value


def _canonical_stage2_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the reproducibility payload, excluding runtime-only fields."""
    payload = json.loads(json.dumps(artifact, ensure_ascii=False))
    for key in ("created_utc", "canonical_content_sha256", "serialized_file_sha256"):
        payload.pop(key, None)
    for row in payload.get("per_question", []):
        if isinstance(row, dict):
            row.pop("latency_ms", None)
    return _canonicalize_stage2(payload)


def _write_stage2_artifact(artifact: dict[str, Any], output_path: str) -> None:
    artifact["canonical_content_sha256"] = sha256_json(_canonical_stage2_payload(artifact))
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.write_text(serialized, encoding="utf-8")
    artifact["serialized_file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    # Rewrite once with the final file hash; it is intentionally not canonical.
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def run_stage2(
    questions: list[dict],
    graph: InMemoryGraphStore,
    config: ProductionNEXUSConfig,
    source_sha: str,
    output_path: str,
    baseline: dict[str, Any],
    *,
    dataset_sha256: str = "",
    source_tree_sha: str = "",
    protocol: str = "registered_stage2_v1",
) -> dict:
    """Revalidate Stage 2 realization gates with reproducible protocol metadata."""
    from benchmarks.naturalness_eval import score_naturalness
    from benchmarks.relevance_judge import RelevanceJudge
    runner = NEXUSRunner(graph, config)
    relevance_judge = RelevanceJudge()

    results = []
    naturalness_scores = []
    accuracy_scores = []
    hallucination_rates = []
    relevance_results = []

    utc_now = datetime.now(timezone.utc)

    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        pipeline_result = runner.run([q], source_sha=source_sha)
        qr = pipeline_result.per_question[0]
        latency = round((time.perf_counter() - t0) * 1000, 3)

        answer = qr.answer or ""
        ground_truth = str(q.get("answer", q.get("entities", "")))
        question_type = str(q.get("question_type", "factual_lookup"))

        # Score naturalness
        evidence = qr.evidence_pack
        facts = [
            str(item.get("text", ""))
            for item in evidence.get("node_facts", [])
            if isinstance(item, dict) and item.get("text")
        ]
        edge_types = sorted({
            str(edge.get("type", ""))
            for path in evidence.get("paths", [])
            for edge in path.get("edges", [])
            if isinstance(edge, dict) and edge.get("type")
        })
        nat_detail = score_naturalness(answer, facts, edge_types)
        nat_score = float(nat_detail["total"])
        naturalness_scores.append(nat_score)

        # Score relevance (3 args: question, answer, question_type)
        rel = relevance_judge.judge(q["question"], answer, question_type)
        relevance_results.append(rel["verdict"])

        # Score accuracy
        from benchmarks.run_benchmark import compute_key_fact_score
        acc_raw = compute_key_fact_score(answer, ground_truth)
        if isinstance(acc_raw, dict):
            acc = acc_raw.get("fuzzy_accuracy")
            if acc is None:
                acc = acc_raw.get("exact_accuracy")
        else:
            acc = acc_raw
        acc = float(acc or 0.0)
        accuracy_scores.append(acc)

        # Hallucination
        hallucination_rates.append(qr.hallucination_rate)

        results.append({
            "question_id": q.get("id", str(i)),
            "question": q["question"][:200],
            "answer": answer[:500],
            "naturalness": nat_score,
            "naturalness_detail": nat_detail,
            "relevance": rel,
            "accuracy": acc,
            "hallucination_rate": qr.hallucination_rate,
            "entity_resolution_method": qr.entity_resolution_method,
            "evidence": evidence,
            "latency_ms": latency,
            "failure_category": qr.failure_category,
        })

    # Compute metrics
    nat_mean = sum(naturalness_scores) / max(1, len(naturalness_scores))
    acc_mean = sum(accuracy_scores) / max(1, len(accuracy_scores))
    rel_pass = sum(1.0 if r == "yes" else 0.5 if r == "partial" else 0.0 for r in relevance_results) / max(1, len(relevance_results))
    hal_mean = sum(hallucination_rates) / max(1, len(hallucination_rates))
    for key in ("naturalness_mean", "accuracy_mean", "hallucination_mean"):
        if not isinstance(baseline.get(key), (int, float)):
            raise ValueError(f"registered baseline missing numeric {key}")
    naturalness_delta = nat_mean - float(baseline["naturalness_mean"])
    accuracy_delta = acc_mean - float(baseline["accuracy_mean"])
    hallucination_delta = hal_mean - float(baseline["hallucination_mean"])
    gates_passed = (
        rel_pass >= 0.77
        and naturalness_delta >= 5.0
        and hallucination_delta <= 0.0
        and accuracy_delta >= -0.02
    )

    artifact = {
        "schema_version": "nexus-stage2-v1",
        "stage": "stage2_realization_l1",
        "created_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "source_tree_sha": source_tree_sha,
        "dataset_sha256": dataset_sha256,
        "protocol": protocol,
        "effective_config": {
            "config_hash": config.config_hash,
            "entity_resolution": "entity_ranker_v3" if config.pipeline_id.entity_ranker_v3_enabled else "lexical",
            "question_limit": len(questions),
            "question_order": "questions.jsonl source order",
        },
        "config_hash": config.config_hash,
        "entity_resolution": "entity_ranker_v3" if config.pipeline_id.entity_ranker_v3_enabled else "lexical",
        "questions_total": len(questions),
        "case_order": [q.get("id", str(i)) for i, q in enumerate(questions)],
        "question_set_sha256": sha256_json(questions),
        "metrics": {
            "naturalness_mean": round(nat_mean, 3),
            "relevance_rate": round(rel_pass, 4),
            "accuracy_mean": round(acc_mean, 4),
            "hallucination_mean": round(hal_mean, 4),
            "naturalness_improvement": round(naturalness_delta, 4),
            "accuracy_delta_vs_baseline": round(accuracy_delta, 4),
            "hallucination_delta_vs_baseline": round(hallucination_delta, 4),
        },
        "registered_baseline": baseline,
        "registered_baseline_sha256": sha256_json(baseline),
        "gates": {
            "naturalness_improvement": {"value": naturalness_delta, "threshold": 5.0, "passed": naturalness_delta >= 5.0},
            "relevance": {"value": rel_pass, "threshold": 0.77, "passed": rel_pass >= 0.77},
            "hallucination": {"value": hallucination_delta, "threshold": 0.0, "passed": hallucination_delta <= 0.0},
            "accuracy": {"value": accuracy_delta, "threshold": -0.02, "passed": accuracy_delta >= -0.02},
        },
        "per_question": results,
        "status": "PASS" if gates_passed else "FAIL",
    }

    _write_stage2_artifact(artifact, output_path)

    return artifact


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 — Dialogue
# ═══════════════════════════════════════════════════════════════════════

def run_stage3(
    dialogues_path: str,
    graph: InMemoryGraphStore,
    config: ProductionNEXUSConfig,
    source_sha: str,
    output_path: str,
) -> dict:
    """Revalidate Stage 3 dialogue gates."""
    from stack.dialogue.state import DialogueState
    from nexus.query.parser import parse_question

    turns = []
    with open(dialogues_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))

    state = DialogueState()
    utc_now = datetime.now(timezone.utc)

    total_turns = 0
    resolved_correct = 0
    single_turn_correct = 0
    single_turn_total = 0
    latencies = []
    per_dialogue = []

    current_dialogue_id = None
    dialogue_turns = []

    for turn in turns:
        did = turn.get("dialogue_id", "")
        if did != current_dialogue_id:
            if dialogue_turns:
                per_dialogue.append({"dialogue_id": current_dialogue_id, "turns": dialogue_turns})
            dialogue_turns = []
            current_dialogue_id = did
            state = DialogueState()

        question = turn["question"]
        t0 = time.perf_counter()
        parsed = parse_question(question, graph, config=config, dialogue_state=state)
        lat = round((time.perf_counter() - t0) * 1000, 3)
        latencies.append(lat)
        total_turns += 1

        # Update dialogue state with resolved entities
        state.update(parsed.entity_ids)

        # Use gt_entities field (correct dataset field name)
        gold_entities = set(turn.get("gt_entities", turn.get("entities", [])))
        is_context = turn.get("resolution_source") == "context"

        if is_context:
            if gold_entities and parsed.entity_ids:
                if gold_entities & set(parsed.entity_ids):
                    resolved_correct += 1

        if not is_context:
            single_turn_total += 1
            if gold_entities & set(parsed.entity_ids):
                single_turn_correct += 1

        dialogue_turns.append({
            "turn_id": turn.get("id", ""),
            "question": question[:200],
            "parsed_entities": parsed.entity_ids[:10],
            "gold_entities": sorted(gold_entities),
            "is_context_reference": is_context,
            "latency_ms": lat,
        })

    if dialogue_turns:
        per_dialogue.append({"dialogue_id": current_dialogue_id, "turns": dialogue_turns})

    # Count context turns correctly
    context_turns = sum(
        1 for t in turns if t.get("resolution_source") == "context"
    )
    ref_resolution = resolved_correct / max(1, context_turns)
    single_turn_acc = single_turn_correct / max(1, single_turn_total)
    latencies.sort()
    p50_lat = latencies[len(latencies) // 2] if latencies else 0

    artifact = {
        "stage": "stage3_dialogue",
        "created_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "total_turns": total_turns,
        "metrics": {
            "reference_resolution": round(ref_resolution, 4),
            "single_turn_accuracy": round(single_turn_acc, 4),
            "dialogue_latency_p50_ms": round(p50_lat, 3),
        },
        "gates": {
            "reference_resolution": ref_resolution,
            "ref_res_pass": ref_resolution >= 0.70,
            "dialogue_latency_p50_ms": p50_lat,
            "latency_pass": p50_lat <= 5.0,
        },
        "per_dialogue": per_dialogue,
        "status": "PASS" if ref_resolution >= 0.70 and p50_lat <= 5.0 else "FAIL",
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    return artifact


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 2 & 3 revalidation")
    parser.add_argument("--stage", choices=["2", "3", "both"], default="both")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--er3", action="store_true", help="Use Entity Ranker V3")
    parser.add_argument("--baseline", default="training/stage2_baseline_v1.json")
    parser.add_argument("--dataset-manifest", default="", help="Optional hash-identified distillation manifest")
    args = parser.parse_args()

    from benchmarks.run_benchmark import build_benchmark_graph
    graph, _ = build_benchmark_graph()

    config = (
        ProductionNEXUSConfig.with_entity_ranker_v3()
        if args.er3 else ProductionNEXUSConfig.lexical_only()
    )
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_tree_sha = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    dataset_sha256 = ""
    if args.dataset_manifest:
        dataset_sha256 = hashlib.sha256(Path(args.dataset_manifest).read_bytes()).hexdigest()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.stage in ("2", "both"):
        qa_path = _project_root / "benchmarks" / "qa-dataset" / "questions.jsonl"
        questions = [json.loads(line) for line in Path(qa_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        questions = questions[:args.limit]
        out = f"{args.output_dir}/stage2_{ts}.json"
        print(f"Stage 2: {len(questions)} questions")
        baseline_path = Path(args.baseline)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        result = run_stage2(
            questions, graph, config, source_sha, out, baseline,
            dataset_sha256=dataset_sha256, source_tree_sha=source_tree_sha,
            protocol=f"registered_stage2_v1_limit_{args.limit}",
        )
        print(f"  Relevance: {result['metrics']['relevance_rate']:.4f} (gate: 0.77)")
        print(f"  Status: {result['status']}")

    if args.stage in ("3", "both"):
        dialogues_path = _project_root / "benchmarks" / "qa-dataset" / "dialogues.jsonl"
        if not Path(dialogues_path).exists():
            print("Stage 3: dialogues.jsonl not found — skipping")
        else:
            out = f"{args.output_dir}/stage3_{ts}.json"
            result = run_stage3(str(dialogues_path), graph, config, source_sha, out)
            print(f"Stage 3: {result['total_turns']} turns")
            print(f"  Reference resolution: {result['metrics']['reference_resolution']:.4f} (gate: 0.70)")
            print(f"  Status: {result['status']}")


if __name__ == "__main__":
    main()
