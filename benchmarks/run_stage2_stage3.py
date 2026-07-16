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
import os
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
from nexus.pipeline.entity_resolver import EntityResolver, coerce_resolution_result
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


def stage2_protocol_for_limit(limit: int) -> str:
    if limit < 1:
        raise ValueError("Stage 2 limit must be >= 1")
    return "registered_stage2_v1" if limit == 30 else f"smoke_stage2_{limit}"


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
    for key in (
        "created_utc",
        "canonical_content_sha256",
        "serialized_file_sha256",
        "serialized_sha256_sidecar",
        "python_hash_seed",
    ):
        payload.pop(key, None)
    for row in payload.get("per_question", []):
        if isinstance(row, dict):
            row.pop("latency_ms", None)
    return _canonicalize_stage2(payload)


def _write_stage2_artifact(artifact: dict[str, Any], output_path: str) -> None:
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(out.suffix + ".sha256")
    if sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite: {sidecar}")
    artifact["serialized_sha256_sidecar"] = sidecar.name
    artifact["canonical_content_sha256"] = sha256_json(
        _canonical_stage2_payload(artifact)
    )
    serialized = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {out.name}\n", encoding="ascii")
    # Returned for callers/tests, but intentionally not written into the file
    # whose bytes it authenticates.
    artifact["serialized_file_sha256"] = digest


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
    entity_resolver: EntityResolver | None = None,
) -> dict:
    """Revalidate Stage 2 realization gates with reproducible protocol metadata.

    'registered_stage2_v1' requires exactly 30 questions with deterministic
    ordering. Non-matching runs are designated as smoke/ad-hoc and cannot
    claim PASS against the registered gate.
    """
    is_registered = protocol == "registered_stage2_v1"
    if is_registered and len(questions) != 30:
        raise ValueError(
            f"registered_stage2_v1 requires exactly 30 questions, got {len(questions)}. "
            "Use a different protocol name for ad-hoc/smoke runs."
        )
    from benchmarks.naturalness_eval import score_naturalness
    from benchmarks.relevance_judge import RelevanceJudge
    runner = NEXUSRunner(graph, config, entity_resolver=entity_resolver)
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

    protocol_kind = "registered" if is_registered else "smoke_or_adhoc"
    registered_gate_status = (
        ("PASS" if gates_passed else "FAIL")
        if is_registered else "NOT_APPLICABLE"
    )
    artifact = {
        "schema_version": "nexus-stage2-v1",
        "stage": "stage2_realization_l1",
        "created_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "source_tree_sha": source_tree_sha,
        "dataset_sha256": dataset_sha256,
        "protocol": protocol,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "protocol_kind": protocol_kind,
        "execution_status": "COMPLETE",
        "registered_gate_status": registered_gate_status,
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
        "status": (
            registered_gate_status
            if is_registered
            else ("SMOKE_PASS" if gates_passed else "SMOKE_FAIL")
        ),
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
    entity_resolver: EntityResolver | None = None,
) -> dict:
    """Revalidate Stage 3 dialogue gates."""
    from stack.dialogue.state import DialogueState
    from stack.pipeline.resolver import DialogueAwareResolver, LexicalResolver
    from nexus.reasoning.model_interface import get_available_model

    turns = []
    with open(dialogues_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))

    state = DialogueState()
    shared_model = get_available_model()
    utc_now = datetime.now(timezone.utc)

    total_turns = 0
    resolved_correct = 0
    single_turn_correct = 0
    single_turn_total = 0
    baseline_single_turn_correct = 0
    latencies = []
    resolver_latencies = []
    pipeline_latencies = []
    per_dialogue = []

    current_dialogue_id = None
    dialogue_turns = []
    active_resolver: EntityResolver | None = None
    active_runner: NEXUSRunner | None = None

    for turn in turns:
        did = turn.get("dialogue_id", "")
        if did != current_dialogue_id:
            if dialogue_turns:
                per_dialogue.append({"dialogue_id": current_dialogue_id, "turns": dialogue_turns})
            dialogue_turns = []
            current_dialogue_id = did
            state = DialogueState()
            active_resolver = (
                DialogueAwareResolver(entity_resolver, state)
                if entity_resolver is not None
                else LexicalResolver(dialogue_state=state, config=config)
            )
            active_runner = NEXUSRunner(
                graph,
                config,
                model=shared_model,
                entity_resolver=active_resolver,
                dialogue_state=state,
            )

        question = turn["question"]
        t0 = time.perf_counter()
        if active_runner is None:
            raise RuntimeError("Stage 3 runner was not initialized")
        pipeline_result = active_runner.run(
            [{"id": turn.get("id", ""), "question": question}],
            source_sha=source_sha,
        )
        if pipeline_result.errors or not pipeline_result.per_question:
            raise RuntimeError(
                "Stage 3 pipeline failed: " + "; ".join(pipeline_result.errors)
            )
        qr = pipeline_result.per_question[0]
        pipeline_lat = round((time.perf_counter() - t0) * 1000, 3)
        # The preregistered 5 ms gate concerns dialogue-state overhead, not
        # the cost of the base neural/lexical resolver. Record all three
        # timings separately so the gate cannot be accidentally redefined.
        lat = qr.resolver_context_latency_ms
        latencies.append(lat)
        resolver_latencies.append(qr.resolver_latency_ms or pipeline_lat)
        pipeline_latencies.append(pipeline_lat)
        total_turns += 1

        # Avoid contaminating future turns with the full top-K list.  Only the
        # highest-ranked selected entity becomes active dialogue context.
        state_update_entities = qr.selected_entry_nodes[:1]
        state.update(state_update_entities)

        # Use gt_entities field (correct dataset field name)
        gold_entities = set(turn.get("gt_entities", turn.get("entities", [])))
        is_context = turn.get("resolution_source") == "context"

        if is_context:
            if gold_entities and qr.selected_entry_nodes:
                if gold_entities & set(qr.selected_entry_nodes):
                    resolved_correct += 1

        if not is_context:
            single_turn_total += 1
            if gold_entities & set(qr.selected_entry_nodes):
                single_turn_correct += 1
            baseline_resolver = (
                entity_resolver
                if entity_resolver is not None
                else LexicalResolver(config=config)
            )
            baseline_result = coerce_resolution_result(
                baseline_resolver.resolve(question, graph),
                resolver_name=baseline_resolver.__class__.__name__,
            )
            if gold_entities & set(baseline_result.selected_entity_ids):
                baseline_single_turn_correct += 1

        dialogue_turns.append({
            "turn_id": turn.get("id", ""),
            "question": question[:200],
            "parsed_entities": qr.selected_entry_nodes[:10],
            "resolver_name": qr.resolver_name,
            "resolver_candidates": qr.resolution_candidates,
            "candidate_pool_size": qr.candidate_pool_size,
            "state_update_entities": state_update_entities,
            "gold_entities": sorted(gold_entities),
            "is_context_reference": is_context,
            "dialogue_state_latency_ms": lat,
            "resolver_latency_ms": qr.resolver_latency_ms,
            "pipeline_latency_ms": pipeline_lat,
        })

    if dialogue_turns:
        per_dialogue.append({"dialogue_id": current_dialogue_id, "turns": dialogue_turns})

    # Count context turns correctly
    context_turns = sum(
        1 for t in turns if t.get("resolution_source") == "context"
    )
    ref_resolution = resolved_correct / max(1, context_turns)
    single_turn_acc = single_turn_correct / max(1, single_turn_total)
    baseline_single_turn_acc = (
        baseline_single_turn_correct / max(1, single_turn_total)
    )
    single_turn_regression = max(
        0.0, baseline_single_turn_acc - single_turn_acc
    )
    latencies.sort()
    p50_lat = latencies[len(latencies) // 2] if latencies else 0
    resolver_latencies.sort()
    resolver_p50 = (
        resolver_latencies[len(resolver_latencies) // 2]
        if resolver_latencies else 0
    )
    pipeline_latencies.sort()
    pipeline_p50 = (
        pipeline_latencies[len(pipeline_latencies) // 2]
        if pipeline_latencies else 0
    )

    artifact = {
        "stage": "stage3_dialogue",
        "created_utc": utc_now.isoformat(),
        "source_sha": source_sha,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "total_turns": total_turns,
        "metrics": {
            "reference_resolution": round(ref_resolution, 4),
            "single_turn_accuracy": round(single_turn_acc, 4),
            "single_turn_baseline_accuracy": round(baseline_single_turn_acc, 4),
            "single_turn_regression": round(single_turn_regression, 4),
            "dialogue_state_latency_p50_ms": round(p50_lat, 3),
            "resolver_latency_p50_ms": round(resolver_p50, 3),
            "pipeline_latency_p50_ms": round(pipeline_p50, 3),
        },
        "gates": {
            "reference_resolution": ref_resolution,
            "ref_res_pass": ref_resolution >= 0.70,
            "single_turn_regression": single_turn_regression,
            "single_turn_regression_pass": single_turn_regression <= 0.02,
            "dialogue_state_latency_p50_ms": p50_lat,
            "latency_pass": p50_lat <= 5.0,
        },
        "per_dialogue": per_dialogue,
        "status": "PASS" if (
            ref_resolution >= 0.70
            and single_turn_regression <= 0.02
            and p50_lat <= 5.0
        ) else "FAIL",
    }

    out = Path(output_path)
    sidecar = out.with_suffix(out.suffix + ".sha256")
    if out.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {out.name}\n", encoding="ascii")

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
    parser.add_argument(
        "--er3-dir",
        default="models/encoder/entity_ranker_v3_20260711T081545Z",
    )
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--baseline", default="training/stage2_baseline_v1.json")
    parser.add_argument("--dataset-manifest", default="", help="Optional hash-identified distillation manifest")
    args = parser.parse_args()

    from benchmarks.run_benchmark import build_benchmark_graph
    graph, _ = build_benchmark_graph()

    entity_resolver: EntityResolver | None = None
    if args.er3:
        from stack.pipeline.resolver import ER3Resolver

        config = ProductionNEXUSConfig.with_entity_ranker_v3(args.er3_dir)
        entity_resolver = ER3Resolver.from_directory(
            args.er3_dir,
            graph,
            weights_path=args.weights_path,
        )
    else:
        config = ProductionNEXUSConfig.lexical_only()
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
            protocol=stage2_protocol_for_limit(args.limit),
            entity_resolver=entity_resolver,
        )
        print(f"  Relevance: {result['metrics']['relevance_rate']:.4f} (gate: 0.77)")
        print(f"  Status: {result['status']}")

    if args.stage in ("3", "both"):
        dialogues_path = _project_root / "benchmarks" / "qa-dataset" / "dialogues.jsonl"
        if not Path(dialogues_path).exists():
            print("Stage 3: dialogues.jsonl not found — skipping")
        else:
            out = f"{args.output_dir}/stage3_{ts}.json"
            result = run_stage3(
                str(dialogues_path), graph, config, source_sha, out,
                entity_resolver=entity_resolver,
            )
            print(f"Stage 3: {result['total_turns']} turns")
            print(f"  Reference resolution: {result['metrics']['reference_resolution']:.4f} (gate: 0.70)")
            print(f"  Status: {result['status']}")


if __name__ == "__main__":
    main()
