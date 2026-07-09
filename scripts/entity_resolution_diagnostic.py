"""
Entity resolution diagnostic for held-out questions (31-200).

Validates that type-prior entity ranking improves Experiment node selection
for metric questions on the held-out set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from nexus.graph.store import InMemoryGraphStore
from nexus.query.parser import parse_question, _is_metric_question, _is_concept_question
from nexus.utils.config import NEXUSConfig
from benchmarks.run_benchmark import build_benchmark_graph, load_questions


def main():
    dataset_path = _PROJECT_ROOT / "benchmarks" / "qa-dataset" / "questions.jsonl"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    # Load all questions, then slice to hold-out (31-200)
    all_qs = load_questions(str(dataset_path))
    held_out = all_qs[30:200]  # q031 through q200 (0-indexed)
    print(f"Loaded {len(held_out)} held-out questions (q031-q200)")

    # Build graph
    graph, graph_provenance = build_benchmark_graph()
    print(f"Graph: {graph_provenance['node_count']} nodes, {graph_provenance['edge_count']} edges")

    config = NEXUSConfig()

    # Track entity resolution stats
    total = 0
    metric_q_count = 0
    concept_q_count = 0
    exp_picked_for_metric = 0
    metric_node_picked_for_metric = 0
    concept_picked_for_concept = 0
    decision_picked_for_concept = 0
    no_entity_found = 0

    # Per-question type tally for metric questions
    metric_q_exp_top1 = 0
    metric_q_any_exp = 0

    # ── unified metric tracking ──
    resolution_hits = 0     # any entities found
    entity_accuracy_hits = 0  # entities match GT expected
    entity_accuracy_total = 0  # questions with GT entities listed

    for q in held_out:
        q_id = q.get("question_id", "?")
        question = q.get("question", "")
        total += 1

        parsed = parse_question(question, graph, config=config)
        entity_ids = parsed.entity_ids
        is_metric = _is_metric_question(question)
        is_concept = _is_concept_question(question)

        # ── Resolution rate: any entities found? ──
        if entity_ids:
            resolution_hits += 1

        # ── Entity accuracy: match against GT expected entities ──
        gt_entity_ids: list[str] = q.get("entities", [])
        if gt_entity_ids:
            entity_accuracy_total += 1
            if any(
                gid == pid or pid.startswith(gid + "_")
                for gid in gt_entity_ids
                for pid in entity_ids
            ):
                entity_accuracy_hits += 1

    for q in held_out:
        q_id = q.get("question_id", "?")
        question = q.get("question", "")
        total += 1

        parsed = parse_question(question, graph, config=config)
        entity_ids = parsed.entity_ids
        is_metric = _is_metric_question(question)
        is_concept = _is_concept_question(question)

        if not entity_ids:
            no_entity_found += 1
            if is_metric:
                metric_q_count += 1
            if is_concept:
                concept_q_count += 1
            continue

        # Check types of resolved entities
        entity_types = []
        for eid in entity_ids:
            node = graph.get_node(eid)
            if node:
                entity_types.append(node.type)

        if is_metric:
            metric_q_count += 1
            if "Experiment" in entity_types:
                exp_picked_for_metric += 1
            if "Metric" in entity_types:
                metric_node_picked_for_metric += 1
            if entity_types and entity_types[0] == "Experiment":
                metric_q_exp_top1 += 1
            if "Experiment" in entity_types:
                metric_q_any_exp += 1

        if is_concept:
            concept_q_count += 1
            if "Concept" in entity_types:
                concept_picked_for_concept += 1
            if "Decision" in entity_types:
                decision_picked_for_concept += 1

    print(f"\n{'='*60}")
    print("  Entity Resolution Diagnostic — Held-out (q031-q200)")
    print(f"{'='*60}")
    print(f"  Total questions evaluated : {total}")
    print(f"  No entity found            : {no_entity_found}")
    print()
    print(f"  -- Unified entity resolution metrics --")
    resolution_rate = resolution_hits / total if total > 0 else 0.0
    entity_accuracy_rate = entity_accuracy_hits / entity_accuracy_total if entity_accuracy_total > 0 else 0.0
    print(f"  resolution_rate  (any entities found)  : {resolution_hits}/{total} = {resolution_rate:.1%}")
    print(f"  entity_accuracy  (matches GT expected)  : {entity_accuracy_hits}/{entity_accuracy_total} = {entity_accuracy_rate:.1%}")
    print()
    print(f"  Metric questions           : {metric_q_count}")
    print(f"    Experiment in results    : {exp_picked_for_metric} ({exp_picked_for_metric/max(metric_q_count,1)*100:.1f}%)")
    print(f"    Experiment as top-1      : {metric_q_exp_top1} ({metric_q_exp_top1/max(metric_q_count,1)*100:.1f}%)")
    print(f"    Any Experiment in top-N  : {metric_q_any_exp} ({metric_q_any_exp/max(metric_q_count,1)*100:.1f}%)")
    print(f"    Metric node in results   : {metric_node_picked_for_metric} ({metric_node_picked_for_metric/max(metric_q_count,1)*100:.1f}%)")
    print()
    print(f"  Concept questions          : {concept_q_count}")
    print(f"    Concept in results       : {concept_picked_for_concept} ({concept_picked_for_concept/max(concept_q_count,1)*100:.1f}%)")
    print(f"    Decision in results      : {decision_picked_for_concept} ({decision_picked_for_concept/max(concept_q_count,1)*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
