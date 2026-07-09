"""
Before/after comparison of type-prior entity ranking on held-out questions.
Compares type_prior_boost=0.0 vs type_prior_boost=0.15.
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


def evaluate(config: NEXUSConfig, questions: list[dict], graph):
    metric_q_count = 0
    exp_picked_for_metric = 0
    exp_top1_for_metric = 0
    concept_q_count = 0
    concept_picked_for_concept = 0
    decision_picked_for_concept = 0
    no_entity = 0

    for q in questions:
        question = q.get("question", "")
        parsed = parse_question(question, graph, config=config)
        entity_ids = parsed.entity_ids
        is_metric = _is_metric_question(question)
        is_concept = _is_concept_question(question)

        if not entity_ids:
            no_entity += 1
            if is_metric:
                metric_q_count += 1
            if is_concept:
                concept_q_count += 1
            continue

        entity_types = []
        for eid in entity_ids:
            node = graph.get_node(eid)
            if node:
                entity_types.append(node.type)

        if is_metric:
            metric_q_count += 1
            if "Experiment" in entity_types:
                exp_picked_for_metric += 1
            if entity_types and entity_types[0] == "Experiment":
                exp_top1_for_metric += 1

        if is_concept:
            concept_q_count += 1
            if "Concept" in entity_types:
                concept_picked_for_concept += 1
            if "Decision" in entity_types:
                decision_picked_for_concept += 1

    return {
        "metric_q_count": metric_q_count,
        "exp_in_results": exp_picked_for_metric,
        "exp_top1": exp_top1_for_metric,
        "concept_q_count": concept_q_count,
        "concept_in_results": concept_picked_for_concept,
        "decision_in_results": decision_picked_for_concept,
        "no_entity": no_entity,
    }


def main():
    dataset_path = _PROJECT_ROOT / "benchmarks" / "qa-dataset" / "questions.jsonl"
    all_qs = load_questions(str(dataset_path))
    held_out = all_qs[30:200]

    graph, _ = build_benchmark_graph()

    config_before = NEXUSConfig(type_prior_boost=0.0)
    config_after = NEXUSConfig(type_prior_boost=0.15)

    res_before = evaluate(config_before, held_out, graph)
    res_after = evaluate(config_after, held_out, graph)

    print(f"  Total held-out questions: {len(held_out)}")
    print()
    print(f"  {'':>35} {'Before (0.0)':>18} {'After (0.15)':>18}")
    print(f"  {'-'*35} {'-'*18} {'-'*18}")
    print(f"  {'Metric questions':>35} {res_before['metric_q_count']:>18} {res_after['metric_q_count']:>18}")
    print(f"  {'  Exp in results':>35} {res_before['exp_in_results']:>18} {res_after['exp_in_results']:>18}")
    print(f"  {'  Exp as top-1':>35} {res_before['exp_top1']:>18} {res_after['exp_top1']:>18}")
    print(f"  {'Concept questions':>35} {res_before['concept_q_count']:>18} {res_after['concept_q_count']:>18}")
    print(f"  {'  Concept in results':>35} {res_before['concept_in_results']:>18} {res_after['concept_in_results']:>18}")
    print(f"  {'  Decision in results':>35} {res_before['decision_in_results']:>18} {res_after['decision_in_results']:>18}")
    print(f"  {'No entity found':>35} {res_before['no_entity']:>18} {res_after['no_entity']:>18}")

    mc = max(res_before['metric_q_count'], 1)
    delta_exp_top1 = res_after['exp_top1'] - res_before['exp_top1']
    print(f"\n  Delta: Exp as top-1 for metric Qs: {delta_exp_top1:+d} ({delta_exp_top1/mc*100:+.1f}pp)")


if __name__ == "__main__":
    main()
