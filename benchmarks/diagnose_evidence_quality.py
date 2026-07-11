"""Evidence quality diagnostic — train and validation only.

Measures per-stage failure categories.
Does not read the consumed frozen split.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root on path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner
from nexus.query.parser import parse_question
from nexus.graph.traversal import traverse_with_intent


def run_evidence_diagnostic(
    questions: list[dict],
    graph: InMemoryGraphStore,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run evidence quality diagnostics on a question set."""

    config = ProductionNEXUSConfig.lexical_only()
    utc_now = datetime.now(timezone.utc)

    failure_categories: Counter[str] = Counter()
    entity_present = 0
    entity_missing = 0
    entry_node_hit = 0

    per_question = []

    for i, record in enumerate(questions):
        question = record["question"]
        gold_entities = set(str(e) for e in record.get("entities", []))
        qid = record.get("id", str(i))

        # Parse
        parsed = parse_question(question, graph, config=config)
        predicted = set(parsed.entity_ids)

        # Entity resolution
        gold_found = gold_entities & predicted
        entity_present += len(gold_found)
        entity_missing += len(gold_entities - predicted)

        # Traversal
        paths = []
        entry_hits = 0
        if parsed.entity_ids:
            paths = traverse_with_intent(
                graph, entry_nodes=parsed.entity_ids,
                query_entities=set(parsed.entity_ids),
                intent=parsed.intent,
                max_depth=config.max_depth,
                beam_width=config.beam_width,
                config=config,
            )
            entry_hits = len(gold_entities & set(parsed.entity_ids[:config.max_entry_nodes]))
            entry_node_hit += entry_hits

        # Categorize
        cat = ""
        if not gold_found:
            if gold_entities:
                cat = "entity_missing"
            else:
                cat = "no_gold_entities"
        elif not paths:
            cat = "no_graph_paths"
        elif entry_hits == 0:
            cat = "entity_ranked_too_low"
        else:
            cat = "paths_available"

        failure_categories[cat] += 1

        per_question.append({
            "question_id": qid,
            "question": question[:200],
            "gold_entities": sorted(gold_entities),
            "predicted_entities": sorted(predicted),
            "gold_found": sorted(gold_found),
            "path_count": len(paths),
            "entry_node_hit": entry_hits > 0,
            "failure_category": cat,
        })

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(questions)}")

    result = {
        "diagnostic": "evidence_quality",
        "created_utc": utc_now.isoformat(),
        "total_questions": len(questions),
        "total_gold_entities": entity_present + entity_missing,
        "entity_present": entity_present,
        "entity_missing": entity_missing,
        "entity_recall": entity_present / max(1, entity_present + entity_missing),
        "entry_node_hits": entry_node_hit,
        "failure_categories": dict(failure_categories),
        "per_question": per_question,
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return result


if __name__ == "__main__":
    from benchmarks.run_benchmark import build_benchmark_graph

    graph, _ = build_benchmark_graph()

    for split_name, split_path in [
        ("train", "stack/encoder/data/train.jsonl"),
        ("validation", "stack/encoder/data/val.jsonl"),
    ]:
        questions = [
            json.loads(line)
            for line in Path(split_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"\n=== {split_name}: {len(questions)} questions ===")
        result = run_evidence_diagnostic(questions, graph)
        print(f"  Entity recall: {result['entity_recall']:.4f}")
        print(f"  Failure categories: {dict(result['failure_categories'])}")
