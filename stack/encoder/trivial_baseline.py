"""Validation-only, non-parametric lexical candidate ranking baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from nexus.query.parser import spot_entities
from stack.encoder.eval_gates import _stage1_exact_name_alias, _stage4_graph_expansion
from stack.encoder.stage1c import _tokens, stage1c_property_candidates


def _node_degree(graph: Any, node_id: str) -> int:
    """Return total graph degree, using both incoming and outgoing edges."""
    return len(graph.get_outgoing(node_id)) + len(graph.get_incoming(node_id))


def rank_candidates(
    candidates: Iterable[Mapping[str, Any]], graph: Any, top_k: int
) -> list[str]:
    """Rank an existing candidate pool by lexical score, degree, then ID.

    The final ID tie-break makes output independent of input pool iteration order.
    Candidate records must contain ``node_id`` and ``lexical_score``.
    """
    if top_k <= 0:
        return []
    unique: dict[str, float] = {}
    for candidate in candidates:
        node_id = str(candidate["node_id"])
        score = float(candidate["lexical_score"])
        unique[node_id] = max(score, unique.get(node_id, score))
    ranked = sorted(
        unique.items(),
        key=lambda item: (-item[1], -_node_degree(graph, item[0]), item[0]),
    )
    return [node_id for node_id, _score in ranked[:top_k]]


def _lexical_score(question: str, node: Any) -> int:
    """Use the fixed overlap formula used by the existing lexical candidate path."""
    properties = node.properties if isinstance(node.properties, dict) else {}
    query = _tokens(question)
    aliases = _tokens(" ".join(str(alias) for alias in getattr(node, "aliases", [])))
    name = _tokens(str(node.id))
    finding = _tokens(str(properties.get("key_finding", "")))
    description = _tokens(str(properties.get("description", "")))
    exact_phrase = any(
        len(_tokens(str(alias))) >= 2 and str(alias).casefold() in question.casefold()
        for alias in getattr(node, "aliases", [])
    )
    return (
        100 * int(exact_phrase)
        + 8 * len(query & aliases)
        + 5 * len(query & name)
        + 4 * len(query & finding)
        + 2 * len(query & description)
    )


def candidate_pool(question: str, graph: Any) -> list[dict[str, Any]]:
    """Build the same graph-backed candidate pool used before re-ranking."""
    stage1 = _stage1_exact_name_alias(question, graph)
    spots, _ = spot_entities(question, graph, cutoff=0.6)
    stage2 = [node_id for _start, _end, _text, node_id in spots]
    stage1c = stage1c_property_candidates(question, graph, limit=30)
    combined: list[str] = []
    seen: set[str] = set()
    for node_id in stage1 + stage2 + stage1c:
        if node_id not in seen:
            seen.add(node_id)
            combined.append(node_id)
    stage4 = _stage4_graph_expansion(combined, graph)
    for node_id in stage4:
        if node_id not in seen:
            seen.add(node_id)
            combined.append(node_id)
    return [
        {"node_id": node_id, "lexical_score": _lexical_score(question, graph.get_node(node_id))}
        for node_id in combined
        if graph.get_node(node_id) is not None
    ]


def _evaluate(questions: list[dict[str, Any]], graph: Any, ks: tuple[int, ...]) -> dict[str, Any]:
    counts = {k: {"correct": 0, "gold": 0, "hit_questions": 0, "exact_questions": 0} for k in ks}
    for question in questions:
        ranked = candidate_pool(question["question"], graph)
        gold = set(question["entities"])
        for k in ks:
            predicted = set(rank_candidates(ranked, graph, k))
            counts[k]["correct"] += len(predicted & gold)
            counts[k]["gold"] += len(gold)
            counts[k]["hit_questions"] += int(bool(predicted & gold))
            counts[k]["exact_questions"] += int(gold.issubset(predicted))
    metrics: dict[str, Any] = {}
    total = len(questions)
    for k in ks:
        item = counts[k]
        metrics[f"recall@{k}"] = item["correct"] / item["gold"] if item["gold"] else 0.0
        metrics[f"hit_rate@{k}"] = item["hit_questions"] / total if total else 0.0
        metrics[f"exact_match@{k}"] = item["exact_questions"] / total if total else 0.0
    metrics["questions"] = total
    metrics["gold_entities"] = sum(len(question["entities"]) for question in questions)
    return metrics


def evaluate_validation(
    data_path: str | Path,
    output_dir: str | Path = "benchmarks/results",
    ks: tuple[int, ...] = (1, 5, 10),
) -> Path:
    """Evaluate only the validation split and serialize a provenance-rich result."""
    path = Path(data_path)
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from benchmarks.run_benchmark import build_benchmark_graph

    graph, graph_metadata = build_benchmark_graph()
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_dir) / f"baseline_val_{timestamp}.json"
    artifact = {
        "artifact": "trivial_lexical_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "provenance": {
            "split": "stack/encoder/data/val.jsonl",
            "split_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "candidate_pipeline": ["stage1_exact_name_alias", "stage2_spot_entities", "stage1c_property_overlap", "stage4_graph_expansion"],
            "ranking": "fixed lexical overlap score; ties by total node degree; final tie by node_id",
            "learned_components": False,
            "validation_only": True,
            "graph_metadata": graph_metadata,
        },
        "metrics": _evaluate(questions, graph, ks),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the non-parametric lexical baseline on validation data")
    parser.add_argument("--data", default="stack/encoder/data/val.jsonl")
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()
    print(evaluate_validation(args.data, args.output_dir))


if __name__ == "__main__":
    main()
