"""Evaluate the Stage 4 rule corpus against preregistered development gates.

Frozen mode is sealed until a future preregistration publishes the frozen
split hash. Development mode scores the embedded gold in rule_corpus_v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.rules import Rule, RuleEngine

PREREGISTRATION_ID = "rule-engine-v1"
DEFAULT_CORPUS = (
    _project_root / "benchmarks" / "qa-dataset" / "rule_corpus_v1.json"
)
PRECISION_MIN = 0.90
RECALL_MIN = 0.90
F1_MIN = 0.90


def load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rules_from_corpus(corpus: dict[str, Any]) -> list[Rule]:
    rules: list[Rule] = []
    for item in corpus["rules"]:
        body = tuple(tuple(atom) for atom in item["body"])
        head = tuple(item["head"])
        rules.append(
            Rule(
                rule_id=str(item["rule_id"]),
                version=str(item["version"]),
                body=body,  # type: ignore[arg-type]
                head=head,  # type: ignore[arg-type]
            )
        )
    return rules


def graph_from_corpus(corpus: dict[str, Any]) -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    payload = corpus["development_graph"]
    for node_id in payload["nodes"]:
        graph.add_node(Node(id=str(node_id), type="Entity", sources=["rule_corpus_v1"]))
    for source, relation, target in payload["edges"]:
        if not graph.has_node(source):
            graph.add_node(Node(id=str(source), type="Entity", sources=["rule_corpus_v1"]))
        if not graph.has_node(target):
            graph.add_node(Node(id=str(target), type="Entity", sources=["rule_corpus_v1"]))
        graph.add_edge(
            Edge(
                type=str(relation),
                source=str(source),
                target=str(target),
                confidence=1.0,
                evidence="rule_corpus_v1",
            )
        )
    return graph


def score_inferred(
    inferred: list[tuple[str, str, str]],
    gold: list[tuple[str, str, str]],
) -> dict[str, float]:
    pred = set(inferred)
    truth = set(gold)
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": float(tp),
        "fp": float(len(pred - truth)),
        "fn": float(len(truth - pred)),
    }


def evaluate_development(corpus: dict[str, Any]) -> dict[str, Any]:
    if corpus.get("preregistration_id") != PREREGISTRATION_ID:
        raise RuntimeError("corpus preregistration_id mismatch")
    rules = rules_from_corpus(corpus)
    if len(rules) < 6:
        raise RuntimeError("rule corpus too small for Stage 4 growth gate")
    graph = graph_from_corpus(corpus)
    result = RuleEngine(rules, max_depth=3, max_activations=256).evaluate(graph)
    inferred = [(f.source, f.relation, f.target) for f in result.inferred]
    for fact in result.inferred:
        if not fact.rule_id or not fact.premises:
            raise RuntimeError("inferred fact missing rule_id or premises")
    gold = [tuple(item) for item in corpus["development_gold"]]
    metrics = score_inferred(inferred, gold)  # type: ignore[arg-type]
    errors: list[str] = []
    if metrics["precision"] < PRECISION_MIN:
        errors.append(f"precision {metrics['precision']} < {PRECISION_MIN}")
    if metrics["recall"] < RECALL_MIN:
        errors.append(f"recall {metrics['recall']} < {RECALL_MIN}")
    if metrics["f1"] < F1_MIN:
        errors.append(f"f1 {metrics['f1']} < {F1_MIN}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "mode": "development",
        "preregistration_id": PREREGISTRATION_ID,
        "corpus_id": corpus.get("corpus_id"),
        "rule_count": len(rules),
        "inferred_count": len(inferred),
        "metrics": metrics,
        "errors": errors,
        "truncated": result.truncated,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--mode", choices=("development", "frozen"), default="development")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "frozen":
        raise RuntimeError(
            "frozen rule evaluation is sealed until a future preregistration "
            "publishes the frozen split hash (see EXPERIMENT_RULE_ENGINE_V1.md)"
        )
    corpus = load_corpus(args.corpus)
    report = evaluate_development(corpus)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
