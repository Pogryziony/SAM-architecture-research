"""Evaluate the Stage 4 rule corpus against preregistered gates.

Development mode scores ``rule_corpus_v1`` under ``rule-engine-v1``.
Frozen mode scores ``rule_corpus_v1_frozen`` under ``rule-engine-v2`` after
verifying the published frozen file SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
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

PREREGISTRATION_ID_DEV = "rule-engine-v1"
PREREGISTRATION_ID_FROZEN = "rule-engine-v2"
DEFAULT_CORPUS = (
    _project_root / "benchmarks" / "qa-dataset" / "rule_corpus_v1.json"
)
DEFAULT_FROZEN_CORPUS = (
    _project_root / "benchmarks" / "qa-dataset" / "rule_corpus_v1_frozen.json"
)
# Published in EXPERIMENT_RULE_ENGINE_V2.md — fail closed on mismatch.
# Hash is over LF-normalized bytes so Windows/Linux checkouts agree.
FROZEN_FILE_SHA256 = (
    "4a548758f9207a30ace958674f478dd4ee46ee6ca37db9004c9b0ff0b34cb5cf"
)
PRECISION_MIN = 0.90
RECALL_MIN = 0.90
F1_MIN = 0.90
MIN_DEV_RULES = 12


def load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """SHA-256 of path contents with newlines normalized to LF."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


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


def graph_from_payload(payload: dict[str, Any], *, source: str) -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    for node_id in payload["nodes"]:
        graph.add_node(Node(id=str(node_id), type="Entity", sources=[source]))
    for source_id, relation, target in payload["edges"]:
        if not graph.has_node(source_id):
            graph.add_node(Node(id=str(source_id), type="Entity", sources=[source]))
        if not graph.has_node(target):
            graph.add_node(Node(id=str(target), type="Entity", sources=[source]))
        graph.add_edge(
            Edge(
                type=str(relation),
                source=str(source_id),
                target=str(target),
                confidence=1.0,
                evidence=source,
            )
        )
    return graph


def graph_from_corpus(corpus: dict[str, Any]) -> InMemoryGraphStore:
    return graph_from_payload(corpus["development_graph"], source="rule_corpus_v1")


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


def _gate_metrics(metrics: dict[str, float]) -> list[str]:
    errors: list[str] = []
    if metrics["precision"] < PRECISION_MIN:
        errors.append(f"precision {metrics['precision']} < {PRECISION_MIN}")
    if metrics["recall"] < RECALL_MIN:
        errors.append(f"recall {metrics['recall']} < {RECALL_MIN}")
    if metrics["f1"] < F1_MIN:
        errors.append(f"f1 {metrics['f1']} < {F1_MIN}")
    return errors


def evaluate_development(corpus: dict[str, Any]) -> dict[str, Any]:
    if corpus.get("preregistration_id") != PREREGISTRATION_ID_DEV:
        raise RuntimeError("corpus preregistration_id mismatch")
    rules = rules_from_corpus(corpus)
    if len(rules) < MIN_DEV_RULES:
        raise RuntimeError(
            f"rule corpus too small for Stage 4 growth gate ({len(rules)} < {MIN_DEV_RULES})"
        )
    graph = graph_from_corpus(corpus)
    result = RuleEngine(rules, max_depth=3, max_activations=512).evaluate(graph)
    inferred = [(f.source, f.relation, f.target) for f in result.inferred]
    for fact in result.inferred:
        if not fact.rule_id or not fact.premises:
            raise RuntimeError("inferred fact missing rule_id or premises")
    gold = [tuple(item) for item in corpus["development_gold"]]
    metrics = score_inferred(inferred, gold)  # type: ignore[arg-type]
    errors = _gate_metrics(metrics)
    return {
        "status": "PASS" if not errors else "FAIL",
        "mode": "development",
        "preregistration_id": PREREGISTRATION_ID_DEV,
        "corpus_id": corpus.get("corpus_id"),
        "rule_count": len(rules),
        "inferred_count": len(inferred),
        "metrics": metrics,
        "errors": errors,
        "truncated": result.truncated,
    }


def evaluate_frozen(
    frozen_corpus: dict[str, Any],
    *,
    rules_corpus: dict[str, Any],
    frozen_path: Path,
    expected_sha256: str = FROZEN_FILE_SHA256,
) -> dict[str, Any]:
    if frozen_corpus.get("preregistration_id") != PREREGISTRATION_ID_FROZEN:
        raise RuntimeError("frozen corpus preregistration_id mismatch")
    digest = sha256_file(frozen_path)
    if digest != expected_sha256:
        raise RuntimeError(
            f"frozen corpus sha256 mismatch: got {digest}, expected {expected_sha256}"
        )
    rules = rules_from_corpus(rules_corpus)
    graph = graph_from_payload(
        frozen_corpus["frozen_graph"], source="rule_corpus_v1_frozen"
    )
    result = RuleEngine(rules, max_depth=3, max_activations=512).evaluate(graph)
    inferred = [(f.source, f.relation, f.target) for f in result.inferred]
    for fact in result.inferred:
        if not fact.rule_id or not fact.premises:
            raise RuntimeError("inferred fact missing rule_id or premises")
    gold = [tuple(item) for item in frozen_corpus["frozen_gold"]]
    if not gold:
        raise RuntimeError("frozen gold is empty")
    metrics = score_inferred(inferred, gold)  # type: ignore[arg-type]
    errors = _gate_metrics(metrics)
    return {
        "status": "PASS" if not errors else "FAIL",
        "mode": "frozen",
        "preregistration_id": PREREGISTRATION_ID_FROZEN,
        "corpus_id": frozen_corpus.get("corpus_id"),
        "frozen_file_sha256": digest,
        "rule_count": len(rules),
        "inferred_count": len(inferred),
        "metrics": metrics,
        "errors": errors,
        "truncated": result.truncated,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--frozen-corpus", type=Path, default=DEFAULT_FROZEN_CORPUS)
    parser.add_argument("--mode", choices=("development", "frozen"), default="development")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "development":
        corpus = load_corpus(args.corpus)
        report = evaluate_development(corpus)
    else:
        rules_corpus = load_corpus(args.corpus)
        frozen_corpus = load_corpus(args.frozen_corpus)
        report = evaluate_frozen(
            frozen_corpus,
            rules_corpus=rules_corpus,
            frozen_path=args.frozen_corpus,
        )
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
