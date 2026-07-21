"""
Relation extraction evaluation for NEXUS.

Loads a gold-standard relation dataset, extracts relations from the graph,
and computes precision/recall/F1 per edge type with strict separation of
semantic edges (typed, confidence > 0.3) from co-occurrence edges
("related_to", confidence = 0.3).

Co-occurrence edges are treated as background links — never counted as
correct for precision/recall calculations. Only semantic (typed) edges
with confidence > 0.3 are evaluated against the gold standard.

Usage:
    python experiments/relation-extraction/evaluate_relations.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GOLD_FILE = _project_root / "benchmarks" / "qa-dataset" / "relation_gold.jsonl"

# ---------------------------------------------------------------------------
# Semantic edge types (typed, non-co-occurrence)
# ---------------------------------------------------------------------------
# Conceptual / typed relations scored against gold.
# Structural parentage (sub_experiment) is excluded — same role as
# co-occurrence for precision: useful in the graph, not a gold FP.
SEMANTIC_EDGE_TYPES = frozenset({
    "derived_from", "validates", "depends_on", "caused_by",
    "blocked_by", "implements", "contradicts", "replaces", "mentioned_in",
})

STRUCTURAL_EDGE_TYPES = frozenset({"sub_experiment"})

COOCCURRENCE_EDGE_TYPE = "related_to"
COOCCURRENCE_CONFIDENCE = 0.3


def load_gold(gold_path: Path) -> tuple[list[dict], list[dict]]:
    """Load the gold-standard relation dataset.

    Returns:
        (positive_examples, negative_examples)
    """
    positives = []
    negatives = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("is_negative", False):
                negatives.append(entry)
            else:
                positives.append(entry)
    return positives, negatives


def extract_semantic_edges(graph: InMemoryGraphStore) -> set[tuple]:
    """Extract semantic edges from the graph.

    Semantic edges: types in ``SEMANTIC_EDGE_TYPES`` with confidence > 0.3.
    Co-occurrence and structural (``sub_experiment``) edges are excluded.

    Returns:
        Set of (source, target, edge_type) tuples.
    """
    semantic = set()
    for source_nid in list(graph._nodes.keys()):
        for edge in graph.get_outgoing(source_nid):
            if edge.type == COOCCURRENCE_EDGE_TYPE:
                continue
            if edge.type in STRUCTURAL_EDGE_TYPES:
                continue
            if edge.type not in SEMANTIC_EDGE_TYPES:
                continue
            if edge.confidence > COOCCURRENCE_CONFIDENCE:
                semantic.add((edge.source, edge.target, edge.type))
    return semantic


def extract_structural_edges(graph: InMemoryGraphStore) -> list[dict]:
    """Extract structural parentage edges (``sub_experiment``)."""
    structural = []
    for source_nid in list(graph._nodes.keys()):
        for edge in graph.get_outgoing(source_nid):
            if edge.type in STRUCTURAL_EDGE_TYPES:
                structural.append({
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.type,
                    "confidence": edge.confidence,
                })
    return structural


def extract_cooccurrence_edges(graph: InMemoryGraphStore) -> list[dict]:
    """Extract co-occurrence edges from the graph.

    Co-occurrence edges: "related_to" with confidence exactly 0.3.
    These are background links, never counted for precision.
    """
    cooccurrence = []
    for source_nid in list(graph._nodes.keys()):
        for edge in graph.get_outgoing(source_nid):
            if edge.type == COOCCURRENCE_EDGE_TYPE:
                cooccurrence.append({
                    "source": edge.source,
                    "target": edge.target,
                    "confidence": edge.confidence,
                })
    return cooccurrence


def evaluate(
    gold_positives: list[dict],
    gold_negatives: list[dict],
    semantic_edges: set[tuple],
) -> dict:
    """Evaluate semantic edge extraction against gold standard.

    Computes per-edge-type precision, recall, and F1.
    Also tracks global false positives (edges in graph not in gold)
    and false negatives (edges in gold not in graph).
    """
    gold_set: set[tuple] = set()
    gold_by_type: dict[str, set[tuple]] = defaultdict(set)
    for entry in gold_positives:
        key = (entry["source"], entry["target"], entry["edge_type"])
        gold_set.add(key)
        gold_by_type[entry["edge_type"]].add(key)

    # Per-edge-type metrics
    per_type = {}
    for etype in sorted(gold_by_type.keys()):
        gold_type_set = gold_by_type[etype]
        pred_type_set = {k for k in semantic_edges if k[2] == etype}

        tp = len(pred_type_set & gold_type_set)
        precision = tp / len(pred_type_set) if pred_type_set else 0.0
        recall = tp / len(gold_type_set) if gold_type_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_type[etype] = {
            "gold_count": len(gold_type_set),
            "predicted_count": len(pred_type_set),
            "tp": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Global metrics
    tp_global = len(semantic_edges & gold_set)
    precision_global = tp_global / len(semantic_edges) if semantic_edges else 0.0
    recall_global = tp_global / len(gold_set) if gold_set else 0.0
    f1_global = (
        2 * precision_global * recall_global / (precision_global + recall_global)
        if (precision_global + recall_global) > 0
        else 0.0
    )

    # False positives: edges in graph (semantic) not in gold
    false_positives = []
    for edge_key in sorted(semantic_edges - gold_set):
        false_positives.append({
            "source": edge_key[0],
            "target": edge_key[1],
            "edge_type": edge_key[2],
        })

    # False negatives: edges in gold not in graph
    false_negatives = []
    for edge_key in sorted(gold_set - semantic_edges):
        false_negatives.append({
            "source": edge_key[0],
            "target": edge_key[1],
            "edge_type": edge_key[2],
        })

    # Check negative examples: any semantic edge matching a negative is an error
    negative_set = set()
    for entry in gold_negatives:
        key = (entry["source"], entry["target"], entry["edge_type"])
        negative_set.add(key)
    false_positive_negatives = sorted(semantic_edges & negative_set)

    return {
        "global": {
            "gold_count": len(gold_set),
            "predicted_count": len(semantic_edges),
            "tp": tp_global,
            "precision": round(precision_global, 4),
            "recall": round(recall_global, 4),
            "f1": round(f1_global, 4),
        },
        "per_type": per_type,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "negative_examples_hit": false_positive_negatives,
    }


def print_report(
    results: dict,
    cooccurrence_count: int,
    structural_count: int = 0,
) -> None:
    """Print a human-readable evaluation report."""
    print("=" * 72)
    print("RELATION EXTRACTION EVALUATION REPORT")
    print("=" * 72)

    g = results["global"]
    print(f"\nGlobal (semantic edges only, confidence > {COOCCURRENCE_CONFIDENCE}):")
    print(f"  Gold pairs:     {g['gold_count']}")
    print(f"  Predicted:      {g['predicted_count']}")
    print(f"  True positives: {g['tp']}")
    print(f"  Precision:      {g['precision']:.4f}")
    print(f"  Recall:         {g['recall']:.4f}")
    print(f"  F1:             {g['f1']:.4f}")

    print(f"\n  Co-occurrence edges (excluded from eval): {cooccurrence_count}")
    print(f"  (Co-occurrence = '{COOCCURRENCE_EDGE_TYPE}' edges with confidence = {COOCCURRENCE_CONFIDENCE})")
    print(f"  Structural edges (excluded from eval): {structural_count}")
    print(f"  (Structural = {sorted(STRUCTURAL_EDGE_TYPES)})")

    print("\n" + "-" * 72)
    print("Per edge type:")
    print(f"  {'Edge Type':<20s} {'Gold':>5s} {'Pred':>5s} {'TP':>4s}  {'Prec':>8s} {'Rec':>8s} {'F1':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*4}  {'-'*8} {'-'*8} {'-'*8}")
    for etype, metrics in sorted(results["per_type"].items()):
        print(
            f"  {etype:<20s} {metrics['gold_count']:>5d} {metrics['predicted_count']:>5d} "
            f"{metrics['tp']:>4d}  {metrics['precision']:>8.4f} {metrics['recall']:>8.4f} "
            f"{metrics['f1']:>8.4f}"
        )

    if results["false_negatives"]:
        print(f"\n  False negatives ({len(results['false_negatives'])} edges in gold not found):")
        for fn in results["false_negatives"]:
            print(f"    {fn['source']} --[{fn['edge_type']}]--> {fn['target']}")

    if results["false_positives"]:
        fp_count = len(results["false_positives"])
        print(f"\n  False positives ({fp_count} edges in graph not in gold):")
        if fp_count <= 20:
            for fp in results["false_positives"]:
                print(f"    {fp['source']} --[{fp['edge_type']}]--> {fp['target']}")
        else:
            for fp in results["false_positives"][:10]:
                print(f"    {fp['source']} --[{fp['edge_type']}]--> {fp['target']}")
            print(f"    ... and {fp_count - 10} more")

    if results["negative_examples_hit"]:
        print(f"\n  ⚠  Negative examples incorrectly extracted ({len(results['negative_examples_hit'])}):")
        for ne in results["negative_examples_hit"]:
            print(f"    {ne[0]} --[{ne[2]}]--> {ne[1]} (expected: NOT related)")

    print("\n" + "=" * 72)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for JSON results (default: do not write).",
    )
    args = parser.parse_args(argv)

    if not GOLD_FILE.exists():
        print(f"ERROR: Gold file not found: {GOLD_FILE}")
        return 1
    positives, negatives = load_gold(GOLD_FILE)
    print(f"Loaded gold dataset: {len(positives)} positive, {len(negatives)} negative examples")

    graph = InMemoryGraphStore()
    graph = populate_graph(EXPERIMENTS_DIR, graph)
    print(f"Graph: {graph.node_count} nodes, {graph.edge_count} edges")

    semantic_edges = extract_semantic_edges(graph)
    cooccurrence_edges = extract_cooccurrence_edges(graph)
    structural_edges = extract_structural_edges(graph)
    print(
        f"Semantic edges: {len(semantic_edges)}, "
        f"Co-occurrence edges: {len(cooccurrence_edges)}, "
        f"Structural edges: {len(structural_edges)}"
    )

    results = evaluate(positives, negatives, semantic_edges)
    print_report(results, len(cooccurrence_edges), len(structural_edges))

    if args.output is not None:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {out_path}")
    else:
        print("\nNo --output provided; results not written to disk.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
