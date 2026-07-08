"""
Corpus-level sanity metrics for the NEXUS entity extraction gate.

Computes node validity and edge quality rates directly from the
InMemoryGraphStore produced by the ingestion pipeline.

Usage:
    from experiments.entity_extraction.corpus_quality import corpus_health_report
    report = corpus_health_report(graph)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus.graph.store import InMemoryGraphStore

# -- Re-use from the entity extractor ------------------------------------------

_ENGLISH_STOPWORDS: set[str] = {
    "the", "and", "for", "was", "not", "are", "with", "that", "this",
    "from", "has", "been", "its", "but", "all", "can", "had", "have",
    "use", "new", "one", "two", "way", "each", "set", "run", "see",
    "end", "may", "via", "also", "only", "very", "any", "our", "per",
    "did", "due", "now", "get", "how", "why", "who", "put",
    "big", "old", "key", "top", "low", "few", "ago", "yet", "own",
    "off", "out", "too", "far", "his", "her", "etc",
    # Additional generic/low-signal terms found in extraction noise
    "a", "an", "in", "on", "at", "to", "of", "by", "as", "it",
    "is", "be", "no", "so", "or", "if", "we", "he", "she",
}

# Valid node types from nexus.graph
_VALID_NODE_TYPES: set[str] = {
    "Entity", "Concept", "Document", "CodeFile", "Function",
    "TestCase", "Bug", "Decision", "Requirement", "Experiment", "Metric",
}

# ---------------------------------------------------------------------------
# Node validity heuristics
# ---------------------------------------------------------------------------

def _is_valid_node_name(name: str) -> bool:
    """
    Return True if a node name passes corpus-level sanity checks.

    A valid node must:
      - Have length >= 3 characters after normalization
      - Not be purely numeric (or numeric with punctuation like "2.1")
      - Not be a known English stopword
      - Not contain a pipe character (table row leakage)
      - Not contain raw backtick characters (markdown leakage)
      - Not be just a bare number with a percent sign (e.g. "68.74%")
    """
    stripped = name.strip()

    # Length floor
    if len(stripped) < 3:
        return False

    lower = stripped.lower()

    # Pure numeric (or section-number like "2.1", "3.1.2")
    if re.fullmatch(r'[\d]+(?:\.[\d]+)*\.?', stripped):
        return False

    # Numeric with percent (e.g. "68.74%", "99.87%")
    if re.fullmatch(r'[\d]+(?:\.[\d]+)?%', stripped):
        return False

    # Numeric with leading sigil (e.g. ">2", "~=0.5")
    if re.fullmatch(r'[~><=]*\s*[\d]+(?:\.[\d]+)?', stripped):
        return False

    # Stopword
    if lower in _ENGLISH_STOPWORDS:
        return False

    # Table-row leakage (pipe characters)
    if '|' in stripped:
        return False

    # Markdown backtick leakage
    if '`' in stripped:
        return False

    return True


def _is_valid_node_type(node_type: str) -> bool:
    """Return True if the node type is one of the recognised NODE_TYPE values."""
    return node_type in _VALID_NODE_TYPES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingestion_validity_rate(graph: InMemoryGraphStore) -> float:
    """
    Fraction of graph nodes that pass basic validity heuristics.

    Heuristics:
      - name length >= 3
      - name is not pure numeric
      - name is not a stopword
      - name does not contain backtick or pipe
      - type is a valid NODE_TYPE
    """
    total = graph.node_count
    if total == 0:
        return 1.0

    valid = 0
    invalid_details: list[tuple[str, str, str]] = []

    for node_id, node in graph._nodes.items():
        name_ok = _is_valid_node_name(node_id)
        type_ok = _is_valid_node_type(node.type)
        if name_ok and type_ok:
            valid += 1
        else:
            reasons = []
            if not name_ok:
                reasons.append("bad_name")
            if not type_ok:
                reasons.append(f"bad_type={node.type}")
            invalid_details.append((node_id, node.type, ",".join(reasons)))

    return valid / total


def edge_quality_rate(graph: InMemoryGraphStore) -> float:
    """
    Fraction of edges with confidence >= 0.7.

    Low-confidence edges indicate the relation extractor is guessing.
    """
    total = graph.edge_count
    if total == 0:
        return 1.0

    high_conf = 0
    for node_id in graph._nodes:
        for edge in graph._edges_out.get(node_id, []):
            if edge.confidence >= 0.7:
                high_conf += 1

    return high_conf / total


def corpus_health_report(graph: InMemoryGraphStore) -> dict:
    """
    Combined corpus-level health report.

    Returns a dict with:
      - node_count: total nodes
      - edge_count: total edges
      - node_validity_rate: fraction of nodes passing validity heuristics
      - edge_quality_rate: fraction of edges with confidence >= 0.7
      - invalid_node_examples: up to 10 examples of invalid nodes
      - node_type_distribution: count per type
    """
    total_nodes = graph.node_count
    valid_nodes = 0
    invalid_examples: list[dict] = []

    for node_id, node in graph._nodes.items():
        name_ok = _is_valid_node_name(node_id)
        type_ok = _is_valid_node_type(node.type)
        if name_ok and type_ok:
            valid_nodes += 1
        elif len(invalid_examples) < 10:
            reasons = []
            if not name_ok:
                reasons.append("invalid_name")
            if not type_ok:
                reasons.append(f"invalid_type={node.type}")
            invalid_examples.append({
                "node_id": node_id,
                "type": node.type,
                "reasons": reasons,
            })

    # Node type distribution
    type_dist: dict[str, int] = {}
    for ntype, nids in graph._type_index.items():
        type_dist[ntype] = len(nids)

    # Edge quality
    total_edges = graph.edge_count
    high_conf_edges = 0
    for nid in graph._nodes:
        for edge in graph._edges_out.get(nid, []):
            if edge.confidence >= 0.7:
                high_conf_edges += 1

    return {
        "node_count": total_nodes,
        "edge_count": total_edges,
        "node_validity_rate": round(valid_nodes / total_nodes, 4) if total_nodes > 0 else 1.0,
        "edge_quality_rate": round(high_conf_edges / total_edges, 4) if total_edges > 0 else 1.0,
        "invalid_node_count": total_nodes - valid_nodes,
        "invalid_node_examples": invalid_examples,
        "node_type_distribution": {t: c for t, c in sorted(type_dist.items(), key=lambda x: -x[1])},
    }


# ---------------------------------------------------------------------------
# CLI entry point (for standalone use)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from nexus.graph.store import InMemoryGraphStore
    from nexus.ingestion.ingest_docs import ingest_directory

    _project_root = Path(__file__).resolve().parents[2]

    graph = InMemoryGraphStore()
    for dir_name in ["sam-lm/docs", "sam-lm/experiments"]:
        dir_path = _project_root / dir_name
        if dir_path.exists():
            print(f"Ingesting {dir_name} ...")
            ingest_directory(dir_path, graph, verbose=False)

    report = corpus_health_report(graph)
    print(f"\n{'='*60}")
    print("  CORPUS HEALTH REPORT")
    print(f"{'='*60}")
    print(f"  Nodes:    {report['node_count']}")
    print(f"  Edges:    {report['edge_count']}")
    print(f"  Validity: {report['node_validity_rate']:.4f}")
    print(f"  Edge Ql:  {report['edge_quality_rate']:.4f}")
    print(f"\n  Invalid nodes ({report['invalid_node_count']}):")
    for ex in report["invalid_node_examples"]:
        safe_id = ex["node_id"].encode('ascii', errors='replace').decode('ascii')
        if len(safe_id) > 60:
            safe_id = safe_id[:57] + "..."
        print(f"    [{ex['type']}] \"{safe_id}\"  -- {ex['reasons']}")
    print(f"\n  Type distribution:")
    for t, c in report["node_type_distribution"].items():
        print(f"    {t:<20} {c:>4}")
    print(f"{'='*60}")
