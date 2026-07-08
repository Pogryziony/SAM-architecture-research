"""
NEXUS evidence builder — converts graph traversal paths into structured,
compact JSON evidence packs suitable for LLM consumption.

Each evidence pack contains:
  - The original question
  - Structured path data (nodes, edges with direction)
  - Human-readable fact strings with confidence scores
  - Unique source references
"""

from __future__ import annotations

import json
from typing import Any

from nexus.graph import Path, Node
from nexus.graph.store import InMemoryGraphStore


def _node_summary(node: Node) -> dict[str, Any]:
    """Build a compact dict representation of a node."""
    props = dict(node.properties)
    # Include key fields, trim long strings
    summary: dict[str, Any] = {"id": node.id, "type": node.type}
    # Add the most informative property
    for key in ("description", "key_finding", "title", "question", "name"):
        if key in props and props[key]:
            val = props[key]
            if isinstance(val, str) and len(val) > 200:
                val = val[:197] + "..."
            summary[key] = val
    return summary


def _edge_summary(step) -> dict[str, Any]:
    """Build a compact dict for an edge with direction info."""
    return {
        "type": step.edge.type,
        "from": step.from_node,
        "to": step.to_node,
        "confidence": round(step.edge.confidence, 2),
        "reversed": step.reversed,
    }


def _fact_from_step(step, graph: InMemoryGraphStore) -> str:
    """Build a human-readable fact string from a single path step."""
    from_node = graph.get_node(step.from_node)
    to_node = graph.get_node(step.to_node)

    from_name = from_node.id if from_node else step.from_node
    to_name = to_node.id if to_node else step.to_node

    # Choose a readable property if available
    for key in ("name", "display_name", "title"):
        if from_node and key in from_node.properties:
            from_name = from_node.properties[key]
            break
    for key in ("name", "display_name", "title"):
        if to_node and key in to_node.properties:
            to_name = to_node.properties[key]
            break

    relation_map = {
        "depends_on": ("depends on", "is a dependency of"),
        "caused_by": ("is caused by", "causes"),
        "blocked_by": ("is blocked by", "blocks"),
        "validates": ("validates", "is validated by"),
        "contradicts": ("contradicts", "is contradicted by"),
        "implements": ("implements", "is implemented by"),
        "derived_from": ("is derived from", "supports"),
        "replaces": ("replaces", "is replaced by"),
        "related_to": ("is related to", "is related to"),
        "mentioned_in": ("is mentioned in", "mentions"),
    }

    fwd, rev = relation_map.get(step.edge.type, (step.edge.type.replace("_", " "), step.edge.type.replace("_", " ")))
    rel_text = rev if step.reversed else fwd
    confidence = round(step.edge.confidence, 2)
    return f"{from_name} {rel_text} {to_name} (confidence: {confidence:.2f})"


def build_evidence(
    question: str,
    paths: list[Path],
    graph: InMemoryGraphStore,
    max_paths: int = 5,
    max_facts_per_path: int = 10,
) -> str:
    """
    Build a structured JSON evidence pack from traversal paths.

    Args:
        question: The original natural language question
        paths: Ranked traversal paths (best first)
        graph: The graph store for node lookups
        max_paths: Maximum number of paths to include
        max_facts_per_path: Max facts per path

    Returns:
        JSON string with evidence pack
    """
    evidence: dict[str, Any] = {
        "question": question,
        "paths": [],
        "facts": [],
        "sources": [],
    }

    all_sources: set[str] = set()
    all_facts: list[str] = []

    for path in paths[:max_paths]:
        if not path.steps:
            continue

        path_data: dict[str, Any] = {
            "score": round(path.score, 3),
            "length": path.length,
            "nodes": [],
            "edges": [],
        }

        # Collect unique nodes along the path
        seen_nodes: set[str] = set()
        for step in path.steps:
            for node_id in (step.from_node, step.to_node):
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    node = graph.get_node(node_id)
                    if node:
                        path_data["nodes"].append(_node_summary(node))
                        for src in node.sources:
                            all_sources.add(src)

        # Add edges
        for step in path.steps[:max_facts_per_path]:
            path_data["edges"].append(_edge_summary(step))
            # Build fact string
            fact = _fact_from_step(step, graph)
            all_facts.append(fact)
            # Add edge evidence as source
            if step.edge.evidence:
                all_sources.add(step.edge.evidence)

        evidence["paths"].append(path_data)

    evidence["facts"] = all_facts
    evidence["sources"] = sorted(all_sources)

    return json.dumps(evidence, indent=2, ensure_ascii=False)


def build_evidence_pack(
    question: str,
    paths: list[Path],
    graph: InMemoryGraphStore,
) -> dict[str, Any]:
    """
    Build and return the evidence pack as a Python dict (no JSON serialization).

    Useful for programmatic access or further processing.
    """
    raw = build_evidence(question, paths, graph)
    return json.loads(raw)
