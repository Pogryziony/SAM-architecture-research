"""
Graph traversal and path scoring for NEXUS.

Implements beam search traversal with:
- Edge type weighting
- Path scoring (confidence, relevance, coverage, recency)
- Path deduplication and selection
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from . import Edge, Path, EDGE_TYPE_WEIGHTS
from .store import InMemoryGraphStore


def score_path(
    path: Path,
    query_entities: set[str],
    edge_type_weights: dict[str, float] | None = None,
) -> float:
    """
    Score a path for relevance to the query.

    Composite score from:
    - Edge confidence (product)
    - Edge type relevance (product of type weights)
    - Entity coverage (fraction of query entities covered)
    - Path length penalty (mild — shorter paths preferred)
    - Recency bonus (prefer recently-updated sources)
    """
    if not path.edges:
        return 0.0

    weights = edge_type_weights or EDGE_TYPE_WEIGHTS

    # Edge confidence product
    edge_conf = math.prod(e.confidence for e in path.edges)

    # Edge type relevance product
    type_score = math.prod(weights.get(e.type, 0.5) for e in path.edges)

    # Entity coverage: how many query entities appear in the path
    path_entities = {path.edges[0].source}
    for e in path.edges:
        path_entities.add(e.target)
    coverage = len(path_entities & query_entities) / max(len(query_entities), 1)

    # Length penalty: mild decay for longer paths
    length_penalty = 1.0 / (1.0 + 0.1 * len(path.edges))

    # Recency bonus
    now = datetime.now(timezone.utc)
    max_age_days = 365
    for e in path.edges:
        if e.created_at:
            try:
                age = (now - datetime.fromisoformat(e.created_at)).days
                max_age_days = min(max_age_days, age)
            except (ValueError, TypeError):
                pass
    recency = max(0.5, 1.0 - max_age_days / 365)

    return edge_conf * type_score * coverage * length_penalty * recency


def beam_search(
    graph: InMemoryGraphStore,
    start_nodes: list[str],
    query_entities: set[str],
    max_depth: int = 4,
    beam_width: int = 5,
    edge_types: Optional[set[str]] = None,
    direction: str = "both",
) -> list[Path]:
    """
    Beam search traversal: at each depth, expand all paths, score, keep top beam_width.

    Args:
        graph: The graph store
        start_nodes: Entry node IDs
        query_entities: Set of entity names from the query (for scoring)
        max_depth: Maximum path length
        beam_width: Number of paths to keep at each depth
        edge_types: Allowed edge types (None = all)
        direction: Traversal direction ('out', 'in', 'both')

    Returns:
        Ranked list of paths (best first)
    """
    # Initialize: one "path" per start node (no edges yet)
    active_paths: list[tuple[str, list[Edge]]] = [
        (node, []) for node in start_nodes if graph.has_node(node)
    ]

    for _ in range(max_depth):
        candidates: list[tuple[str, list[Edge]]] = []

        for current, path_edges in active_paths:
            edges = graph.get_edges(current, direction)
            for edge in edges:
                if edge_types and edge.type not in edge_types:
                    continue

                # Determine next node
                if direction == "out":
                    next_node = edge.target if edge.source == current else edge.source
                elif direction == "in":
                    next_node = edge.source if edge.target == current else edge.target
                else:  # both
                    next_node = edge.target if edge.source == current else edge.source

                # Avoid cycles
                visited = {path_edges[0].source} if path_edges else {current}
                for pe in path_edges:
                    visited.add(pe.source)
                    visited.add(pe.target)
                if next_node in visited:
                    continue

                candidates.append((next_node, path_edges + [edge]))

        if not candidates:
            break

        # Score all candidates, keep top beam_width
        scored = []
        for next_node, edges in candidates:
            path = Path(edges=list(edges))
            path.score = score_path(path, query_entities)
            scored.append((next_node, edges, path.score))

        scored.sort(key=lambda x: x[2], reverse=True)
        active_paths = [(node, edges) for node, edges, _ in scored[:beam_width]]

    # Return all completed paths, sorted by score
    results = []
    for _, edges in active_paths:
        if edges:
            path = Path(edges=list(edges))
            path.score = score_path(path, query_entities)
            results.append(path)

    results.sort(key=lambda p: p.score, reverse=True)

    # Deduplicate: remove paths that are subpaths of another
    return _deduplicate_paths(results)


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    """Remove paths that are subsets of other paths."""
    if len(paths) <= 1:
        return paths

    # Sort by length (longest first) and score
    paths = sorted(paths, key=lambda p: (p.length, p.score), reverse=True)

    keep = []
    for i, path in enumerate(paths):
        path_nodes = set(path.nodes)
        is_subpath = False
        for j in range(i):
            if path_nodes.issubset(set(paths[j].nodes)):
                is_subpath = True
                break
        if not is_subpath:
            keep.append(path)

    return keep


def traverse_with_intent(
    graph: InMemoryGraphStore,
    entry_nodes: list[str],
    query_entities: set[str],
    intent: str = "causal_explanation",
    max_depth: int = 4,
    beam_width: int = 5,
) -> list[Path]:
    """
    High-level traversal that adapts parameters based on query intent.

    Args:
        graph: The graph store
        entry_nodes: Resolved node IDs for entities in query
        query_entities: Normalized entity name set from query
        intent: Query intent type
        max_depth: Maximum traversal depth
        beam_width: Beam width for search
    """
    intent_config = {
        "causal_explanation": {
            "direction": "in",
            "edge_types": {"caused_by", "blocked_by", "depends_on"},
        },
        "impact_analysis": {
            "direction": "out",
            "edge_types": {"validates", "depends_on", "caused_by", "implements"},
        },
        "factual_lookup": {
            "direction": "both",
            "edge_types": None,
            "max_depth": 1,
        },
        "diagnostic": {
            "direction": "in",
            "edge_types": {"caused_by", "blocked_by", "depends_on"},
        },
        "dependency_chain": {
            "direction": "both",
            "edge_types": {"depends_on", "implements"},
        },
    }

    config = intent_config.get(intent, intent_config["causal_explanation"])

    return beam_search(
        graph=graph,
        start_nodes=entry_nodes,
        query_entities=query_entities,
        max_depth=config.get("max_depth", max_depth),
        beam_width=beam_width,
        edge_types=config.get("edge_types"),
        direction=config.get("direction", "both"),
    )
