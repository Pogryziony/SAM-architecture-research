"""
Graph traversal and path scoring for NEXUS.

Implements beam search traversal with:
- Edge type weighting
- Path scoring (delegates to scoring.py)
- Path deduplication and selection
"""

from __future__ import annotations

from typing import Optional

from . import Edge, Path, PathStep, EDGE_TYPE_WEIGHTS
from .store import InMemoryGraphStore
from .scoring import score_path, rank_paths
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG


def beam_search(
    graph: InMemoryGraphStore,
    start_nodes: list[str],
    query_entities: set[str],
    max_depth: int | None = None,
    beam_width: int | None = None,
    edge_types: Optional[set[str]] = None,
    direction: str = "both",
    config: NEXUSConfig = DEFAULT_CONFIG,
) -> list[Path]:
    """
    Beam search traversal: at each depth, expand all paths, score, keep top beam_width.

    Args:
        graph: The graph store
        start_nodes: Entry node IDs
        query_entities: Set of entity names from the query (for scoring)
        max_depth: Maximum path length (default from config)
        beam_width: Number of paths to keep at each depth (default from config)
        edge_types: Allowed edge types (None = all)
        direction: Traversal direction ('out', 'in', 'both')
        config: NEXUSConfig with tunable parameters

    Returns:
        Ranked list of paths (best first)
    """
    if max_depth is None:
        max_depth = config.max_depth
    if beam_width is None:
        beam_width = config.beam_width
    # Initialize: one "path" per start node (no steps yet)
    active_paths: list[tuple[str, list[PathStep], set[str]]] = [
        (node, [], {node}) for node in start_nodes if graph.has_node(node)
    ]

    for _ in range(max_depth):
        candidates: list[tuple[str, list[PathStep], set[str]]] = []

        for current, steps, visited in active_paths:
            edges = graph.get_edges(current, direction)
            for edge in edges:
                if edge_types and edge.type not in edge_types:
                    continue

                # Determine next node and direction flag
                if direction == "out":
                    if edge.source != current:
                        continue
                    next_node = edge.target
                    reversed_flag = False
                elif direction == "in":
                    if edge.target != current:
                        continue
                    next_node = edge.source
                    reversed_flag = True
                else:  # both
                    if edge.source == current:
                        next_node = edge.target
                        reversed_flag = False
                    elif edge.target == current:
                        next_node = edge.source
                        reversed_flag = True
                    else:
                        continue

                # Cycle protection
                if next_node in visited:
                    continue

                step = PathStep(edge=edge, reversed=reversed_flag)
                candidates.append((next_node, steps + [step], visited | {next_node}))

        if not candidates:
            break

        # Score all candidates, keep top beam_width
        scored = []
        for next_node, steps, visited in candidates:
            path = Path(steps=list(steps))
            path.score = score_path(path, query_entities)
            scored.append((next_node, steps, visited, path.score))

        scored.sort(key=lambda x: x[3], reverse=True)
        active_paths = [(node, steps, visited) for node, steps, visited, _ in scored[:beam_width]]

    # Return all completed paths, sorted by score
    results = []
    for _, steps, _ in active_paths:
        if steps:
            p = Path(steps=list(steps))
            p.score = score_path(p, query_entities)
            results.append(p)

    return rank_paths(results, query_entities)


def traverse_with_intent(
    graph: InMemoryGraphStore,
    entry_nodes: list[str],
    query_entities: set[str],
    intent: str = "causal_explanation",
    max_depth: int | None = None,
    beam_width: int | None = None,
    config: NEXUSConfig = DEFAULT_CONFIG,
) -> list[Path]:
    """
    High-level traversal that adapts parameters based on query intent.

    Args:
        graph: The graph store
        entry_nodes: Resolved node IDs for entities in query
        query_entities: Normalized entity name set from query
        intent: Query intent type
        max_depth: Maximum traversal depth (default from config)
        beam_width: Beam width for search (default from config)
        config: NEXUSConfig with tunable parameters
    """
    if max_depth is None:
        max_depth = config.max_depth
    if beam_width is None:
        beam_width = config.beam_width

    intent_config = {
        "causal_explanation": {
            "direction": "in",
            "edge_types": {"caused_by", "blocked_by", "depends_on", "derived_from"},
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
            "edge_types": {"caused_by", "blocked_by", "depends_on", "derived_from"},
        },
        "dependency_chain": {
            "direction": "both",
            "edge_types": {"depends_on", "implements"},
        },
        "comparison": {
            "direction": "both",
            "edge_types": None,
        },
    }

    intent_params = intent_config.get(intent, intent_config["causal_explanation"])

    return beam_search(
        graph=graph,
        start_nodes=entry_nodes,
        query_entities=query_entities,
        max_depth=intent_params.get("max_depth", max_depth),
        beam_width=beam_width,
        edge_types=intent_params.get("edge_types"),
        direction=intent_params.get("direction", "both"),
        config=config,
    )
