"""
Graph traversal and path scoring for NEXUS.

Implements beam search traversal with:
- Edge type weighting
- Path scoring (delegates to scoring.py)
- Path deduplication and selection
- Explicit expansion budgets with truncation reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import Path, PathStep
from .store import InMemoryGraphStore
from .scoring import score_path, rank_paths
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG


@dataclass
class TraversalStats:
    """Deterministic expansion diagnostics for a single traversal call."""

    expanded_edges: int = 0
    expanded_nodes: int = 0
    truncated: bool = False
    truncation_reason: str = ""
    max_depth_reached: int = 0
    paths_returned: int = 0

    def to_dict(self) -> dict:
        return {
            "expanded_edges": self.expanded_edges,
            "expanded_nodes": self.expanded_nodes,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "max_depth_reached": self.max_depth_reached,
            "paths_returned": self.paths_returned,
        }


def beam_search(
    graph: InMemoryGraphStore,
    start_nodes: list[str],
    query_entities: set[str],
    max_depth: int | None = None,
    beam_width: int | None = None,
    edge_types: Optional[set[str]] = None,
    direction: str = "both",
    config: NEXUSConfig = DEFAULT_CONFIG,
    stats: TraversalStats | None = None,
) -> list[Path]:
    """
    Beam search traversal: at each depth, expand all paths, score, keep top beam_width.

    When expansion budgets from *config* are exhausted, search stops early and
    ``stats.truncated`` is set. Callers must treat truncation as incomplete search.
    """
    if max_depth is None:
        max_depth = config.max_depth
    if beam_width is None:
        beam_width = config.beam_width
    if stats is None:
        stats = TraversalStats()

    max_edges = max(1, int(getattr(config, "max_expanded_edges", 10_000)))
    max_nodes = max(1, int(getattr(config, "max_expanded_nodes", 5_000)))

    active_paths: list[tuple[str, list[PathStep], set[str]]] = [
        (node, [], {node}) for node in start_nodes if graph.has_node(node)
    ]
    seen_nodes: set[str] = {node for node, _, _ in active_paths}
    stats.expanded_nodes = len(seen_nodes)

    for depth in range(max_depth):
        if stats.truncated:
            break
        candidates: list[tuple[str, list[PathStep], set[str]]] = []

        for current, steps, visited in active_paths:
            if stats.truncated:
                break
            edges = graph.get_edges(current, direction)
            for edge in edges:
                if stats.expanded_edges >= max_edges:
                    stats.truncated = True
                    stats.truncation_reason = "max_expanded_edges"
                    break
                if edge_types and edge.type not in edge_types:
                    continue

                if edge.confidence < min(0.3, getattr(config, 'edge_confidence_threshold', 0.3)):
                    continue

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

                if next_node in visited:
                    continue

                stats.expanded_edges += 1
                seen_nodes.add(next_node)
                stats.expanded_nodes = len(seen_nodes)
                if stats.expanded_nodes >= max_nodes:
                    stats.truncated = True
                    stats.truncation_reason = "max_expanded_nodes"
                step = PathStep(edge=edge, reversed=reversed_flag)
                candidates.append((next_node, steps + [step], visited | {next_node}))
                if stats.truncated:
                    break

        if not candidates:
            break

        scored = []
        for next_node, steps, visited in candidates:
            path = Path(steps=list(steps))
            path.score = score_path(path, query_entities)
            scored.append((next_node, steps, visited, path.score))

        scored.sort(key=lambda x: (-x[3], x[0], tuple(
            (step.edge.type, step.from_node, step.to_node) for step in x[1]
        )))
        active_paths = [(node, steps, visited) for node, steps, visited, _ in scored[:beam_width]]
        stats.max_depth_reached = depth + 1

    results = []
    for _, steps, _ in active_paths:
        if steps:
            p = Path(steps=list(steps))
            p.score = score_path(p, query_entities)
            results.append(p)

    ranked = rank_paths(results, query_entities)
    stats.paths_returned = len(ranked)
    return ranked


def traverse_with_intent(
    graph: InMemoryGraphStore,
    entry_nodes: list[str],
    query_entities: set[str],
    intent: str = "causal_explanation",
    max_depth: int | None = None,
    beam_width: int | None = None,
    config: NEXUSConfig = DEFAULT_CONFIG,
    stats: TraversalStats | None = None,
) -> list[Path]:
    """
    High-level traversal that adapts parameters based on query intent.
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
        stats=stats,
    )
