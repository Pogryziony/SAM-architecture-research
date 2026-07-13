"""
Path scoring algorithms for NEXUS graph traversal.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from . import Path, EDGE_TYPE_WEIGHTS


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
    if not path.steps:
        return 0.0

    weights = edge_type_weights or EDGE_TYPE_WEIGHTS

    # Edge confidence product
    edge_conf = math.prod(s.edge.confidence for s in path.steps)

    # Edge type relevance product
    type_score = math.prod(weights.get(s.edge.type, 0.5) for s in path.steps)

    # Entity coverage: how many query entities appear in the path
    path_entities = set(path.nodes)
    coverage = len(path_entities & query_entities) / max(len(query_entities), 1)

    # Length penalty: mild decay for longer paths
    length_penalty = 1.0 / (1.0 + 0.1 * len(path.steps))

    # Recency bonus
    now = datetime.now(timezone.utc)
    max_age_days = 365
    for step in path.steps:
        if step.edge.created_at:
            try:
                age = (now - datetime.fromisoformat(step.edge.created_at)).days
                max_age_days = min(max_age_days, age)
            except (ValueError, TypeError):
                pass
    recency = max(0.5, 1.0 - max_age_days / 365)

    return edge_conf * type_score * coverage * length_penalty * recency


def rank_paths(
    paths: list[Path],
    query_entities: set[str],
    edge_type_weights: dict[str, float] | None = None,
) -> list[Path]:
    """Score and sort paths by relevance. Returns paths with scores set."""
    for path in paths:
        path.score = score_path(path, query_entities, edge_type_weights)
    paths.sort(key=lambda p: (
        -p.score,
        tuple((step.edge.type, step.from_node, step.to_node) for step in p.steps),
    ))
    return _deduplicate_paths(paths)


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    """Remove paths that are subsets of other paths."""
    if len(paths) <= 1:
        return paths

    # Sort by length (longest first) and score
    paths = sorted(paths, key=lambda p: (
        -p.length,
        -p.score,
        tuple((step.edge.type, step.from_node, step.to_node) for step in p.steps),
    ))

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


__all__ = ["score_path", "rank_paths", "EDGE_TYPE_WEIGHTS"]
