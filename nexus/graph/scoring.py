"""
Path scoring algorithms for NEXUS graph traversal.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from . import Path, PathStep, EDGE_TYPE_WEIGHTS

if TYPE_CHECKING:
    from nexus.graph.store import InMemoryGraphStore

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_MENTION_STOP_PARTS = frozenset({
    "exp", "concept", "decision", "the", "and", "of", "to", "in", "for", "a", "an",
})


def entity_mention_score(
    entity_id: str,
    question: str,
    graph: "InMemoryGraphStore | None" = None,
) -> float:
    """Higher means the entity is explicitly grounded in the question text."""
    q_lower = question.casefold()
    q_tokens = {
        token.casefold()
        for token in _TOKEN_RE.findall(question)
        if len(token) > 1
    }
    eid = str(entity_id)
    score = 0.0
    if eid.casefold() in q_lower:
        score += 100.0
    parts = {
        part.casefold()
        for part in eid.split("_")
        if len(part) > 2 and part.casefold() not in _MENTION_STOP_PARTS and not part.isdigit()
    }
    overlap = parts & q_tokens
    if overlap:
        score += 10.0 * len(overlap)
    if graph is None:
        return score
    node = graph.get_node(eid)
    if node is None:
        return score
    for alias in node.aliases:
        alias_text = str(alias).strip()
        if len(alias_text) >= 3 and alias_text.casefold() in q_lower:
            score += 40.0
    for key in ("name", "title", "display_name", "description"):
        value = str((node.properties or {}).get(key, "")).strip()
        if len(value) >= 4 and value.casefold() in q_lower:
            score += 20.0
            break
    return score


def focus_query_entities(
    entity_ids: list[str] | tuple[str, ...],
    path_score_focus: int,
    *,
    question: str = "",
    graph: "InMemoryGraphStore | None" = None,
    max_mention_focus: int = 6,
) -> set[str]:
    """Return the entity set used for path scoring/ranking.

    Expansion may still use the full entry list. Scoring prefers question-
    grounded entries (mentions/aliases), then pads with the leading entry
    prefix. A non-positive ``path_score_focus`` means score against every
    entry entity.
    """
    ids = [str(entity_id) for entity_id in entity_ids if entity_id]
    if not ids:
        return set()
    if not path_score_focus or path_score_focus <= 0:
        return set(ids)

    preferred: list[str] = []
    if question and graph is not None:
        ranked = sorted(
            ids,
            key=lambda eid: (
                -entity_mention_score(eid, question, graph),
                ids.index(eid),
            ),
        )
        preferred = [
            eid for eid in ranked
            if entity_mention_score(eid, question, graph) > 0.0
        ][: max(1, int(max_mention_focus))]

    if preferred:
        focus: list[str] = list(preferred)
        for eid in ids:
            if len(focus) >= int(path_score_focus):
                break
            if eid not in focus:
                focus.append(eid)
        return set(focus)
    return set(ids[: int(path_score_focus)])


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


def select_proof_paths(
    paths: Sequence[Path],
    cover_entities: set[str],
    max_paths: int,
) -> list[Path]:
    """Keep top-ranked paths, then fill remaining slots to cover focus entities.

    Only real ranked paths are selected — never synthetic identity edges.
    """
    ranked = list(paths)
    if max_paths < 1 or len(ranked) <= max_paths:
        return ranked

    keep_ranked = max(1, min(len(ranked), max_paths // 2))
    selected: list[Path] = list(ranked[:keep_ranked])
    selected_ids = {id(path) for path in selected}
    covered = set()
    for path in selected:
        covered |= set(path.nodes) & cover_entities

    remaining = [path for path in ranked if id(path) not in selected_ids]
    while len(selected) < max_paths and remaining:
        missing = cover_entities - covered
        if not missing:
            selected.extend(remaining[: max_paths - len(selected)])
            break
        best = max(
            remaining,
            key=lambda path: (
                len(set(path.nodes) & missing),
                path.score,
                tuple(
                    (step.edge.type, step.from_node, step.to_node)
                    for step in path.steps
                ),
            ),
        )
        if len(set(best.nodes) & missing) <= 0:
            selected.extend(remaining[: max_paths - len(selected)])
            break
        selected.append(best)
        remaining = [path for path in remaining if path is not best]
        covered |= set(best.nodes) & cover_entities
    return selected[:max_paths]


def incident_paths_for_entities(
    graph: "InMemoryGraphStore",
    entity_ids: Sequence[str],
    *,
    max_paths: int = 12,
    as_valid_at: str = "",
    as_known_at: str = "",
) -> list[Path]:
    """Build one-hop paths from real incident edges for audit when traversal is empty.

    Optional bi-temporal cutoffs keep incident enrichment aligned with traversal.
    """
    from nexus.graph.bitemporal import filter_edges_bitemporal

    paths: list[Path] = []
    seen: set[tuple[str, str, str, bool]] = set()
    for entity_id in entity_ids:
        eid = str(entity_id)
        if not graph.has_node(eid):
            continue
        outgoing = list(graph.get_outgoing(eid))
        incoming = list(graph.get_incoming(eid))
        if as_valid_at or as_known_at:
            outgoing = filter_edges_bitemporal(
                outgoing, as_valid_at=as_valid_at, as_known_at=as_known_at,
            )
            incoming = filter_edges_bitemporal(
                incoming, as_valid_at=as_valid_at, as_known_at=as_known_at,
            )
        for edge in outgoing:
            key = (edge.source, edge.type, edge.target, False)
            if key in seen:
                continue
            seen.add(key)
            paths.append(Path(steps=[PathStep(edge=edge, reversed=False)]))
            if len(paths) >= max_paths:
                return paths
        for edge in incoming:
            key = (edge.source, edge.type, edge.target, True)
            if key in seen:
                continue
            seen.add(key)
            paths.append(Path(steps=[PathStep(edge=edge, reversed=True)]))
            if len(paths) >= max_paths:
                return paths
    return paths


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


__all__ = [
    "entity_mention_score",
    "focus_query_entities",
    "incident_paths_for_entities",
    "score_path",
    "rank_paths",
    "select_proof_paths",
    "EDGE_TYPE_WEIGHTS",
]
