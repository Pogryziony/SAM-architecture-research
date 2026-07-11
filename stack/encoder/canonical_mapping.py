"""Canonical entity mapping — graph-derived, deterministic, no frozen test inspection.

Maps granular graph entities (metrics, sub-experiments, concept variants) to
canonical experiment/concept/decision IDs via explicit graph parent edges and
provenance properties. It never reads frozen test labels.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical experiment pattern: Exp_<n>... (uppercase E, digits after underscore)
_EXP_PATTERN = re.compile(r"^Exp_\d+.*$")
# Canonical concept pattern: Concept_<Name> (alphabetic name only, no variants/underscores)
_CONCEPT_PATTERN = re.compile(r"^Concept_[A-Z][a-zA-Z]+$")
_DECISION_PATTERN = re.compile(r"^Decision_[A-Z][a-zA-Z0-9]+$")
_PARENT_EDGE_TYPES = frozenset({"derived_from", "mentioned_in", "belongs_to"})
_PARENT_PROPERTY_KEYS = (
    "parent_id", "parent_entity", "experiment_id", "source_entity", "mentioned_in"
)


def _is_canonical_id(node_id: str) -> bool:
    return bool(
        _EXP_PATTERN.match(node_id)
        or _CONCEPT_PATTERN.match(node_id)
        or _DECISION_PATTERN.match(node_id)
    )


def _parent_ids(node_id: str, graph: Any) -> list[str]:
    """Return deterministic explicit graph/property parents for a node."""
    parents = {
        edge.target
        for edge in graph.get_outgoing(node_id)
        if edge.type in _PARENT_EDGE_TYPES and graph.get_node(edge.target) is not None
    }
    node = graph.get_node(node_id)
    properties = getattr(node, "properties", {}) if node is not None else {}
    if isinstance(properties, dict):
        for key in _PARENT_PROPERTY_KEYS:
            value = properties.get(key)
            values = value if isinstance(value, list) else [value]
            for candidate in values:
                candidate_id = str(candidate or "").strip()
                if candidate_id and graph.get_node(candidate_id) is not None:
                    parents.add(candidate_id)
    return sorted(parents)


def _find_canonical(
    node_id: str, graph: Any, visited: frozenset[str] | None = None, depth: int = 0
) -> str | None:
    """Follow derived_from edges to find the nearest canonical ancestor.

    Returns the first canonical-pattern node encountered during upward
    traversal.  A pattern-matching node is always canonical when returned
    via this function (it is the ancestor another node is looking for).
    """
    if depth > 10:
        return None
    v = visited or frozenset()
    if node_id in v:
        return None  # cycle guard
    v = v | {node_id}

    node = graph.get_node(node_id) if graph is not None else None
    if node is None:
        return None

    # If this node matches a canonical pattern, it IS the canonical ancestor.
    if _is_canonical_id(node_id):
        return node_id

    for parent_id in _parent_ids(node_id, graph):
        result = _find_canonical(parent_id, graph, v, depth + 1)
        if result is not None:
            return result

    return None


def build_canonical_mapping(graph: Any) -> dict[str, str]:
    """Build a deterministic graph-derived canonical entity mapping.

    For each node in the graph, follow ``derived_from`` edges to find
    the nearest canonical ancestor (Exp_* or Concept_* or Decision_*).
    Returns a dict mapping granular node IDs to their canonical ancestors.

    All nodes whose ID matches the canonical patterns (Exp_*, Concept_*,
    Decision_*) are treated as canonical and map to themselves, regardless
    of any derived_from edges they may have.  Only non-pattern-matching
    granular IDs (metrics, sub-variants, etc.) are followed upstream.

    The mapping is:
    - One-to-one for canonical nodes (map to themselves)
    - Many-to-one for granular entities (multiple metrics → one experiment)
    - Missing-parent nodes are excluded (not in mapping)
    - Cycles are guarded by depth limit and visited set
    - No frozen test labels are ever inspected

    Returns:
        dict[str, str]: node_id → canonical_ancestor_id
    """
    mapping: dict[str, str] = {}

    for node_id in sorted(graph._nodes.keys()):
        node = graph.get_node(node_id)
        if node is None:
            continue

        matches_pattern = _is_canonical_id(node_id)

        # Pattern-matching nodes are always canonical, regardless of
        # derived_from edges.  Only non-pattern granular IDs follow
        # upstream edges to locate their canonical ancestor.
        if matches_pattern:
            mapping[node_id] = node_id
            continue

        # Find canonical ancestor via derived_from traversal
        canonical = _find_canonical(node_id, graph)
        if canonical is not None:
            mapping[node_id] = canonical

    return mapping


def apply_canonical_mapping(
    ranked_ids: list[str],
    mapping: dict[str, str],
    top_k: int = 10,
) -> list[str]:
    """Apply canonical mapping to ranked entity IDs, deduplicating after mapping.

    Order of ranked_ids is preserved within each canonical group (first
    granular entity that maps to a canonical target claims that target's
    position).  Mapped entities that collide with an earlier canonical
    entity are dropped.

    Args:
        ranked_ids: Ranked entity IDs (before canonical mapping)
        mapping: Canonical mapping dict from build_canonical_mapping
        top_k: Cap at this many unique canonical entities

    Returns:
        Deduplicated, capped list of canonical entity IDs
    """
    seen: set[str] = set()
    result: list[str] = []

    for node_id in ranked_ids:
        if len(result) >= top_k:
            break
        canonical = mapping.get(node_id, node_id)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result


def export_canonical_mapping_metadata(
    mapping: dict[str, str], graph: Any
) -> dict[str, Any]:
    """Produce provenance metadata about the canonical mapping.

    Does NOT inspect frozen test labels.  Returns mapping statistics.
    """
    total_nodes = len(graph._nodes) if graph is not None else 0
    mapped = len(mapping)
    unique_targets = len(set(mapping.values()))
    self_referential = sum(1 for k, v in mapping.items() if k == v)

    # Many-to-one: count mappings where >1 source maps to same target
    target_counts: dict[str, int] = {}
    for target in mapping.values():
        target_counts[target] = target_counts.get(target, 0) + 1
    many_to_one = sum(1 for count in target_counts.values() if count > 1)

    # Missing-parent nodes: graph nodes not in mapping
    unmapped = total_nodes - mapped

    return {
        "total_graph_nodes": total_nodes,
        "mapped_nodes": mapped,
        "unique_canonical_targets": unique_targets,
        "self_referential_mappings": self_referential,
        "many_to_one_targets": many_to_one,
        "unmapped_nodes": unmapped,
        "edge_type_used": sorted(_PARENT_EDGE_TYPES),
        "property_keys_used": list(_PARENT_PROPERTY_KEYS),
        "canonical_patterns": {
            "experiment": _EXP_PATTERN.pattern,
            "concept": _CONCEPT_PATTERN.pattern,
            "decision": _DECISION_PATTERN.pattern,
        },
        "deterministic": True,
        "frozen_test_inspection": False,
    }
