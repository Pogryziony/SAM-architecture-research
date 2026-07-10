"""Improved entity text representation for the V3 ranker.

Entity text includes: node ID, canonical/display name, node type, aliases,
description, key_finding, source/provenance, and short typed-relation summary.
"""
from __future__ import annotations

from typing import Any


def build_entity_text(node_id: str, graph: Any) -> str:
    """Build a rich text representation of an entity node.

    Includes all available metadata from the graph node:
    - node ID (with underscores replaced by spaces)
    - node type
    - display name / canonical name
    - aliases
    - description
    - key_finding
    - source / provenance
    - typed relation summary (neighbor types and counts)

    Args:
        node_id: The graph node ID.
        graph: InMemoryGraphStore instance.

    Returns:
        A single string with all entity metadata, space-separated.
    """
    node = graph.get_node(node_id) if graph is not None else None
    if node is None:
        return node_id.replace("_", " ")

    parts: list[str] = []

    # Node ID (readable)
    readable_id = node_id.replace("_", " ")
    parts.append(readable_id)

    # Node type
    node_type = str(getattr(node, "type", ""))
    if node_type:
        parts.append(f"type:{node_type}")

    # Properties
    properties = getattr(node, "properties", {}) or {}

    # Display name
    display_name = str(properties.get("display_name", properties.get("name", ""))).strip()
    if display_name:
        parts.append(f"name:{display_name}")

    # Aliases
    aliases = [str(a).strip() for a in getattr(node, "aliases", []) if str(a).strip()]
    if aliases:
        parts.append(f"aliases:{' '.join(aliases[:10])}")

    # Description
    description = str(properties.get("description", "")).strip()
    if description:
        parts.append(f"description:{description}")

    # Key finding
    key_finding = str(properties.get("key_finding", "")).strip()
    if key_finding:
        parts.append(f"finding:{key_finding}")

    # Source / provenance
    source = str(properties.get("source", properties.get("source_snippet", ""))).strip()
    if source:
        parts.append(f"source:{source}")

    # Typed relation summary
    if graph is not None:
        outgoing = graph.get_outgoing(node_id)
        if outgoing:
            from collections import Counter
            edge_types = Counter(e.type for e in outgoing)
            relation_parts = [f"{typ}:{count}" for typ, count in sorted(edge_types.items())]
            if relation_parts:
                parts.append(f"relations:{' '.join(relation_parts)}")

    return " | ".join(parts)


def build_entity_texts(node_ids: list[str], graph: Any) -> list[str]:
    """Build rich text representations for a batch of entity nodes.

    Args:
        node_ids: List of graph node IDs.
        graph: InMemoryGraphStore instance.

    Returns:
        List of entity text strings, one per node_id.
    """
    return [build_entity_text(nid, graph) for nid in node_ids]


def build_entity_text_compact(node_id: str, graph: Any) -> str:
    """Compact variant: ID, type, aliases, and key finding only.

    Suitable for memory-constrained use where full descriptions are too long.
    """
    node = graph.get_node(node_id) if graph is not None else None
    if node is None:
        return node_id.replace("_", " ")

    parts: list[str] = [node_id.replace("_", " ")]

    node_type = str(getattr(node, "type", ""))
    if node_type:
        parts.append(f"type:{node_type}")

    aliases = [str(a).strip() for a in getattr(node, "aliases", []) if str(a).strip()]
    if aliases:
        parts.append(f"aliases:{' '.join(aliases[:5])}")

    properties = getattr(node, "properties", {}) or {}
    key_finding = str(properties.get("key_finding", "")).strip()
    if key_finding:
        parts.append(f"finding:{key_finding}")

    return " | ".join(parts)
