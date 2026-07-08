"""
Entity deduplication for NEXUS ingestion.

Removes duplicate entities after normalization so that downstream
graph construction does not create multiple near-identical nodes.
"""

from __future__ import annotations

from typing import Any

from nexus.ingestion.normalizer import normalize_entity_name


def deduplicate_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate entities by normalized name (ignoring type).

    Keeps the **first** occurrence and discards later duplicates.
    """
    seen: set[str] = set()
    result: list[dict[str, Any]] = []

    for entity in entities:
        key = normalize_entity_name(entity["name"])
        if key not in seen:
            seen.add(key)
            result.append(entity)

    return result


def merge_entity_lists(lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Merge multiple entity lists and deduplicate.

    Order is preserved within each list, and the first occurrence
    wins across all lists.
    """
    combined: list[dict[str, Any]] = []
    for lst in lists:
        combined.extend(lst)
    return deduplicate_entities(combined)
