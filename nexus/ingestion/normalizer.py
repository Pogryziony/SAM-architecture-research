"""
Entity name normalization and canonicalization for NEXUS ingestion.

Provides consistent normalization so that variants of the same entity
(underscores vs spaces, different casing) map to the same key for
deduplication and graph-lookup.
"""

from __future__ import annotations

import re


def normalize_entity_name(name: str) -> str:
    """
    Produce a normalized, comparable key for an entity name.

    Rules (in order):
      1. Strip leading/trailing whitespace.
      2. Lowercase.
      3. Replace hyphens with spaces.
      4. Replace underscores with spaces.
      5. Remove trailing punctuation (commas, periods, colons).
      6. Collapse multiple spaces.
    """
    name = name.strip()
    name = name.lower()
    name = name.replace("-", " ")
    name = name.replace("_", " ")
    # Remove standalone leading punctuation/symbols from headers like "## 7.1 Blah"
    name = re.sub(r'^[\d.]+\s+', '', name)
    # Remove trailing punctuation
    name = re.sub(r'[.,;:!?]+$', '', name)
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def canonicalize(name: str, entity_type: str = "Entity") -> str:
    """
    Produce a canonical node ID for Neo4j / internal storage.

    Strategy:
      - For Concepts / Experiments / Functions / Metrics, use Title_Case
        with underscores, e.g. "Oracle_Memory".
      - For CodeFiles, use the original relative path casing.
      - For everything else, use slug_case (lower, underscores).
    """
    normalized = normalize_entity_name(name)

    # Types that benefit from Title_Case IDs
    title_case_types = {"Concept", "Experiment", "Function", "Metric",
                        "Bug", "Decision", "Requirement", "TestCase", "Document"}

    if entity_type in title_case_types:
        # Title Case each word, join with underscores
        words = normalized.split()
        return "_".join(w[0].upper() + w[1:] for w in words if w)

    # Default: slug_case (lowercase with underscores)
    return normalized.replace(" ", "_")
