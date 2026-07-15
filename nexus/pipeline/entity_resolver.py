"""Entity resolver protocol — defines the interface for entity resolution.

nexus/pipeline/ uses this protocol so it never directly imports stack/.
Concrete implementations (ER3, lexical, dialogue) live in stack/pipeline/
or are injected at runtime.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from nexus.graph.store import InMemoryGraphStore


@runtime_checkable
class EntityResolver(Protocol):
    """Protocol for pluggable entity resolution.

    Any resolver must accept a question string and a graph, and return
    a list of entity IDs suitable for graph traversal.
    """

    def resolve(self, question: str, graph: InMemoryGraphStore) -> list[str]:
        """Resolve entity IDs from a question.

        Args:
            question: Natural language question text.
            graph: The populated knowledge graph.

        Returns:
            List of entity IDs, ordered by relevance (highest first).
        """
        ...
