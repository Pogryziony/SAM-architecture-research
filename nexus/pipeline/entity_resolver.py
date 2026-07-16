"""Entity resolver protocol — defines the interface for entity resolution.

nexus/pipeline/ uses this protocol so it never directly imports stack/.
Concrete implementations (ER3, lexical, dialogue) live in stack/pipeline/
or are injected at runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from nexus.graph.store import InMemoryGraphStore


@dataclass(frozen=True)
class ResolutionCandidate:
    """One entity considered by a resolver before top-K selection."""

    entity_id: str
    score: float | None = None


@dataclass
class ResolutionResult:
    """Auditable entity-resolution result shared by every resolver.

    ``candidate_pool_size`` describes the pool before top-K selection.  It is
    deliberately separate from ``selected_entity_ids`` so pipeline reports do
    not confuse retrieval breadth with the final traversal inputs.
    """

    selected_entity_ids: list[str] = field(default_factory=list)
    candidates: list[ResolutionCandidate] = field(default_factory=list)
    candidate_pool_size: int = 0
    resolver_name: str = "unknown"
    resolver_version: str = "1"
    threshold: float | None = None
    fallback_used: bool = False
    rejection_reason: str = ""
    latency_ms: float = 0.0
    context_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        self.selected_entity_ids = [str(item) for item in self.selected_entity_ids]
        if self.candidate_pool_size < len(self.candidates):
            self.candidate_pool_size = len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_entity_ids": list(self.selected_entity_ids),
            "candidates": [asdict(item) for item in self.candidates],
            "candidate_pool_size": self.candidate_pool_size,
            "resolver_name": self.resolver_name,
            "resolver_version": self.resolver_version,
            "threshold": self.threshold,
            "fallback_used": self.fallback_used,
            "rejection_reason": self.rejection_reason,
            "latency_ms": self.latency_ms,
            "context_latency_ms": self.context_latency_ms,
        }


def coerce_resolution_result(
    value: ResolutionResult | list[str],
    *,
    resolver_name: str = "legacy",
) -> ResolutionResult:
    """Convert the historical ``list[str]`` resolver contract safely.

    The compatibility path keeps existing third-party resolvers working while
    making the missing diagnostics explicit instead of inspecting their
    private attributes.
    """

    if isinstance(value, ResolutionResult):
        return value
    selected = [str(item) for item in value]
    return ResolutionResult(
        selected_entity_ids=selected,
        candidates=[ResolutionCandidate(entity_id=item) for item in selected],
        candidate_pool_size=len(selected),
        resolver_name=resolver_name,
        resolver_version="legacy-list-v1",
    )


@runtime_checkable
class EntityResolver(Protocol):
    """Protocol for pluggable entity resolution.

    Any resolver must accept a question string and a graph, and return
    a list of entity IDs suitable for graph traversal.
    """

    def resolve(
        self, question: str, graph: InMemoryGraphStore
    ) -> ResolutionResult | list[str]:
        """Resolve entity IDs from a question.

        Args:
            question: Natural language question text.
            graph: The populated knowledge graph.

        Returns:
            Structured resolution diagnostics.  ``list[str]`` remains accepted
            temporarily for compatibility and is normalized by NEXUSRunner.
        """
        ...
