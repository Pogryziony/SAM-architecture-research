"""Generic domain-pack interfaces for NEXUS runtime adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore


@dataclass(frozen=True)
class DomainPackMeta:
    domain_id: str
    version: str
    description: str
    locales: tuple[str, ...] = ("en",)


class DomainPack(ABC):
    """Versioned adapter for domain-specific knowledge and evaluation tasks."""

    @property
    @abstractmethod
    def meta(self) -> DomainPackMeta:
        ...

    @abstractmethod
    def build_graph(self) -> InMemoryGraphStore:
        """Construct the domain graph snapshot for evaluation."""

    @abstractmethod
    def entity_aliases(self) -> dict[str, list[str]]:
        """Canonical entity ID → aliases."""

    @abstractmethod
    def evaluation_tasks(self) -> list[dict[str, Any]]:
        """Return evaluation questions for this domain pack."""

    def relation_vocabulary(self) -> frozenset[str]:
        from nexus.graph import EDGE_TYPES

        return EDGE_TYPES

    def answer_schemas(self) -> dict[str, Any]:
        return {"default": {"type": "text"}}

    def provenance(self) -> dict[str, Any]:
        return {
            "domain_id": self.meta.domain_id,
            "domain_pack_version": self.meta.version,
            "description": self.meta.description,
        }


def load_domain_pack(domain_id: str) -> DomainPack:
    """Load a registered domain pack by id."""
    from nexus.domain.mini_pack import MiniDomainPack
    from nexus.domain.sam_pack import SamDomainPack

    registry: dict[str, type[DomainPack]] = {
        "sam": SamDomainPack,
        "mini": MiniDomainPack,
    }
    try:
        cls = registry[domain_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown domain pack {domain_id!r}; known: {sorted(registry)}"
        ) from exc
    return cls()


def populate_store(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    for node in nodes:
        store.add_node(node)
    for edge in edges:
        store.add_edge(edge)
    return store
