"""SAM research-project domain pack — one explicit DomainPack implementation."""

from __future__ import annotations

from typing import Any

from nexus.domain.pack import DomainPack, DomainPackMeta
from nexus.graph.store import InMemoryGraphStore
from nexus.utils.config import NEXUSConfig


class SamDomainPack(DomainPack):
    """Curated SAM/NEXUS research corpus domain.

    Ownership: ``nexus.ingestion.canonical_graph`` and evaluation datasets under
    ``benchmarks/qa-dataset``. Version identity is ``sam-v1``.
    """

    @property
    def meta(self) -> DomainPackMeta:
        return DomainPackMeta(
            domain_id="sam",
            version="sam-v1",
            description=(
                "SAM architecture research corpus (experiments, decisions, "
                "metrics) used by the internal oracle_v1 contract."
            ),
            locales=("en", "pl"),
        )

    def build_graph(self) -> InMemoryGraphStore:
        from nexus.ingestion.canonical_graph import build_canonical_sam_graph

        graph, _provenance = build_canonical_sam_graph(NEXUSConfig())
        return graph

    def entity_aliases(self) -> dict[str, list[str]]:
        graph = self.build_graph()
        aliases: dict[str, list[str]] = {}
        # Public enumeration via get_all_nodes when available; else safe fallback.
        nodes = getattr(graph, "get_all_nodes", None)
        if callable(nodes):
            iterable = nodes()
        else:
            iterable = [
                graph.get_node(nid)
                for nid in sorted(graph._nodes.keys())  # noqa: SLF001
            ]
        for node in iterable:
            if node is None:
                continue
            if getattr(node, "aliases", None):
                aliases[node.id] = list(node.aliases)
        return aliases

    def evaluation_tasks(self) -> list[dict[str, Any]]:
        from pathlib import Path
        import json

        path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "qa-dataset"
            / "oracle_v1.jsonl"
        )
        if not path.exists():
            return []
        tasks: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("domain", "sam")
            tasks.append(row)
        return tasks
