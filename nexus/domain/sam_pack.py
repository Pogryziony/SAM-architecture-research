"""SAM research-project domain pack — one explicit DomainPack implementation."""

from __future__ import annotations

from typing import Any

from nexus.domain.pack import DomainPack, DomainPackMeta
from nexus.graph.store import InMemoryGraphStore
from nexus.utils.config import NEXUSConfig


class SamDomainPack(DomainPack):
    """Curated SAM/NEXUS research corpus domain.

    Ownership: repository ingestion under ``nexus/ingestion`` and
    ``benchmarks/run_benchmark.build_benchmark_graph``. Version identity is
    ``sam-v1`` and must be recorded on evaluation artifacts.
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
        from benchmarks.run_benchmark import build_benchmark_graph

        graph, _provenance = build_benchmark_graph(NEXUSConfig())
        return graph

    def entity_aliases(self) -> dict[str, list[str]]:
        graph = self.build_graph()
        aliases: dict[str, list[str]] = {}
        for node_id, node in graph._nodes.items():  # noqa: SLF001 — pack adapter
            if node.aliases:
                aliases[node_id] = list(node.aliases)
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
