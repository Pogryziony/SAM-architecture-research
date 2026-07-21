"""Optional Kuzu-backed graph store (Stage 4+ production path scaffold).

Implements the subset of InMemoryGraphStore used by traversal/tests:
add_node, add_edge, get_node, has_node, get_outgoing, get_incoming,
node_count, edge_count. Requires the optional ``kuzu`` dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus.graph import Edge, Node


class KuzuGraphStore:
    """Embedded Kuzu adapter with an in-process schema for Node/Edge."""

    def __init__(self, database_path: str | Path | None = None):
        try:
            import kuzu  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(
                "KuzuGraphStore requires the optional 'kuzu' package. "
                "Install with: pip install 'nexus-graph[kuzu]'"
            ) from exc

        self._kuzu = kuzu
        path = str(database_path) if database_path is not None else ":memory:"
        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    def _init_schema(self) -> None:
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Entity("
            "id STRING, type STRING, properties STRING, PRIMARY KEY (id))"
        )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS Rel("
            "FROM Entity TO Entity, type STRING, confidence DOUBLE, evidence STRING)"
        )

    def add_node(self, node: Node) -> None:
        import json

        props = json.dumps(node.properties or {}, sort_keys=True)
        # Upsert via delete+create for scaffold simplicity.
        self._conn.execute("MATCH (n:Entity) WHERE n.id = $id DELETE n", {"id": node.id})
        self._conn.execute(
            "CREATE (n:Entity {id: $id, type: $type, properties: $props})",
            {"id": node.id, "type": node.type, "props": props},
        )
        self._nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise KeyError("both edge endpoints must exist before add_edge")
        self._conn.execute(
            "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
            "CREATE (a)-[:Rel {type: $type, confidence: $conf, evidence: $ev}]->(b)",
            {
                "src": edge.source,
                "tgt": edge.target,
                "type": edge.type,
                "conf": float(edge.confidence),
                "ev": edge.evidence or "",
            },
        )
        self._edges.append(edge)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> list[Edge]:
        return [edge for edge in self._edges if edge.source == node_id]

    def get_incoming(self, node_id: str) -> list[Edge]:
        return [edge for edge in self._edges if edge.target == node_id]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def to_memory_dict(self) -> dict[str, Any]:
        """Debug helper for tests."""
        return {
            "nodes": sorted(self._nodes),
            "edges": [
                (edge.source, edge.type, edge.target) for edge in self._edges
            ],
        }
