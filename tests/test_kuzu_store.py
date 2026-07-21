"""Optional Kuzu store scaffold tests (skip when kuzu is unavailable)."""

from __future__ import annotations

import pytest

pytest.importorskip("kuzu")

from nexus.graph import Edge, Node
from nexus.graph.kuzu_store import KuzuGraphStore


def test_kuzu_round_trip_nodes_and_edges(tmp_path):
    store = KuzuGraphStore(tmp_path / "toy.kuzu")
    store.add_node(Node(id="A", type="Entity", properties={"name": "a"}))
    store.add_node(Node(id="B", type="Entity", properties={"name": "b"}))
    store.add_edge(Edge(type="depends_on", source="A", target="B", confidence=0.9))
    assert store.has_node("A")
    assert store.get_node("B") is not None
    assert store.node_count == 2
    assert store.edge_count == 1
    outgoing = store.get_outgoing("A")
    assert len(outgoing) == 1
    assert outgoing[0].target == "B"
