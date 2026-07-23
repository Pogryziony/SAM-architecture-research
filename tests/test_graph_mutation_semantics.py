"""Graph mutation semantics: indexes, edge updates, type changes."""

from __future__ import annotations

from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore


def test_node_type_change_updates_type_index():
    g = InMemoryGraphStore()
    g.add_node(Node(id="X", type="Entity"))
    assert [n.id for n in g.nodes_of_type("Entity")] == ["X"]
    g.add_node(Node(id="X", type="Concept"))
    assert [n.id for n in g.nodes_of_type("Entity")] == []
    assert [n.id for n in g.nodes_of_type("Concept")] == ["X"]


def test_alias_removal_clears_stale_alias_index():
    g = InMemoryGraphStore()
    g.add_node(Node(id="City", type="Entity", aliases=["Warsaw"]))
    assert g.find_entity_exact("Warsaw") == "City"
    g.add_node(Node(id="City", type="Entity", aliases=["Capital"]))
    assert g.find_entity_exact("Warsaw") is None
    assert g.find_entity_exact("Capital") == "City"


def test_property_removal_clears_keyword_index():
    g = InMemoryGraphStore()
    g.add_node(
        Node(
            id="Exp",
            type="Experiment",
            properties={"key_finding": "oracle memory accuracy ninety"},
        )
    )
    assert g.find_entity_by_keywords("oracle memory accuracy")
    g.add_node(Node(id="Exp", type="Experiment", properties={}))
    assert g.find_entity_by_keywords("oracle memory accuracy") == []
    assert g.get_property_text("Exp") == ""


def test_edge_update_preserves_evidence_and_temporal_fields():
    g = InMemoryGraphStore()
    g.add_node(Node(id="A", type="Entity"))
    g.add_node(Node(id="B", type="Entity"))
    g.add_edge(
        Edge(
            type="depends_on",
            source="A",
            target="B",
            confidence=0.5,
            evidence="old.md",
            observed_at="2024-01-01T00:00:00",
        )
    )
    stored = g.add_edge(
        Edge(
            type="depends_on",
            source="A",
            target="B",
            confidence=0.9,
            evidence="new.md",
            observed_at="2025-06-01T00:00:00",
            retracted_at="2026-01-01T00:00:00",
        )
    )
    assert g.edge_count == 1
    out = g.get_outgoing("A")[0]
    assert out is stored
    assert out.confidence == 0.9
    assert out.evidence == "new.md"
    assert out.observed_at == "2025-06-01T00:00:00"
    assert out.retracted_at == "2026-01-01T00:00:00"


def test_kuzu_store_is_marked_experimental():
    from nexus.graph.kuzu_store import KuzuGraphStore

    assert KuzuGraphStore.EXPERIMENTAL is True
    assert KuzuGraphStore.AUTHORITATIVE_BACKEND is False
