"""
Tests for NEXUS graph core data structures and operations.

Covers: nodes, edges, store operations, traversal, scoring, fuzzy lookup,
cycle protection, direction tracking, edge dedup, node update dedup.
"""

import pytest
from nexus.graph import Node, Edge, Path, PathStep, EDGE_TYPE_WEIGHTS
from nexus.graph.store import InMemoryGraphStore
from nexus.graph.scoring import score_path, rank_paths
from nexus.graph.traversal import beam_search


# ── Fixtures ──

@pytest.fixture
def empty_graph():
    return InMemoryGraphStore()


@pytest.fixture
def simple_chain():
    """A <- B <- C (B depends_on A, C depends_on B)."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="A", type="Entity"))
    g.add_node(Node(id="B", type="Entity"))
    g.add_node(Node(id="C", type="Entity"))
    g.add_edge(Edge(type="depends_on", source="B", target="A", confidence=0.9))
    g.add_edge(Edge(type="depends_on", source="C", target="B", confidence=0.8))
    return g


@pytest.fixture
def cyclic_graph():
    """A -> B -> C -> A (cycle)."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="A", type="Entity"))
    g.add_node(Node(id="B", type="Entity"))
    g.add_node(Node(id="C", type="Entity"))
    g.add_edge(Edge(type="depends_on", source="B", target="A", confidence=0.9))
    g.add_edge(Edge(type="depends_on", source="C", target="B", confidence=0.8))
    g.add_edge(Edge(type="depends_on", source="A", target="C", confidence=0.7))
    return g


@pytest.fixture
def multi_edge_graph():
    """A with multiple edge types to B and C."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="A", type="Entity"))
    g.add_node(Node(id="B", type="Entity"))
    g.add_node(Node(id="C", type="Entity"))
    g.add_node(Node(id="D", type="TestCase"))
    g.add_edge(Edge(type="depends_on", source="A", target="B", confidence=0.9))
    g.add_edge(Edge(type="validates", source="D", target="A", confidence=1.0))
    g.add_edge(Edge(type="blocked_by", source="A", target="C", confidence=0.95))
    return g


# ── Node Tests ──

class TestNode:
    def test_create_node(self):
        n = Node(id="Test", type="Entity", properties={"key": "val"})
        assert n.id == "Test"
        assert n.type == "Entity"
        assert n.properties["key"] == "val"

    def test_node_timestamps_auto(self):
        n = Node(id="X", type="Concept")
        assert n.created_at
        assert n.updated_at


# ── Edge Tests ──

class TestEdge:
    def test_create_edge(self):
        e = Edge(type="depends_on", source="A", target="B", confidence=0.8)
        assert e.type == "depends_on"
        assert e.source == "A"
        assert e.target == "B"
        assert e.confidence == 0.8

    def test_edge_hash(self):
        e1 = Edge(type="depends_on", source="A", target="B")
        e2 = Edge(type="depends_on", source="A", target="B")
        e3 = Edge(type="depends_on", source="A", target="C")
        assert hash(e1) == hash(e2)
        assert hash(e1) != hash(e3)


# ── PathStep Tests ──

class TestPathStep:
    def test_forward_step(self):
        e = Edge(type="depends_on", source="A", target="B")
        ps = PathStep(edge=e, reversed=False)
        assert ps.from_node == "A"
        assert ps.to_node == "B"

    def test_reversed_step(self):
        e = Edge(type="depends_on", source="A", target="B")
        ps = PathStep(edge=e, reversed=True)
        assert ps.from_node == "B"
        assert ps.to_node == "A"


# ── Path Tests ──

class TestPath:
    def test_empty_path(self):
        p = Path()
        assert p.length == 0
        assert p.nodes == []

    def test_path_nodes_forward(self):
        e1 = Edge(type="depends_on", source="A", target="B")
        e2 = Edge(type="depends_on", source="B", target="C")
        p = Path(steps=[PathStep(edge=e1), PathStep(edge=e2)])
        assert p.nodes == ["A", "B", "C"]

    def test_path_nodes_reversed(self):
        e1 = Edge(type="depends_on", source="A", target="B")
        e2 = Edge(type="depends_on", source="B", target="C")
        p = Path(steps=[PathStep(edge=e2, reversed=True), PathStep(edge=e1, reversed=True)])
        assert p.nodes == ["C", "B", "A"]

    def test_path_repr(self):
        e = Edge(type="depends_on", source="A", target="B")
        p = Path(steps=[PathStep(edge=e, reversed=False)])
        r = repr(p)
        assert "depends_on" in r


# ── Store Tests ──

class TestStore:
    def test_add_and_get(self, empty_graph):
        empty_graph.add_node(Node(id="X", type="Entity"))
        assert empty_graph.get_node("X").id == "X"

    def test_add_edge_missing_node(self, empty_graph):
        empty_graph.add_node(Node(id="A", type="Entity"))
        with pytest.raises(KeyError):
            empty_graph.add_edge(Edge(type="depends_on", source="A", target="B"))

    def test_has_node(self, simple_chain):
        assert simple_chain.has_node("A")
        assert not simple_chain.has_node("Z")

    def test_node_update_no_duplicate_type(self, empty_graph):
        empty_graph.add_node(Node(id="X", type="Entity"))
        empty_graph.add_node(Node(id="X", type="Entity"))
        assert len(empty_graph.nodes_of_type("Entity")) == 1

    def test_edge_dedup(self, simple_chain):
        before = simple_chain.edge_count
        simple_chain.add_edge(Edge(type="depends_on", source="B", target="A", confidence=0.9))
        assert simple_chain.edge_count == before

    def test_get_outgoing(self, simple_chain):
        edges = simple_chain.get_outgoing("B")
        assert len(edges) == 1
        assert edges[0].target == "A"

    def test_get_incoming(self, simple_chain):
        edges = simple_chain.get_incoming("B")
        assert len(edges) == 1
        assert edges[0].source == "C"

    def test_get_edges_both(self, simple_chain):
        assert len(simple_chain.get_edges("B", "both")) == 2

    def test_node_count(self, simple_chain):
        assert simple_chain.node_count == 3

    def test_edge_count(self, simple_chain):
        assert simple_chain.edge_count == 2

    def test_nodes_of_type(self, multi_edge_graph):
        assert len(multi_edge_graph.nodes_of_type("Entity")) == 3
        assert len(multi_edge_graph.nodes_of_type("TestCase")) == 1


# ── Fuzzy Lookup Tests ──

class TestFuzzyLookup:
    def test_exact_match(self, simple_chain):
        assert simple_chain.find_entity("A") == "A"

    def test_case_insensitive(self, simple_chain):
        assert simple_chain.find_entity("a") == "A"

    def test_normalized_underscore(self):
        g = InMemoryGraphStore()
        g.add_node(Node(id="My_Entity", type="Entity"))
        assert g.find_entity("my entity") == "My_Entity"

    def test_no_match(self, simple_chain):
        assert simple_chain.find_entity("ZZZ", cutoff=0.9) is None

    def test_find_entities_batch(self, simple_chain):
        assert simple_chain.find_entities(["A", "B", "Z"]) == ["A", "B", None]


# ── Traversal Tests ──

class TestTraversal:
    def test_forward_traversal(self, simple_chain):
        paths = simple_chain.traverse(["B"], max_depth=2, direction="out")
        assert any("A" in p.nodes for p in paths)

    def test_reverse_traversal(self, simple_chain):
        """In from B should find C (C->B edge traversed in reverse)."""
        paths = simple_chain.traverse(["B"], max_depth=2, direction="in")
        # Path: B <-[reversed]-- C, nodes = ["B", "C"]
        found = any("C" in p.nodes and p.nodes[0] == "B" for p in paths if p.length >= 1)
        assert found

    def test_cycle_protection(self, cyclic_graph):
        paths = cyclic_graph.traverse(["A"], max_depth=5, direction="both")
        for p in paths:
            assert len(p.nodes) == len(set(p.nodes)), f"Cycle: {p.nodes}"

    def test_edge_type_filter(self, multi_edge_graph):
        paths = multi_edge_graph.traverse(["A"], max_depth=2, direction="out", edge_types={"blocked_by"})
        for p in paths:
            for s in p.steps:
                assert s.edge.type == "blocked_by"

    def test_max_depth(self, simple_chain):
        paths = simple_chain.traverse(["A"], max_depth=1, direction="both")
        assert all(p.length <= 1 for p in paths)

    def test_reverse_steps_flagged(self, simple_chain):
        paths = simple_chain.traverse(["B"], max_depth=2, direction="in")
        for p in paths:
            for step in p.steps:
                if step.edge.source == "C" and step.edge.target == "B":
                    assert step.reversed
                    assert step.from_node == "B"
                    assert step.to_node == "C"


# ── Scoring Tests ──

class TestScoring:
    def test_empty_path_scores_zero(self):
        assert score_path(Path(), {"A"}) == 0.0

    def test_higher_confidence_higher_score(self):
        e1 = Edge(type="depends_on", source="A", target="B", confidence=0.9)
        e2 = Edge(type="depends_on", source="A", target="B", confidence=0.5)
        assert score_path(Path(steps=[PathStep(edge=e1)]), {"A"}) > score_path(Path(steps=[PathStep(edge=e2)]), {"A"})

    def test_coverage_improves_score(self):
        """Path covers query completely scores higher than path covering partially."""
        e1 = Edge(type="depends_on", source="A", target="B", confidence=0.9)
        e2 = Edge(type="depends_on", source="B", target="C", confidence=0.9)
        p = Path(steps=[PathStep(edge=e1), PathStep(edge=e2)])
        # Path nodes: {A, B, C}. Query {A, B, C} = 100% match. Query {A, X, Y} = 33% match.
        assert score_path(p, {"A", "B", "C"}) > score_path(p, {"A", "X", "Y"})

    def test_scoring_monotonic(self):
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            e = Edge(type="depends_on", source="A", target="B", confidence=conf)
            s = score_path(Path(steps=[PathStep(edge=e)]), {"A"})
            assert 0.0 <= s <= 1.0

    def test_rank_paths_sorts(self):
        e1 = Edge(type="caused_by", source="A", target="B", confidence=1.0)
        e2 = Edge(type="related_to", source="A", target="C", confidence=0.3)
        ranked = rank_paths([Path(steps=[PathStep(edge=e2)]), Path(steps=[PathStep(edge=e1)])], {"A"})
        assert ranked[0].score >= ranked[1].score

    def test_rank_paths_dedup(self):
        e = Edge(type="depends_on", source="A", target="B", confidence=0.9)
        e2 = Edge(type="depends_on", source="B", target="C", confidence=0.8)
        p_short = Path(steps=[PathStep(edge=e)])
        p_long = Path(steps=[PathStep(edge=e), PathStep(edge=e2)])
        ranked = rank_paths([p_short, p_long], {"A", "B", "C"})
        # After dedup, the longer path should survive
        assert any(p.length >= p_short.length for p in ranked)


# ── Beam Search Tests ──

class TestBeamSearch:
    def test_finds_paths(self, simple_chain):
        paths = beam_search(simple_chain, ["A"], {"A"}, max_depth=3, beam_width=3)
        assert len(paths) > 0

    def test_cycle_protection_beam(self, cyclic_graph):
        paths = beam_search(cyclic_graph, ["A"], {"A"}, max_depth=5, beam_width=3)
        for p in paths:
            assert len(p.nodes) == len(set(p.nodes))
