"""Stage 4 bounded rule engine tests."""

from __future__ import annotations

from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.rules import Rule, RuleEngine


def _toy_graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="A", type="Entity", sources=["a.md"]))
    graph.add_node(Node(id="B", type="Entity", sources=["a.md"]))
    graph.add_node(Node(id="C", type="Entity", sources=["a.md"]))
    graph.add_edge(Edge(type="depends_on", source="A", target="B", confidence=1.0, evidence="a.md"))
    graph.add_edge(Edge(type="depends_on", source="B", target="C", confidence=1.0, evidence="a.md"))
    return graph


def test_transitive_depends_on_rule_records_premises_and_rule_id():
    graph = _toy_graph()
    rule = Rule(
        rule_id="trans_depends_on",
        version="1",
        body=(("x", "depends_on", "y"), ("y", "depends_on", "z")),
        head=("x", "depends_on", "z"),
    )
    result = RuleEngine([rule], max_depth=2, max_activations=32).evaluate(graph)
    inferred = {(f.source, f.relation, f.target) for f in result.inferred}
    assert ("A", "depends_on", "C") in inferred
    fact = next(f for f in result.inferred if f.source == "A" and f.target == "C")
    assert fact.rule_id == "trans_depends_on"
    assert fact.rule_version == "1"
    assert ("A", "depends_on", "B") in fact.premises
    assert ("B", "depends_on", "C") in fact.premises
    assert result.truncated is False


def test_rule_engine_respects_activation_bound():
    graph = _toy_graph()
    rule = Rule(
        rule_id="trans_depends_on",
        version="1",
        body=(("x", "depends_on", "y"), ("y", "depends_on", "z")),
        head=("x", "depends_on", "z"),
    )
    result = RuleEngine([rule], max_depth=2, max_activations=1).evaluate(graph)
    assert result.truncated is True
    assert result.truncation_reason == "max_activations"
