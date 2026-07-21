"""Stage 7 deterministic render wired into answer_question."""

from __future__ import annotations

from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig, validate_config
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import DummyModel


def _tiny_graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Alpha", type="Entity", aliases=["Alpha"]))
    graph.add_node(Node(id="Beta", type="Entity", aliases=["Beta"]))
    graph.add_edge(
        Edge(
            type="depends_on",
            source="Alpha",
            target="Beta",
            confidence=1.0,
            evidence="test",
        )
    )
    return graph


def test_deterministic_render_backend_is_allowed():
    config = ProductionNEXUSConfig.deterministic_render()
    assert validate_config(config) == []
    assert config.realizer_backend == "deterministic_render"


def test_answer_question_uses_deterministic_render_without_llm():
    config = ProductionNEXUSConfig.deterministic_render(max_entry_nodes=4)
    result = answer_question(
        "Does Alpha have the depends_on relation to Beta?",
        _tiny_graph(),
        model=DummyModel(),
        config=config,
        entry_nodes_override=["Alpha", "Beta"],
    )
    assert result["realization"] is not None
    assert result["realization"]["strategy"] == "deterministic_render"
    assert "Alpha depends_on Beta" in result["answer"]
    assert result["realization"].get("coverage_errors") == []
