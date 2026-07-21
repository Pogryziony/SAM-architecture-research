"""Stage 7 deterministic render wired into answer_question."""

from __future__ import annotations

from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig, validate_config
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import DummyModel


class _NeverGenerate(DummyModel):
    def generate(self, prompt: str, **kwargs):  # type: ignore[override]
        raise AssertionError("LLM/synth must not be called for grounded path render")


def _tiny_graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Alpha", type="Entity", aliases=["Alpha"]))
    graph.add_node(Node(id="Beta", type="Entity", aliases=["Beta"]))
    graph.add_node(Node(id="Gamma", type="Entity", aliases=["Gamma"]))
    graph.add_edge(
        Edge(
            type="depends_on",
            source="Alpha",
            target="Beta",
            confidence=1.0,
            evidence="test",
        )
    )
    graph.add_edge(
        Edge(
            type="depends_on",
            source="Beta",
            target="Gamma",
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
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Alpha", "Beta"],
    )
    assert result["realization"] is not None
    assert result["realization"]["strategy"] == "deterministic_render"
    assert "Alpha depends_on Beta" in result["answer"]
    assert result["realization"].get("coverage_errors") == []


def test_grounded_v1_uses_path_render_for_causal_intent():
    config = ProductionNEXUSConfig.grounded(max_entry_nodes=6, max_paths=6)
    result = answer_question(
        "Why does Alpha depend on Gamma?",
        _tiny_graph(),
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Alpha", "Beta", "Gamma"],
    )
    assert result["realization"] is not None
    assert result["realization"]["strategy"] == "deterministic_render"
    assert "depends_on" in result["answer"]
    assert "Alpha" in result["answer"] and "Beta" in result["answer"]


def test_l1_acceptance_backend_is_allowed():
    config = ProductionNEXUSConfig.l1_acceptance()
    assert validate_config(config) == []
    assert config.realizer_backend == "l1_acceptance"


def test_l1_acceptance_uses_path_render_for_relation_question():
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    result = answer_question(
        "Does Alpha have the depends_on relation to Beta?",
        _tiny_graph(),
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Alpha", "Beta"],
    )
    assert result["realization"]["strategy"] == "deterministic_render"
    assert "Alpha depends_on Beta" in result["answer"]


def test_l1_acceptance_copies_node_fact_for_factual_lookup():
    graph = _tiny_graph()
    from nexus.graph import Node

    graph.add_node(
        Node(
            id="Concept_Works",
            type="Concept",
            aliases=["architecture validated"],
            properties={
                "description": "Oracle memory = 99.87-100%, oracle filter = 100%",
            },
        )
    )
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    result = answer_question(
        "What is the concept that architecture is validated?",
        graph,
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Concept_Works"],
    )
    assert result["realization"]["strategy"] == "l1_node_fact"
    assert "99.87" in result["answer"]
    assert not result["answer"].startswith("Concept_Works:")


def test_l1_acceptance_edge_catalog_from_weights():
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    result = answer_question(
        "What are the edge type weights for traversal and why?",
        _tiny_graph(),
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Alpha"],
    )
    assert result["realization"]["strategy"] == "edge_catalog"
    assert "caused_by=1.00" in result["answer"]
    assert "mentioned_in=0.20" in result["answer"]
