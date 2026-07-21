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
