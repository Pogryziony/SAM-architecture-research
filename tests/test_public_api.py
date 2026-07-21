"""Public API / CLI surface tests."""

from __future__ import annotations

from nexus.api import ask, main
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig


def _tiny_graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Warsaw", type="Entity", properties={"key_finding": "Capital of Poland"}))
    graph.add_node(Node(id="Poland", type="Entity"))
    graph.add_edge(
        Edge(type="related_to", source="Warsaw", target="Poland", confidence=1.0)
    )
    return graph


def test_ask_uses_grounded_profile_by_default():
    result = ask("What is Warsaw?", graph=_tiny_graph(), question_id="t1")
    assert result.question_id == "t1"
    assert isinstance(result.answer, str)


def test_ask_accepts_pointer_copy_profile():
    config = ProductionNEXUSConfig.pointer_copy()
    result = ask("What is Warsaw?", graph=_tiny_graph(), config=config)
    assert result.question


def test_cli_profiles_command(capsys):
    code = main(["profiles"])
    assert code == 0
    out = capsys.readouterr().out
    assert "library_default" in out
    assert "synth" in out


def test_cli_ask_help_lists_profiles():
    try:
        main(["ask", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
