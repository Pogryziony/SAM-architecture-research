from __future__ import annotations

import ast
from pathlib import Path

from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from stack.encoder.trivial_baseline import rank_candidates


MODULE_PATH = Path(__file__).parents[1] / "stack" / "encoder" / "trivial_baseline.py"


def _graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    for node_id in ("low_degree", "high_degree", "lexical_tie"):
        graph.add_node(Node(id=node_id, type="Entity"))
    graph.add_edge(Edge(type="related_to", source="high_degree", target="low_degree"))
    graph.add_edge(Edge(type="related_to", source="high_degree", target="lexical_tie"))
    return graph


def test_baseline_has_no_torch_or_weight_loading_and_ranks_deterministically():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if node.module
    )
    assert "torch" not in imports
    assert "model" not in source.lower()
    assert "weight" not in source.lower()

    candidates = [
        {"node_id": "low_degree", "lexical_score": 4},
        {"node_id": "high_degree", "lexical_score": 4},
        {"node_id": "lexical_tie", "lexical_score": 4},
    ]
    first = rank_candidates(candidates, _graph(), top_k=3)
    second = rank_candidates(list(reversed(candidates)), _graph(), top_k=3)
    assert first == second == ["high_degree", "lexical_tie", "low_degree"]
    assert rank_candidates(candidates, _graph(), top_k=2) == ["high_degree", "lexical_tie"]
