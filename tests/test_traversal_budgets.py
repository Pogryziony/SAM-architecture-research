"""Traversal expansion budgets and truncation reporting."""

from __future__ import annotations

import nexus.graph.traversal as traversal_mod
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.graph.traversal import TraversalStats, beam_search
from nexus.utils.config import NEXUSConfig


def _chain_graph(n: int = 20) -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    for index in range(n):
        graph.add_node(Node(id=f"n{index}", type="Entity"))
    for index in range(n - 1):
        graph.add_edge(
            Edge(
                type="depends_on",
                source=f"n{index}",
                target=f"n{index + 1}",
                confidence=1.0,
            )
        )
    return graph


def test_beam_search_respects_edge_budget():
    graph = _chain_graph(30)
    stats = TraversalStats()
    config = NEXUSConfig(max_depth=10, beam_width=5, max_expanded_edges=3, max_expanded_nodes=10_000)
    paths = beam_search(
        graph,
        start_nodes=["n0"],
        query_entities={"n0"},
        direction="out",
        config=config,
        stats=stats,
    )
    assert stats.truncated is True
    assert stats.truncation_reason == "max_expanded_edges"
    assert stats.expanded_edges <= 3
    assert isinstance(paths, list)
    assert stats.elapsed_ms >= 0.0


def test_beam_search_respects_node_budget():
    graph = _chain_graph(30)
    stats = TraversalStats()
    config = NEXUSConfig(max_depth=10, beam_width=5, max_expanded_edges=10_000, max_expanded_nodes=4)
    beam_search(
        graph,
        start_nodes=["n0"],
        query_entities={"n0"},
        direction="out",
        config=config,
        stats=stats,
    )
    assert stats.truncated is True
    assert stats.truncation_reason == "max_expanded_nodes"
    assert stats.expanded_nodes >= 4


def test_beam_search_respects_wall_clock_budget(monkeypatch):
    graph = _chain_graph(30)
    stats = TraversalStats()
    ticks = {"n": 0}

    def fake_perf_counter() -> float:
        ticks["n"] += 1
        # First call is start; later calls report that the budget is exhausted.
        return 0.0 if ticks["n"] == 1 else 0.05

    monkeypatch.setattr(traversal_mod.time, "perf_counter", fake_perf_counter)
    beam_search(
        graph,
        start_nodes=["n0"],
        query_entities={"n0"},
        direction="out",
        config=NEXUSConfig(
            max_depth=10,
            beam_width=5,
            max_expanded_edges=10_000,
            max_expanded_nodes=10_000,
            max_traversal_ms=10.0,
        ),
        stats=stats,
    )
    assert stats.truncated is True
    assert stats.truncation_reason == "max_traversal_ms"
    assert stats.elapsed_ms >= 10.0


def test_default_budgets_do_not_truncate_small_graphs():
    graph = _chain_graph(5)
    stats = TraversalStats()
    paths = beam_search(
        graph,
        start_nodes=["n0"],
        query_entities={"n0", "n4"},
        direction="out",
        config=NEXUSConfig(),
        stats=stats,
    )
    assert stats.truncated is False
    assert stats.truncation_reason == ""
    assert paths


def test_beam_search_applies_bitemporal_cutoffs():
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="A", type="Entity"))
    graph.add_node(Node(id="B", type="Entity"))
    graph.add_node(Node(id="C", type="Entity"))
    graph.add_edge(
        Edge(
            type="depends_on",
            source="A",
            target="B",
            confidence=1.0,
            valid_from="2020-01-01T00:00:00+00:00",
            observed_at="2020-01-01T00:00:00+00:00",
        )
    )
    graph.add_edge(
        Edge(
            type="depends_on",
            source="A",
            target="C",
            confidence=1.0,
            valid_from="2024-01-01T00:00:00+00:00",
            observed_at="2024-01-01T00:00:00+00:00",
        )
    )
    paths = beam_search(
        graph,
        start_nodes=["A"],
        query_entities={"A", "B", "C"},
        direction="out",
        config=NEXUSConfig(
            max_depth=2,
            beam_width=5,
            as_valid_at="2021-06-01T00:00:00+00:00",
            as_known_at="2021-06-01T00:00:00+00:00",
        ),
    )
    targets = {step.to_node for path in paths for step in path.steps}
    assert "B" in targets
    assert "C" not in targets
