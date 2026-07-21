"""UnionResolver handoff: lexical ∪ ER3 with question-gated entry pruning."""

from __future__ import annotations

from nexus.graph import Node
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.entity_resolver import ResolutionCandidate, ResolutionResult
from stack.pipeline.resolver import UnionResolver, mention_score


class _StubResolver:
    def __init__(self, selected: list[str], pool: list[str] | None = None, scores: dict[str, float] | None = None):
        self.selected = list(selected)
        self.pool = list(pool if pool is not None else selected)
        self.scores = scores or {}

    def resolve(self, question: str, graph: InMemoryGraphStore) -> ResolutionResult:
        candidates = [
            ResolutionCandidate(
                entity_id=entity_id,
                score=self.scores.get(entity_id, 1.0 / (index + 1)),
            )
            for index, entity_id in enumerate(self.pool)
        ]
        return ResolutionResult(
            selected_entity_ids=list(self.selected),
            candidates=candidates,
            candidate_pool_size=len(candidates),
            resolver_name="stub",
            resolver_version="1",
        )


def _toy_graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(Node(
        id="Concept_SelectorBottleneck",
        type="Concept",
        properties={"description": "Selector precision bottleneck"},
        aliases=["selector bottleneck"],
    ))
    graph.add_node(Node(
        id="Exp_0_12_Selection",
        type="Experiment",
        properties={"title": "Selection bottleneck"},
        aliases=["selection experiment"],
    ))
    for index in range(10):
        graph.add_node(Node(
            id=f"Exp_Hub_{index}",
            type="Experiment",
            properties={"title": f"Hub experiment {index}"},
        ))
    return graph


def test_mention_score_prefers_id_and_alias_hits():
    graph = _toy_graph()
    question = "What validates Concept_SelectorBottleneck?"
    assert mention_score("Concept_SelectorBottleneck", question, graph) > 0
    assert mention_score("Exp_Hub_0", question, graph) == 0.0
    alias_q = "Explain the selector bottleneck"
    assert mention_score("Concept_SelectorBottleneck", alias_q, graph) > 0


def test_union_prefers_mentioned_entities_over_static_hub_topk():
    graph = _toy_graph()
    hubs = [f"Exp_Hub_{index}" for index in range(10)]
    er3 = _StubResolver(
        selected=hubs[:10],
        pool=[*hubs, "Concept_SelectorBottleneck", "Exp_0_12_Selection"],
        scores={**{hub: 0.99 for hub in hubs}, "Concept_SelectorBottleneck": 0.2, "Exp_0_12_Selection": 0.1},
    )
    lexical = _StubResolver(selected=["Concept_SelectorBottleneck"])
    resolver = UnionResolver(er3, lexical, top_k=10)
    result = resolver.resolve(
        "What validates Concept_SelectorBottleneck?",
        graph,
    )
    assert result.resolver_name == "union_lexical_er3"
    assert "Concept_SelectorBottleneck" in result.selected_entity_ids
    assert result.selected_entity_ids[0] == "Concept_SelectorBottleneck"
    # Full ER3 pool retained for pool_recall diagnostics.
    assert result.candidate_pool_size >= 12
    assert "Concept_SelectorBottleneck" in {item.entity_id for item in result.candidates}


def test_union_ignores_noncanonical_lexical_noise():
    graph = _toy_graph()
    hubs = [f"Exp_Hub_{index}" for index in range(8)]
    er3 = _StubResolver(selected=hubs[:5], pool=[*hubs, "Concept_SelectorBottleneck"])
    lexical = _StubResolver(
        selected=["Document_Chunks_+_Embeddings", "Explicitly", "Concept_SelectorBottleneck"]
    )
    resolver = UnionResolver(er3, lexical, top_k=5)
    result = resolver.resolve(
        "What validates Concept_SelectorBottleneck?",
        graph,
    )
    assert "Document_Chunks_+_Embeddings" not in result.selected_entity_ids
    assert "Explicitly" not in result.selected_entity_ids
    assert "Concept_SelectorBottleneck" in result.selected_entity_ids


def test_union_falls_back_to_er3_when_no_mentions():
    graph = _toy_graph()
    hubs = [f"Exp_Hub_{index}" for index in range(8)]
    er3 = _StubResolver(selected=hubs[:5], pool=hubs)
    lexical = _StubResolver(selected=[])
    resolver = UnionResolver(er3, lexical, top_k=5)
    result = resolver.resolve("What is the architecture direction?", graph)
    # Ungrounded handoff keeps ER3 quality anchors then diversifies the rest.
    assert result.selected_entity_ids[:5] == hubs[:5]
    assert len(result.selected_entity_ids) == 5
    assert set(result.selected_entity_ids).issubset(set(hubs))
