"""Path score focus: hub fillers must not bury gold edges in the proof window."""

from __future__ import annotations

from nexus.graph import Edge, Node, Path, PathStep
from nexus.graph.scoring import (
    focus_query_entities,
    incident_paths_for_entities,
    rank_paths,
    score_path,
    select_proof_paths,
)
from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import DummyModel
from nexus.utils.config import NEXUSConfig
from stack.pipeline.resolver import UnionResolver


class _StubResolver:
    def __init__(self, selected: list[str], pool: list[str], scores: dict[str, float] | None = None):
        self.selected = list(selected)
        self.pool = list(pool)
        self.scores = scores or {}

    def resolve(self, question: str, graph: InMemoryGraphStore):
        from nexus.pipeline.entity_resolver import ResolutionCandidate, ResolutionResult

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


def test_focus_query_entities_uses_leading_ids_only():
    assert focus_query_entities(["A", "B", "C"], 2) == {"A", "B"}
    assert focus_query_entities(["A", "B", "C"], 0) == {"A", "B", "C"}


def test_focus_prefers_mentioned_entity_even_when_not_first():
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Exp_0_WrongHub", type="Experiment", properties={"title": "Hub"}))
    graph.add_node(Node(
        id="Exp_0_6_Validation",
        type="Experiment",
        properties={"title": "Full validation"},
        aliases=["validation experiment"],
    ))
    entries = ["Exp_0_WrongHub", "Exp_1_Other", "Exp_0_6_Validation"]
    focus = focus_query_entities(
        entries,
        2,
        question="What did Exp_0_6_Validation show?",
        graph=graph,
    )
    assert "Exp_0_6_Validation" in focus


def test_focused_scoring_prefers_gold_edge_over_hub_chain():
    hubs = [f"Exp_{index}_Hub" for index in range(10)]
    gold_src = "Exp_0_12_Selection"
    gold_tgt = "Concept_SelectorBottleneck"
    entries = [gold_src, gold_tgt, *hubs]

    hub_edge = Edge(type="depends_on", source=hubs[0], target=hubs[1], confidence=0.95)
    gold_edge = Edge(type="validates", source=gold_src, target=gold_tgt, confidence=0.95)
    hub_path = Path(steps=[PathStep(edge=hub_edge)])
    gold_path = Path(steps=[PathStep(edge=gold_edge)])

    focused = rank_paths([hub_path, gold_path], focus_query_entities(entries, 2))
    assert score_path(hub_path, set(entries)) >= score_path(gold_path, set(entries))
    assert focused[0].nodes == gold_path.nodes


def test_select_proof_paths_fills_missing_focus_coverage():
    focus = {"GoldA", "GoldB"}
    hub = Path(steps=[PathStep(edge=Edge(
        type="depends_on", source="Hub1", target="Hub2", confidence=0.99
    ))])
    hub.score = 0.9
    gold = Path(steps=[PathStep(edge=Edge(
        type="validates", source="GoldA", target="GoldB", confidence=0.8
    ))])
    gold.score = 0.1
    selected = select_proof_paths([hub, gold], focus, max_paths=2)
    assert any("GoldA" in path.nodes for path in selected)


def test_answer_question_keeps_gold_path_inside_max_paths_with_hub_entries():
    graph = InMemoryGraphStore()
    gold_src = "Exp_0_12_Selection"
    gold_tgt = "Concept_SelectorBottleneck"
    hubs = [f"Exp_{index}_HubPack" for index in range(8)]
    for entity_id, node_type, props in [
        (gold_src, "Experiment", {"title": "Selection bottleneck"}),
        (gold_tgt, "Concept", {"description": "Selector bottleneck"}),
        *[(hub, "Experiment", {"title": hub}) for hub in hubs],
    ]:
        graph.add_node(Node(id=entity_id, type=node_type, properties=props, sources=["docs/a.md"]))
    for index in range(len(hubs) - 1):
        graph.add_edge(Edge(
            type="depends_on",
            source=hubs[index],
            target=hubs[index + 1],
            confidence=0.99,
            evidence="docs/hub.md",
        ))
    graph.add_edge(Edge(
        type="validates",
        source=gold_src,
        target=gold_tgt,
        confidence=0.95,
        evidence="docs/gold.md",
    ))

    # Gold buried after hub prefixes — mention-aware focus must still recover it.
    entries = [*hubs[:4], gold_src, gold_tgt, *hubs[4:]]
    config = NEXUSConfig(
        max_entry_nodes=12,
        path_score_focus=2,
        max_paths=12,
        max_depth=2,
        beam_width=20,
        require_structured_provenance=False,
    )
    result = answer_question(
        "What validates Concept_SelectorBottleneck?",
        graph,
        model=DummyModel(),
        config=config,
        entry_nodes_override=entries,
    )
    proof_edges = {
        frozenset((step["from_node"], step["to_node"]))
        for step in result["reasoning_audit"].get("proof_steps", [])
        if step["relation"] == "validates"
    }
    assert frozenset((gold_src, gold_tgt)) in proof_edges


def test_zero_hop_audit_uses_incident_edges():
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Exp_0_1_Solo", type="Experiment", properties={"title": "Solo"}, sources=["a.md"]))
    graph.add_node(Node(id="Concept_Solo", type="Concept", properties={"description": "Solo concept"}, sources=["a.md"]))
    graph.add_edge(Edge(
        type="validates",
        source="Exp_0_1_Solo",
        target="Concept_Solo",
        confidence=0.9,
        evidence="a.md",
    ))
    # No multi-hop chain from an isolated second entry — force zero-hop style audit helpers.
    paths = incident_paths_for_entities(graph, ["Exp_0_1_Solo"], max_paths=4)
    assert paths
    assert paths[0].steps[0].edge.type == "validates"


def test_union_limits_ungrounded_fillers_when_grounded_exists():
    graph = InMemoryGraphStore()
    graph.add_node(Node(
        id="Concept_SelectorBottleneck",
        type="Concept",
        properties={"description": "Selector bottleneck"},
        aliases=["selector bottleneck"],
    ))
    hubs = [f"Exp_{index}_Pad" for index in range(10)]
    for hub in hubs:
        graph.add_node(Node(id=hub, type="Experiment", properties={"title": hub}))
    er3 = _StubResolver(
        selected=hubs[:10],
        pool=[*hubs, "Concept_SelectorBottleneck"],
        scores={**{hub: 0.99 for hub in hubs}, "Concept_SelectorBottleneck": 0.1},
    )
    lexical = _StubResolver(selected=["Concept_SelectorBottleneck"], pool=["Concept_SelectorBottleneck"])
    resolver = UnionResolver(er3, lexical, top_k=12, max_ungrounded_fillers=4)
    result = resolver.resolve("What validates Concept_SelectorBottleneck?", graph)
    assert "Concept_SelectorBottleneck" in result.selected_entity_ids
    assert len(result.selected_entity_ids) <= 5  # 1 grounded + 4 fillers


def test_union_diversifies_ungrounded_packs_across_questions():
    graph = InMemoryGraphStore()
    hubs = [f"Exp_{index}_SharedHub" for index in range(12)]
    for hub in hubs:
        graph.add_node(Node(
            id=hub,
            type="Experiment",
            properties={"title": hub.replace("_", " "), "description": hub},
        ))
    er3 = _StubResolver(
        selected=hubs,
        pool=hubs,
        scores={hub: 0.95 for hub in hubs},
    )
    lexical = _StubResolver(selected=[], pool=[])
    resolver = UnionResolver(er3, lexical, top_k=12)
    a = resolver.resolve("How does aggregation affect noisy memory tolerance?", graph)
    b = resolver.resolve("Why did chain retrieval require the validation set?", graph)
    assert a.selected_entity_ids != b.selected_entity_ids
