"""Exact node-ID mentions must survive entry ranking for family/oracle questions."""

from __future__ import annotations

from nexus.graph.family_curations import apply_oracle_family_curations
from nexus.graph import Node
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.query.parser import parse_question, spot_entities


def _family_graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Decision_PivotToNEXUS", type="Decision"))
    # Hub noise that previously crowded out exact ID mentions via fuzzy n-grams.
    for nid in (
        "Concept_OracleMemory",
        "Step_3:_Relation_Extraction",
        "Rag_Vs_Graph_Sam_(nexus)_Detailed_Comparison",
    ):
        graph.add_node(
            Node(
                id=nid,
                type="Concept",
                properties={"description": "valid depend on memory relation extraction"},
            )
        )
    apply_oracle_family_curations(graph)
    return graph


def test_exact_id_mentions_rank_into_entry_set():
    graph = _family_graph()
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=5)
    q = (
        "As valid in 2019, did Concept_LegacyFlatMemory depend on "
        "Module_LegacySelector?"
    )
    parsed = parse_question(q, graph, config=config)
    assert "Concept_LegacyFlatMemory" in parsed.entity_ids
    assert "Module_LegacySelector" in parsed.entity_ids


def test_grounded_spotting_rejects_ungrounded_hub_fuzzy():
    graph = _family_graph()
    config = ProductionNEXUSConfig.l1_acceptance()
    q = (
        "As valid in 2019, did Concept_LegacyFlatMemory depend on "
        "Module_LegacySelector?"
    )
    spots, _wb = spot_entities(q, graph, cutoff=config.fuzzy_cutoff, config=config)
    spotted = {nid for *_, nid in spots}
    assert "Concept_LegacyFlatMemory" in spotted
    assert "Concept_OracleMemory" not in spotted


def test_temporal_active_replace_mentions_decision_and_legacy():
    graph = _family_graph()
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=5)
    q = (
        "As known after the NEXUS pivot, according to the graph, what does "
        "Decision_PivotToNEXUS replace?"
    )
    parsed = parse_question(q, graph, config=config)
    assert "Decision_PivotToNEXUS" in parsed.entity_ids
