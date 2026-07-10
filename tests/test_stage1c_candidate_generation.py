from __future__ import annotations

from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from stack.encoder.stage1c import (
    generate_stage1c_pairs,
    stage1c_property_candidates,
)


def _graph() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(Node(
        id="Exp_Oracle",
        type="Experiment",
        aliases=["oracle memory"],
        properties={
            "key_finding": "Oracle memory reaches 99.87 percent accuracy",
            "description": "Validation of external memory retrieval",
        },
    ))
    graph.add_node(Node(
        id="Concept_Retrieval",
        type="Concept",
        aliases=["retrieval mismatch"],
        properties={
            "key_finding": "Projection mismatch prevents hidden-state lookup",
            "description": "A retrieval failure mode",
        },
    ))
    graph.add_edge(Edge(type="related_to", source="Concept_Retrieval", target="Exp_Oracle"))
    return graph


def test_property_candidates_use_key_finding_without_test_data():
    graph = _graph()
    candidates = stage1c_property_candidates(
        "What accuracy did oracle memory reach?", graph, limit=10
    )
    assert candidates[0] == "Exp_Oracle"


def test_property_candidates_are_deterministic_and_do_not_return_unrelated_nodes():
    graph = _graph()
    candidates = stage1c_property_candidates(
        "Why does projection mismatch prevent hidden state lookup?", graph, limit=10
    )
    assert candidates == ["Concept_Retrieval"]


def test_generated_pairs_have_provenance_and_no_frozen_split_access():
    pairs = generate_stage1c_pairs(_graph())
    assert pairs
    assert all(pair["source_id"].startswith("graph:") for pair in pairs)
    assert all(pair["label_source"] in {"exact_alias", "key_finding", "weak"} for pair in pairs)
    assert all("test.jsonl" not in pair["source_id"] for pair in pairs)
