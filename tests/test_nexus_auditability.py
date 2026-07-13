"""Tests for deterministic NEXUS proof, provenance, and oracle evaluation."""

from __future__ import annotations

from nexus.graph import Edge, Node, Path, PathStep
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig, validate_config
from nexus.pipeline.runner import NEXUSRunner
from nexus.reasoning.audit import build_reasoning_audit
from nexus.reasoning.evidence_builder import build_evidence_pack
from nexus.reasoning.model_interface import DummyModel
from nexus.reasoning.verifier import VerificationResult


def _verified() -> VerificationResult:
    return VerificationResult(
        supported_count=1,
        unsupported_claims=[],
        hallucination_rate=0.0,
        passed=True,
    )


def _auditable_graph(*, with_counter: bool = False) -> tuple[InMemoryGraphStore, Path]:
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="A", type="Entity", sources=["facts/a.md"]))
    graph.add_node(Node(id="B", type="Entity", sources=["facts/b.md"]))
    support = Edge(
        type="depends_on",
        source="A",
        target="B",
        confidence=0.9,
        evidence="decisions/dependency.md",
    )
    graph.add_edge(support)
    if with_counter:
        graph.add_node(Node(id="C", type="Entity", sources=["reviews/c.md"]))
        graph.add_edge(Edge(
            type="contradicts",
            source="C",
            target="A",
            confidence=0.8,
            evidence="reviews/challenge.md",
        ))
    path = Path(steps=[PathStep(edge=support)], score=0.85)
    return graph, path


def test_audit_builds_stable_replayable_proof_with_provenance():
    graph, path = _auditable_graph()
    pack = build_evidence_pack("What does A depend on?", [path], graph)

    first = build_reasoning_audit(
        [path], graph, pack, _verified(), "A depends on B."
    ).to_dict()
    second = build_reasoning_audit(
        [path], graph, pack, _verified(), "A depends on B."
    ).to_dict()

    assert first == second
    assert first["proof_valid"] is True
    assert first["recommended_action"] == "answer"
    assert first["provenance_coverage"] == 1.0
    assert len(first["proof_steps"]) == 1
    assert first["proof_steps"][0]["sources"] == [
        "decisions/dependency.md", "facts/a.md", "facts/b.md",
    ]


def test_counter_evidence_is_exposed_and_forces_conditional_answer():
    graph, path = _auditable_graph(with_counter=True)
    pack = build_evidence_pack("What does A depend on?", [path], graph)
    audit = build_reasoning_audit(
        [path], graph, pack, _verified(), "A depends on B."
    )

    assert len(audit.counter_evidence) == 1
    assert audit.counter_evidence[0].source == "C"
    assert audit.counter_evidence[0].target == "A"
    assert audit.recommended_action == "conditional_answer"


def test_missing_provenance_cannot_produce_unconditional_answer():
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="A", type="Entity"))
    graph.add_node(Node(id="B", type="Entity"))
    edge = Edge(type="depends_on", source="A", target="B", confidence=1.0)
    graph.add_edge(edge)
    path = Path(steps=[PathStep(edge=edge)], score=1.0)
    pack = build_evidence_pack("What does A depend on?", [path], graph)

    audit = build_reasoning_audit(
        [path], graph, pack, _verified(), "A depends on B."
    )
    assert audit.provenance_coverage == 0.0
    assert audit.recommended_action == "conditional_answer"


def test_path_edge_missing_from_store_invalidates_proof():
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="A", type="Entity"))
    graph.add_node(Node(id="B", type="Entity"))
    edge = Edge(type="depends_on", source="A", target="B", confidence=1.0)
    path = Path(steps=[PathStep(edge=edge)], score=1.0)

    audit = build_reasoning_audit(
        [path], graph, {}, _verified(), "A depends on B."
    )
    assert audit.proof_valid is False
    assert audit.recommended_action == "abstain"
    assert any("absent from graph" in error for error in audit.errors)


def test_insufficient_answer_is_always_an_abstention():
    graph, path = _auditable_graph()
    pack = build_evidence_pack("What does A depend on?", [path], graph)
    audit = build_reasoning_audit(
        [path], graph, pack, _verified(), "Insufficient evidence to answer."
    )
    assert audit.recommended_action == "abstain"


def test_readiness_thresholds_are_hashed_and_validated():
    base = ProductionNEXUSConfig.lexical_only()
    changed = ProductionNEXUSConfig.lexical_only(
        readiness_answer_threshold=0.80
    )
    invalid = ProductionNEXUSConfig.lexical_only(
        readiness_answer_threshold=0.30,
        readiness_conditional_threshold=0.60,
    )

    assert base.config_hash != changed.config_hash
    assert changed.to_dict()["nexus_config"]["readiness_answer_threshold"] == 0.80
    assert any("must be <=" in error for error in validate_config(invalid))


def test_oracle_mode_uses_gold_entities_and_serializes_audit_fields():
    graph = InMemoryGraphStore()
    graph.add_node(Node(
        id="Exp_Alpha", type="Experiment", aliases=["alpha"],
        properties={"key_finding": "Alpha result"}, sources=["alpha.md"],
    ))
    graph.add_node(Node(
        id="Exp_Beta", type="Experiment", aliases=["beta"],
        properties={"key_finding": "Beta result"}, sources=["beta.md"],
    ))
    graph.add_edge(Edge(
        type="derived_from", source="Exp_Beta", target="Exp_Alpha",
        confidence=0.9, evidence="comparison.md",
    ))
    runner = NEXUSRunner(
        graph, ProductionNEXUSConfig.lexical_only(), model=DummyModel()
    )

    result = runner.run_oracle([{
        "id": "q1",
        "question": "What about alpha?",
        "gold_entities": ["Exp_Beta"],
    }], source_sha="abc123")
    serialized = runner.serialize_result(result)
    question = result.per_question[0]

    assert result.evaluation_mode == "oracle"
    assert question.predicted_entities == ["Exp_Beta"]
    assert question.entity_resolution_method == "oracle"
    assert question.lexical_fallback_used is False
    assert question.path_scores
    assert question.proof_steps_count > 0
    assert serialized["evaluation_mode"] == "oracle"
    assert "reasoning_readiness_score" in serialized["per_question"][0]
    assert "proof_valid" in serialized["per_question"][0]


def test_oracle_mode_fails_closed_when_gold_entities_are_missing():
    runner = NEXUSRunner(
        InMemoryGraphStore(), ProductionNEXUSConfig.lexical_only(),
        model=DummyModel(),
    )
    result = runner.run_oracle([{"id": "q1", "question": "Question"}])

    assert result.evaluation_mode == "oracle"
    assert result.questions_total == 0
    assert any("gold_entities" in error for error in result.errors)
