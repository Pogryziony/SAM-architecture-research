"""L1 prose evidence selection and temporal PIT family end-to-end."""

from __future__ import annotations

from nexus.graph import Edge, Node
from nexus.graph.family_curations import apply_oracle_family_curations
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import DummyModel
from benchmarks.record_answer_plan_status import _is_binding
from benchmarks.scoring import compute_fact_score


class _NeverGenerate(DummyModel):
    def generate(self, prompt: str, **kwargs):  # type: ignore[override]
        raise AssertionError("LLM must not run on L1 acceptance path")


def _graph_with_findings() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(
        Node(
            id="Concept_ArchitectureWorks",
            type="Concept",
            aliases=["architecture validated"],
            properties={
                "description": (
                    "Oracle memory = 99.87-100%, oracle filter = 100% "
                    "— the core+memory architecture IS valid"
                ),
            },
        )
    )
    graph.add_node(
        Node(
            id="Exp_0_6_Validation",
            type="Experiment",
            properties={
                "key_finding": (
                    "Oracle memory: 99.87%. Retrieved memory = core_only (68.74%)."
                ),
            },
        )
    )
    graph.add_node(Node(id="Exp_0_Diagnosis", type="Experiment"))
    graph.add_node(Node(id="Exp_0_2_CompactPKM", type="Experiment"))
    graph.add_node(Node(id="Exp_0_3_PKM_Candidates", type="Experiment"))
    graph.add_edge(
        Edge(
            type="validates",
            source="Exp_0_6_Validation",
            target="Concept_ArchitectureWorks",
            confidence=1.0,
            observed_at="2026-07-08T00:00:00+00:00",
            valid_from="2026-07-08T00:00:00+00:00",
        )
    )
    graph.add_edge(
        Edge(
            type="depends_on",
            source="Exp_0_2_CompactPKM",
            target="Exp_0_Diagnosis",
            confidence=1.0,
            observed_at="2026-07-08T00:00:00+00:00",
            valid_from="2026-07-08T00:00:00+00:00",
        )
    )
    graph.add_edge(
        Edge(
            type="depends_on",
            source="Exp_0_3_PKM_Candidates",
            target="Exp_0_2_CompactPKM",
            confidence=1.0,
            observed_at="2026-07-08T00:00:00+00:00",
            valid_from="2026-07-08T00:00:00+00:00",
        )
    )
    return graph


def test_l1_causal_why_uses_node_fact_not_path_dump():
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    result = answer_question(
        "Why is it important that architecture is validated?",
        _graph_with_findings(),
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Concept_ArchitectureWorks"],
    )
    assert result["realization"]["strategy"] == "l1_node_fact"
    assert "99.87" in result["answer"]
    assert "depends_on" not in result["answer"]


def test_l1_dependency_chain_includes_experiment_count():
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=6)
    result = answer_question(
        "Walk through the entire SAM experiment dependency chain from start to end.",
        _graph_with_findings(),
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Exp_0_Diagnosis", "Exp_0_3_PKM_Candidates"],
    )
    assert result["realization"]["strategy"] == "l1_dependency_chain"
    assert "Exp_0_Diagnosis" in result["answer"]
    assert "→" in result["answer"]
    assert "3 experiments" in result["answer"]


def test_l1_compare_metrics_from_enriched_evidence():
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    result = answer_question(
        "Compare SAM core_only vs SAM oracle_memory performance.",
        _graph_with_findings(),
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Concept_ArchitectureWorks"],
    )
    assert result["realization"]["strategy"] == "l1_compare_metrics"
    assert "68.74" in result["answer"]
    assert "99.87" in result["answer"]
    gold = (
        "core_only: 68.74% overall, 22% on 3-hop. "
        "oracle_memory: 99.87% overall, 100% on 3-hop."
    )
    assert compute_fact_score(result["answer"], gold)["fuzzy_accuracy"] >= 0.5


def test_temporal_family_pit_cutoffs_abstain_end_to_end():
    graph = _graph_with_findings()
    graph.add_node(Node(id="Decision_PivotToNEXUS", type="Decision"))
    graph.add_edge(
        Edge(
            type="implements",
            source="Decision_PivotToNEXUS",
            target="Concept_ArchitectureWorks",
            confidence=1.0,
            observed_at="2026-07-08T00:00:00+00:00",
            valid_from="2026-07-08T00:00:00+00:00",
        )
    )
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    runner = NEXUSRunner(graph, config, model=_NeverGenerate())
    record = {
        "id": "family_temporal_001",
        "question": "As known before any pivot decision existed, what architecture replaced NEXUS?",
        "gold_entities": ["Decision_PivotToNEXUS"],
        "as_known_at": "2020-01-01T00:00:00+00:00",
        "as_valid_at": "",
    }
    pipeline = runner.run_oracle([record], source_sha="test")
    assert not pipeline.errors
    qr = pipeline.per_question[0]
    assert "No valid temporal facts" in qr.answer
    assert qr.reasoning_action in {"abstain", "answer", "conditional_answer", ""}


def _graph_with_family_curations() -> InMemoryGraphStore:
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Decision_PivotToNEXUS", type="Decision"))
    apply_oracle_family_curations(graph)
    return graph


def test_l1_qualitative_compare_rag_updates():
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    result = answer_question(
        "Compare RAG vs NEXUS for knowledge updates.",
        _graph_with_family_curations(),
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Decision_PivotToNEXUS"],
    )
    assert result["realization"]["strategy"] == "l1_qualitative_compare"
    assert "re-index" in result["answer"].casefold()
    assert "replaces" in result["answer"].casefold()
    assert "O(1)" in result["answer"]


def test_l1_qualitative_compare_phase_prose():
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    result = answer_question(
        "Compare SAM phase 1-4 research vs NEXUS phase 5.",
        _graph_with_family_curations(),
        model=_NeverGenerate(),
        config=config,
        entry_nodes_override=["Decision_PivotToNEXUS"],
    )
    assert result["realization"]["strategy"] == "l1_qualitative_compare"
    assert "Phase 1-4" in result["answer"]
    assert "Phase 5" in result["answer"]


def test_temporal_valid_window_and_retract_families():
    graph = _graph_with_family_curations()
    config = ProductionNEXUSConfig.l1_acceptance(max_entry_nodes=4)
    runner = NEXUSRunner(graph, config, model=_NeverGenerate())
    records = [
        {
            "id": "family_temporal_002",
            "question": (
                "As valid in 2019, did Concept_LegacyFlatMemory depend on "
                "Module_LegacySelector?"
            ),
            "gold_entities": ["Concept_LegacyFlatMemory"],
            "as_known_at": "",
            "as_valid_at": "2019-06-01T00:00:00+00:00",
        },
        {
            "id": "family_temporal_003",
            "question": (
                "As known in mid-2026, does Claim_TempPivotDependency still "
                "depend on Module_LegacySelector?"
            ),
            "gold_entities": ["Claim_TempPivotDependency"],
            "as_known_at": "2026-06-01T00:00:00+00:00",
            "as_valid_at": "",
        },
        {
            "id": "family_temporal_004",
            "question": (
                "As known after the NEXUS pivot, according to the graph, "
                "what does Decision_PivotToNEXUS replace?"
            ),
            "gold_entities": ["Decision_PivotToNEXUS", "Concept_LegacyFlatMemory"],
            "as_known_at": "2026-07-09T00:00:00+00:00",
            "as_valid_at": "2026-07-09T00:00:00+00:00",
        },
    ]
    pipeline = runner.run_oracle(records, source_sha="test")
    assert not pipeline.errors
    by_id = {qr.question_id: qr for qr in pipeline.per_question}
    assert "as-valid-at" in by_id["family_temporal_002"].answer
    assert "retracted" in by_id["family_temporal_003"].answer.casefold()
    assert "Concept_LegacyFlatMemory" in by_id["family_temporal_004"].answer
    assert "replaces" in by_id["family_temporal_004"].answer.casefold()


def test_answer_plan_binding_requires_oracle_fact_and_predicted_lag():
    binding, reason = _is_binding(
        {
            "proof_valid_rate": 0.98,
            "gold_path_recall_mean": 0.96,
            "entry_recall_mean": 0.95,
            "fact_accuracy_mean": 0.40,
            "oracle_fact_accuracy_mean": 0.55,
            "realizer_backend": "l1_acceptance",
            "dummy_model": False,
            "model": "SynthesizingModel",
        }
    )
    assert binding is True
    assert "gap=" in reason

    blocked, blocked_reason = _is_binding(
        {
            "proof_valid_rate": 0.98,
            "gold_path_recall_mean": 0.96,
            "entry_recall_mean": 0.95,
            "fact_accuracy_mean": 0.54,
            "oracle_fact_accuracy_mean": 0.55,
            "realizer_backend": "l1_acceptance",
            "dummy_model": False,
            "model": "SynthesizingModel",
        }
    )
    assert blocked is False
    assert "NOT the binding constraint" in blocked_reason
