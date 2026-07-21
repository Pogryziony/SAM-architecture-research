"""Architecture validation campaign helpers and ER alias hygiene."""

from __future__ import annotations

from benchmarks.run_architecture_validation import decide_verdict
from benchmarks.run_benchmark import build_benchmark_graph
from benchmarks.run_oracle_vs_predicted import DEFAULT_ER3_DIR, build_predicted_runner
from nexus.reasoning.model_interface import DummyModel


def test_decide_verdict_validated_when_thresholds_and_baselines_hold():
    nexus = {
        "metrics": {
            "entry_recall_mean": 0.95,
            "gold_path_recall_mean": 0.96,
            "proof_valid_rate": 0.97,
            "fact_accuracy_mean": 0.72,
        }
    }
    rag = {"metrics": {"fact_accuracy_mean": 0.40}}
    llm = {"metrics": {"fact_accuracy_mean": 0.20}}
    verdict = decide_verdict(
        nexus, rag, llm, oracle_fact=0.73, predicted_fact=0.72
    )
    assert verdict["decision"] == "VALIDATED"
    assert verdict["checks"]["answerplan_binding"] is False


def test_decide_verdict_rejects_when_surface_fails():
    nexus = {
        "metrics": {
            "entry_recall_mean": 0.95,
            "gold_path_recall_mean": 0.96,
            "proof_valid_rate": 0.97,
            "fact_accuracy_mean": 0.55,
        }
    }
    rag = {"metrics": {"fact_accuracy_mean": 0.40}}
    llm = {"metrics": {"fact_accuracy_mean": 0.20}}
    verdict = decide_verdict(
        nexus, rag, llm, oracle_fact=0.55, predicted_fact=0.55
    )
    assert verdict["decision"] == "REJECTED"


def test_entry_zero_aliases_recover_union_handoff():
    graph, _ = build_benchmark_graph()
    runner, _ = build_predicted_runner(
        graph,
        predicted_resolver="union",
        er3_dir=DEFAULT_ER3_DIR,
        model=DummyModel(),
        realizer_backend="l1_acceptance",
    )
    resolver = runner._entity_resolver
    cases = [
        (
            "How does the rule-based verifier work?",
            ["Decision_PivotToNEXUS"],
        ),
        (
            "Why does SAM's 3-hop reasoning collapse between +8 and +16 distractors?",
            ["Decision_PivotToNEXUS"],
        ),
        (
            "What was the transition point from Phase 1 (Pipeline Setup) to Phase 2 (Core Validation)?",
            ["Exp_0_Diagnosis", "Exp_0_13B_RealisticDistractors"],
        ),
    ]
    for question, gold in cases:
        selected = list(resolver.resolve(question, graph).selected_entity_ids)
        hits = [g for g in gold if g in selected]
        assert hits, f"expected some gold in entry for: {question}"
