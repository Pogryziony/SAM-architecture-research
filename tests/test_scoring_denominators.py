"""Scoring aggregates must publish denominators for reduced subsets."""

from __future__ import annotations

from benchmarks.run_architecture_validation import _mean_with_denominators
from benchmarks.run_oracle_vs_predicted import summarize_rows


def test_summarize_rows_exposes_fact_denominator_when_some_unscorable():
    rows = [
        {
            "fact_accuracy": 1.0,
            "token_f1": 1.0,
            "gold_path_recall": 1.0,
            "gold_entity_coverage": 1.0,
            "entry_recall": 1.0,
            "pool_recall": 1.0,
            "predicted_abstain": False,
            "should_abstain": False,
            "proof_valid": True,
            "provenance_coverage": 1.0,
            "latency_ms": 10.0,
            "selected_entry_nodes": ["A"],
        },
        {
            "fact_accuracy": None,
            "token_f1": 0.0,
            "gold_path_recall": None,
            "gold_entity_coverage": 0.0,
            "entry_recall": 0.0,
            "pool_recall": 0.0,
            "predicted_abstain": True,
            "should_abstain": True,
            "proof_valid": False,
            "provenance_coverage": 0.0,
            "latency_ms": 11.0,
            "selected_entry_nodes": [],
        },
    ]
    metrics = summarize_rows(rows)
    assert metrics["questions_total"] == 2
    assert metrics["fact_accuracy_mean"] == 1.0
    assert metrics["fact_accuracy_n_scored"] == 1
    assert metrics["fact_accuracy_n_unscorable"] == 1
    assert metrics["gold_path_recall_n_scored"] == 1
    assert metrics["gold_path_recall_n_unscorable"] == 1
    assert metrics["proof_valid_n_scored"] == 2


def test_architecture_mean_with_denominators():
    stats = _mean_with_denominators([1.0, None, 0.0])
    assert stats["mean"] == 0.5
    assert stats["n_scored"] == 2
    assert stats["n_total"] == 3
    assert stats["n_unscorable"] == 1
