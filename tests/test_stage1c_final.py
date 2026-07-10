from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.stage1c_final import K, rank_feature_logistic, validate_final_artifact


def test_ranker_rejects_k_above_hard_cap_without_posthoc_filtering():
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        rank_feature_logistic("question", [], graph=None, ranker={"kind": "feature_logistic", "weights": [0] * 6}, k=11)


def test_ranker_returns_at_most_ten_in_ranking_path():
    class Graph:
        def get_node(self, _node_id):
            return None

        def get_outgoing(self, _node_id):
            return []

        def get_incoming(self, _node_id):
            return []

    ranker = {"kind": "feature_logistic", "weights": [0.0] * 6}
    ranked = rank_feature_logistic("question", [f"N{i}" for i in range(30)], Graph(), ranker, K)
    assert len(ranked) == K


def test_final_artifact_schema_accepts_complete_mechanical_shape():
    gates = {name: {"passed": True} for name in (
        "primary_recall", "control_validation", "intent_accuracy", "paraphrase_drop",
        "resolution_rate", "latency_p50", "rss_delta",
    )}
    artifact = {
        "meta": {"question_count": 1, "validated_ids_match": True, "k": 10},
        "winner": "feature_logistic",
        "metrics": {key: 0.5 for key in (
            "recall@1", "recall@5", "recall@10", "precision@10", "intent_accuracy",
            "resolution_rate", "latency_p50_ms", "rss_delta_mb",
        )},
        "baseline": {}, "gates": gates, "decision": "HONEST PASS",
        "question_details": [{}],
    }
    assert validate_final_artifact(artifact) == []


def test_final_artifact_rejects_nonmechanical_decision():
    artifact = {
        "meta": {"question_count": 1, "validated_ids_match": True, "k": 10},
        "winner": "feature_logistic", "metrics": {key: 0.5 for key in (
            "recall@1", "recall@5", "recall@10", "precision@10", "intent_accuracy",
            "resolution_rate", "latency_p50_ms", "rss_delta_mb",
        )}, "baseline": {},
        "gates": {"primary_recall": {"passed": False}}, "decision": "HONEST PASS",
        "question_details": [{}],
    }
    assert any("mechanically" in error for error in validate_final_artifact(artifact))
