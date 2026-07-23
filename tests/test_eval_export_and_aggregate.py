"""Phase 2: runner→schema export, aggregation, zero-hop failure fix."""

from __future__ import annotations

import copy

import pytest

from nexus.domain import load_domain_pack
from nexus.evaluation import (
    RESULT_SCHEMA_VERSION,
    TerminalOutcome,
    aggregate_question_records,
    assert_homogeneous_identity,
    compare_paired_artifacts,
    regenerate_aggregates,
    validate_result_artifact,
)
from nexus.evaluation.export import classify_failure_category
from nexus.evaluation.validate import ValidationError
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner, QuestionResult
from nexus.reasoning.model_interface import DummyModel


def test_zero_hop_answer_not_classified_as_no_graph_paths():
    qr = QuestionResult(
        question_id="z1",
        predicted_entities=["City_Warsaw"],
        selected_entry_nodes=["City_Warsaw"],
        graph_paths_count=0,
        answer="Warsaw is the capital of Poland.",
        evidence_pack={"facts": ["Warsaw is the capital of Poland."]},
        evidence_pack_keys=["facts"],
        verifier_passed=True,
    )
    assert classify_failure_category(qr) == ""


def test_runner_eval_emits_one_terminal_per_question():
    pack = load_domain_pack("mini")
    runner = NEXUSRunner(
        pack.build_graph(),
        ProductionNEXUSConfig.grounded(),
        model=DummyModel(),
    )
    questions = pack.evaluation_tasks()
    artifact = runner.run_eval(
        questions,
        dataset_id="mini-tasks",
        dataset_sha256="a" * 64,
        system_id="nexus_grounded",
        profile="grounded",
        domain_pack_id="mini",
        domain_pack_version="mini-v1",
        model_id="DummyModel",
        comparison_mode="system_level",
    )
    assert artifact["schema_version"] == RESULT_SCHEMA_VERSION
    assert artifact["questions_total"] == len(questions)
    assert len(artifact["per_question"]) == len(questions)
    assert validate_result_artifact(artifact) == []
    outcomes = {r["terminal_outcome"] for r in artifact["per_question"]}
    assert outcomes <= {
        "answered",
        "abstained",
        "failed",
        "timed_out",
        "not_run",
        "invalid_input",
        "error",
    }
    # Unanswerable mini_q3 should abstain, not no_graph_paths failure alone
    by_id = {r["question_id"]: r for r in artifact["per_question"]}
    assert by_id["mini_q3"]["terminal_outcome"] == "abstained"


def test_aggregates_regenerate_exactly():
    pack = load_domain_pack("mini")
    runner = NEXUSRunner(
        pack.build_graph(),
        ProductionNEXUSConfig.lexical_only(),
        model=DummyModel(),
    )
    artifact = runner.run_eval(
        pack.evaluation_tasks(),
        dataset_id="mini-tasks",
        dataset_sha256="b" * 64,
        profile="lexical",
        model_id="DummyModel",
    )
    original = copy.deepcopy(artifact["aggregates"])
    artifact["aggregates"] = {}
    rebuilt = regenerate_aggregates(artifact)
    assert rebuilt["questions_total"]["denominator"] == len(artifact["per_question"])
    assert rebuilt["terminal_outcome_counts"] == original["terminal_outcome_counts"]
    assert "grounded_correct" in rebuilt


def test_mixed_identity_fails_closed():
    rows = [
        {
            "question_id": "a",
            "dataset_id": "d1",
            "dataset_sha256": "x",
            "system_id": "s",
            "profile": "p",
            "config_hash": "h1",
            "source_commit": "c",
        },
        {
            "question_id": "b",
            "dataset_id": "d1",
            "dataset_sha256": "x",
            "system_id": "s",
            "profile": "p",
            "config_hash": "h2",
            "source_commit": "c",
        },
    ]
    with pytest.raises(ValidationError, match="mixed identity"):
        assert_homogeneous_identity(rows)


def test_paired_compare_rejects_not_run_and_placeholders():
    base_row = {
        "question_id": "q1",
        "domain": "mini",
        "question_type": "single_hop",
        "dataset_id": "d",
        "dataset_sha256": "c" * 64,
        "system_id": "nexus_a",
        "profile": "grounded",
        "config_hash": "h",
        "config_identity_schema": "nexus-config-identity-v2",
        "model_id": "m",
        "checkpoint_id": "",
        "source_commit": "s",
        "executed_at_utc": "2026-07-22T00:00:00Z",
        "terminal_outcome": "answered",
        "metrics": {
            "grounded_correct": {
                "applicable": True,
                "value": 1.0,
                "numerator": 1.0,
                "denominator": 1.0,
                "reason": "t",
            }
        },
    }
    left = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": "t",
        "source_commit": "s",
        "dataset_id": "d",
        "dataset_sha256": "c" * 64,
        "system_id": "nexus_a",
        "profile": "grounded",
        "config_hash": "h",
        "questions_total": 1,
        "per_question": [base_row],
        "aggregates": aggregate_question_records([base_row]),
        "status": "VALID",
    }
    right = copy.deepcopy(left)
    right["system_id"] = "placeholder_evidence_blind"
    right["per_question"][0] = {
        **base_row,
        "system_id": "placeholder_evidence_blind",
        "metrics": {
            "grounded_correct": {
                "applicable": True,
                "value": 0.0,
                "numerator": 0.0,
                "denominator": 1.0,
                "reason": "t",
            }
        },
    }
    right["arm_metadata"] = {"is_placeholder": True}
    with pytest.raises(ValidationError, match="placeholder"):
        compare_paired_artifacts(left, right)

    right2 = copy.deepcopy(left)
    right2["system_id"] = "nexus_b"
    right2["status"] = "NOT_RUN"
    right2["per_question"][0] = {**base_row, "system_id": "nexus_b"}
    with pytest.raises(ValidationError, match="NOT_RUN"):
        compare_paired_artifacts(left, right2)


def test_paired_compare_deterministic_fixture():
    def _art(system: str, values: list[float]) -> dict:
        rows = []
        for i, v in enumerate(values):
            rows.append(
                {
                    "question_id": f"q{i}",
                    "domain": "mini",
                    "question_type": "t",
                    "dataset_id": "d",
                    "dataset_sha256": "d" * 64,
                    "system_id": system,
                    "profile": "p",
                    "config_hash": "h",
                    "config_identity_schema": "nexus-config-identity-v2",
                    "model_id": "m",
                    "checkpoint_id": "",
                    "source_commit": "s",
                    "executed_at_utc": "t",
                    "terminal_outcome": "answered",
                    "metrics": {
                        "grounded_correct": {
                            "applicable": True,
                            "value": v,
                            "numerator": v,
                            "denominator": 1.0,
                            "reason": "fixture",
                        }
                    },
                }
            )
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "created_utc": "t",
            "source_commit": "s",
            "dataset_id": "d",
            "dataset_sha256": "d" * 64,
            "system_id": system,
            "profile": "p",
            "config_hash": "h",
            "questions_total": len(rows),
            "per_question": rows,
            "aggregates": aggregate_question_records(rows),
            "status": "VALID",
        }

    left = _art("nexus_strong", [1.0, 1.0, 1.0, 0.0])
    right = _art("nexus_weak", [0.0, 1.0, 0.0, 0.0])
    result = compare_paired_artifacts(left, right, n_bootstrap=200, seed=1)
    assert result["paired_n"] == 4
    assert result["bootstrap"]["mean_diff"] == pytest.approx(0.5)
    assert "mcnemar" in result
    assert result["superiority_verdict"] in {
        "LEFT_BETTER",
        "RIGHT_BETTER",
        "INCONCLUSIVE",
    }
