"""Tests for evaluation result schema, metrics, and stats."""

from __future__ import annotations

import pytest

from nexus.evaluation import (
    RESULT_SCHEMA_VERSION,
    TerminalOutcome,
    build_question_record,
    compute_grounded_correct,
    empty_metric_applicability,
    mcnemar_exact,
    paired_bootstrap_ci,
    paired_effect_size,
    summarize_metrics,
    validate_result_artifact,
)


def _metric(value: float, reason: str = "test") -> dict:
    return {
        "applicable": True,
        "value": value,
        "numerator": value,
        "denominator": 1.0,
        "reason": reason,
    }


def test_every_question_has_terminal_outcome_and_metrics():
    metrics = empty_metric_applicability()
    metrics["grounded_correct"] = _metric(1.0)
    row = build_question_record(
        question_id="q1",
        domain="mini",
        question_type="single_hop",
        dataset_id="mini-v1",
        dataset_sha256="abc",
        system_id="nexus_l1",
        profile="l1_acceptance",
        config_hash="deadbeef",
        config_identity_schema="nexus-config-identity-v2",
        model_id="none",
        checkpoint_id="none",
        source_commit="5181031",
        executed_at_utc="2026-07-22T00:00:00Z",
        terminal_outcome=TerminalOutcome.ANSWERED,
        metrics=metrics,
    )
    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": "2026-07-22T00:00:00Z",
        "source_commit": "5181031",
        "dataset_id": "mini-v1",
        "dataset_sha256": "abc",
        "system_id": "nexus_l1",
        "profile": "l1_acceptance",
        "config_hash": "deadbeef",
        "questions_total": 1,
        "per_question": [row.to_dict()],
        "aggregates": {
            "grounded_correct": summarize_metrics(
                [row.to_dict()], "grounded_correct"
            ).to_dict()
        },
        "status": "VALID",
    }
    assert validate_result_artifact(artifact) == []


def test_incomplete_artifact_fails_validation():
    errors = validate_result_artifact({"schema_version": RESULT_SCHEMA_VERSION})
    assert errors
    assert any("missing top-level field" in e for e in errors)


def test_legacy_schema_requires_flag():
    errors = validate_result_artifact(
        {
            "schema_version": "nexus-architecture-validation-v1",
            "source_sha": "abc",
        }
    )
    assert any("legacy_schema=true" in e for e in errors)
    ok = validate_result_artifact(
        {
            "schema_version": "nexus-architecture-validation-v1",
            "legacy_schema": True,
            "source_sha": "abc",
        }
    )
    assert ok == []


def test_grounded_correct_includes_abstain_questions():
    ok = compute_grounded_correct(
        answer="Insufficient evidence to answer.",
        gold_answer="Insufficient evidence to answer.",
        should_abstain=True,
        answer_correct=None,
        material_claims_supported=None,
        citations_entail=None,
        temporal_ok=None,
    )
    assert ok.value == 1.0
    assert ok.denominator == 1.0


def test_grounded_correct_fail_closed_without_adjudication():
    bad = compute_grounded_correct(
        answer="Something.",
        gold_answer="Gold.",
        should_abstain=False,
        answer_correct=None,
        material_claims_supported=None,
        citations_entail=None,
        temporal_ok=None,
    )
    assert bad.value == 0.0
    assert "fail_closed" in bad.reason


def test_summarize_metrics_exposes_numerator_denominator():
    rows = [
        {"metrics": {"grounded_correct": _metric(1.0)}},
        {"metrics": {"grounded_correct": _metric(0.0)}},
        {
            "metrics": {
                "grounded_correct": {
                    "applicable": False,
                    "value": None,
                    "numerator": None,
                    "denominator": None,
                    "reason": "n/a",
                }
            }
        },
    ]
    summary = summarize_metrics(rows, "grounded_correct")
    assert summary.numerator == 1.0
    assert summary.denominator == 2.0
    assert summary.value == pytest.approx(0.5)


def test_paired_bootstrap_and_mcnemar():
    left = [1.0, 1.0, 0.0, 1.0]
    right = [0.0, 1.0, 0.0, 0.0]
    ci = paired_bootstrap_ci(left, right, n_bootstrap=200, seed=1)
    assert ci["n"] == 4
    assert ci["mean_diff"] is not None
    assert ci["ci_low"] is not None
    mc = mcnemar_exact(
        [True, True, False, True],
        [False, True, False, False],
    )
    assert mc["b"] == 2
    assert mc["c"] == 0
    effect = paired_effect_size(left, right)
    assert effect["n"] == 4
