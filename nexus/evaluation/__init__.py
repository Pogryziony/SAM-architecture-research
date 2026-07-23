"""Evaluation schemas, metrics, aggregation, and statistical helpers for NEXUS."""

from nexus.evaluation.aggregate import (
    aggregate_question_records,
    assert_homogeneous_identity,
    regenerate_aggregates,
)
from nexus.evaluation.compare import compare_paired_artifacts
from nexus.evaluation.export import (
    classify_failure_category,
    classify_terminal_outcome,
    pipeline_to_eval_artifact,
    question_result_to_outcome,
)
from nexus.evaluation.metrics import (
    MetricValue,
    compute_grounded_correct,
    compute_proxy_key_fact_correct,
    summarize_metrics,
)
from nexus.evaluation.schema import (
    RESULT_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSIONS,
    QuestionOutcome,
    TerminalOutcome,
    build_question_record,
    empty_metric_applicability,
    normalize_terminal_outcome,
)
from nexus.evaluation.stats import (
    mcnemar_exact,
    paired_bootstrap_ci,
    paired_effect_size,
)
from nexus.evaluation.validate import ValidationError, validate_result_artifact

__all__ = [
    "LEGACY_SCHEMA_VERSIONS",
    "MetricValue",
    "QuestionOutcome",
    "RESULT_SCHEMA_VERSION",
    "TerminalOutcome",
    "ValidationError",
    "aggregate_question_records",
    "assert_homogeneous_identity",
    "build_question_record",
    "classify_failure_category",
    "classify_terminal_outcome",
    "compare_paired_artifacts",
    "compute_grounded_correct",
    "compute_proxy_key_fact_correct",
    "empty_metric_applicability",
    "mcnemar_exact",
    "normalize_terminal_outcome",
    "paired_bootstrap_ci",
    "paired_effect_size",
    "pipeline_to_eval_artifact",
    "question_result_to_outcome",
    "regenerate_aggregates",
    "summarize_metrics",
    "validate_result_artifact",
]
