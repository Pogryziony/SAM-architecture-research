"""Fail-closed validation for evaluation result artifacts."""

from __future__ import annotations

from typing import Any

from nexus.evaluation.schema import (
    LEGACY_SCHEMA_VERSIONS,
    RESULT_SCHEMA_VERSION,
    TerminalOutcome,
    normalize_terminal_outcome,
)


class ValidationError(ValueError):
    """Raised when an evaluation artifact violates the schema contract."""


_REQUIRED_QUESTION_FIELDS = (
    "question_id",
    "domain",
    "question_type",
    "dataset_id",
    "dataset_sha256",
    "system_id",
    "profile",
    "config_hash",
    "config_identity_schema",
    "model_id",
    "checkpoint_id",
    "source_commit",
    "executed_at_utc",
    "terminal_outcome",
    "metrics",
)


def validate_result_artifact(artifact: dict[str, Any]) -> list[str]:
    """Validate a result artifact. Returns errors (empty = valid).

    Legacy schemas are accepted only when explicitly marked
    ``legacy_schema: true`` and the schema is in ``LEGACY_SCHEMA_VERSIONS``.
    They are not upgraded silently.
    """
    errors: list[str] = []
    schema = artifact.get("schema_version")
    if schema == RESULT_SCHEMA_VERSION:
        errors.extend(_validate_v1(artifact))
    elif schema in LEGACY_SCHEMA_VERSIONS:
        if not artifact.get("legacy_schema"):
            errors.append(
                "legacy schema requires explicit legacy_schema=true "
                f"(got schema_version={schema!r})"
            )
        if not artifact.get("source_commit") and not artifact.get("source_sha"):
            errors.append("legacy artifact missing source_commit/source_sha")
    else:
        errors.append(f"unsupported schema_version: {schema!r}")
    return errors


def assert_valid_result_artifact(artifact: dict[str, Any]) -> None:
    errors = validate_result_artifact(artifact)
    if errors:
        raise ValidationError("; ".join(errors))


def _validate_v1(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "created_utc",
        "source_commit",
        "dataset_id",
        "dataset_sha256",
        "system_id",
        "profile",
        "config_hash",
        "questions_total",
        "per_question",
        "aggregates",
        "status",
    ):
        if key not in artifact:
            errors.append(f"missing top-level field: {key}")

    rows = artifact.get("per_question")
    if not isinstance(rows, list):
        errors.append("per_question must be a list")
        return errors

    total = artifact.get("questions_total")
    if isinstance(total, int) and total != len(rows):
        errors.append(
            f"questions_total ({total}) != len(per_question) ({len(rows)})"
        )

    seen: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"per_question[{i}] is not an object")
            continue
        for field in _REQUIRED_QUESTION_FIELDS:
            if field not in row:
                errors.append(f"per_question[{i}] missing {field}")
        qid = row.get("question_id")
        if isinstance(qid, str):
            if qid in seen:
                errors.append(f"duplicate question_id: {qid}")
            seen.add(qid)
        outcome = row.get("terminal_outcome")
        try:
            normalize_terminal_outcome(outcome)
        except Exception:
            errors.append(
                f"per_question[{i}] invalid terminal_outcome: {outcome!r}"
            )
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            errors.append(f"per_question[{i}] metrics must be a non-empty object")
        else:
            for name, slot in metrics.items():
                if not isinstance(slot, dict):
                    errors.append(
                        f"per_question[{i}].metrics[{name}] must be an object"
                    )
                    continue
                if "applicable" not in slot or "value" not in slot:
                    errors.append(
                        f"per_question[{i}].metrics[{name}] needs "
                        "applicable and value"
                    )
                if slot.get("applicable") and (
                    slot.get("numerator") is None or slot.get("denominator") is None
                ):
                    errors.append(
                        f"per_question[{i}].metrics[{name}] applicable "
                        "metric missing numerator/denominator"
                    )

    aggregates = artifact.get("aggregates")
    if isinstance(aggregates, dict):
        for name, slot in aggregates.items():
            if not isinstance(slot, dict):
                continue
            if slot.get("applicable") and slot.get("denominator") in (None, 0):
                # denominator 0 with applicable true is invalid
                if slot.get("denominator") is None:
                    errors.append(
                        f"aggregates[{name}] applicable metric missing denominator"
                    )
    return errors
