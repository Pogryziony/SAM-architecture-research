"""Paired statistical comparison over schema-valid evaluation artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from nexus.evaluation.adjudication import artifact_has_pending_adjudication
from nexus.evaluation.aggregate import assert_homogeneous_identity
from nexus.evaluation.schema import RESULT_SCHEMA_VERSION, normalize_terminal_outcome
from nexus.evaluation.stats import mcnemar_exact, paired_bootstrap_ci, paired_effect_size
from nexus.evaluation.validate import ValidationError, validate_result_artifact


_PLACEHOLDER_SYSTEM_MARKERS = (
    "placeholder",
    "synthesizingmodel",
    "evidenceblind",
    "dummy",
)


def _rows_by_id(artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = artifact.get("per_question") or []
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            raise ValidationError("question_id missing in per_question row")
        if qid in out:
            raise ValidationError(f"duplicate question_id: {qid}")
        out[qid] = row
    return out


def _metric_binary(row: Mapping[str, Any], metric_name: str) -> bool | None:
    slot = (row.get("metrics") or {}).get(metric_name) or {}
    if not slot.get("applicable"):
        return None
    value = slot.get("value")
    if value is None:
        return None
    return float(value) >= 0.5


def _is_placeholder_artifact(artifact: Mapping[str, Any]) -> bool:
    system = str(artifact.get("system_id") or "").casefold()
    meta = artifact.get("arm_metadata") or {}
    if meta.get("is_placeholder"):
        return True
    return any(marker in system for marker in _PLACEHOLDER_SYSTEM_MARKERS)


def compare_paired_artifacts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    metric_name: str = "grounded_correct",
    n_bootstrap: int = 2000,
    seed: int = 0,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    """Paired comparison; fail closed on identity / coverage mismatches.

    Preregistered error/timeout policy: FAILED and TIMED_OUT count as
    incorrect (0.0) for binary metrics when the metric slot is missing —
    they remain in the denominator. NOT_RUN arms cannot produce a
    superiority verdict.
    """
    for art, label in ((left, "left"), (right, "right")):
        errors = validate_result_artifact(dict(art))
        if errors:
            raise ValidationError(f"{label}: " + "; ".join(errors))
        if art.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValidationError(
                f"{label} schema_version must be {RESULT_SCHEMA_VERSION}"
            )
        if art.get("status") == "NOT_RUN":
            raise ValidationError(
                f"{label} status is NOT_RUN; refusing statistical comparison"
            )
        if (not allow_placeholders) and _is_placeholder_artifact(art):
            raise ValidationError(
                f"{label} looks like a placeholder/deterministic control; "
                "refusing superiority statistics "
                "(pass allow_placeholders=True only for diagnostic fixtures)"
            )
        if artifact_has_pending_adjudication(art):
            raise ValidationError(
                f"{label} has PENDING_ADJUDICATION; refusing superiority verdict"
            )

    left_mode = str(left.get("comparison_mode") or "")
    right_mode = str(right.get("comparison_mode") or "")
    if left_mode and right_mode and left_mode != right_mode:
        raise ValidationError(
            "comparison_mode mismatch; controlled and system_level "
            "families must not be mixed in one statistical verdict"
        )

    if left.get("dataset_id") != right.get("dataset_id"):
        raise ValidationError("dataset_id mismatch")
    if left.get("dataset_sha256") != right.get("dataset_sha256"):
        raise ValidationError("dataset_sha256 mismatch")

    left_rows = _rows_by_id(left)
    right_rows = _rows_by_id(right)
    assert_homogeneous_identity(list(left_rows.values()))
    assert_homogeneous_identity(list(right_rows.values()))

    if set(left_rows) != set(right_rows):
        missing_l = sorted(set(right_rows) - set(left_rows))
        missing_r = sorted(set(left_rows) - set(right_rows))
        raise ValidationError(
            "question set mismatch; "
            f"missing_left={missing_l[:5]}; missing_right={missing_r[:5]}"
        )

    left_scores: list[float] = []
    right_scores: list[float] = []
    left_bin: list[bool] = []
    right_bin: list[bool] = []
    excluded: list[str] = []
    for qid in sorted(left_rows):
        lrow, rrow = left_rows[qid], right_rows[qid]
        l_out = normalize_terminal_outcome(lrow.get("terminal_outcome", "failed"))
        r_out = normalize_terminal_outcome(rrow.get("terminal_outcome", "failed"))
        if l_out.value == "not_run" or r_out.value == "not_run":
            raise ValidationError(
                f"question {qid} has NOT_RUN terminal outcome; "
                "refusing paired comparison"
            )
        lb = _metric_binary(lrow, metric_name)
        rb = _metric_binary(rrow, metric_name)
        # Fail/timeout without metric → 0.0 (remain in denominator)
        if lb is None:
            if l_out.value in {"failed", "timed_out", "invalid_input"}:
                lb = False
            else:
                excluded.append(qid)
                continue
        if rb is None:
            if r_out.value in {"failed", "timed_out", "invalid_input"}:
                rb = False
            else:
                excluded.append(qid)
                continue
        left_bin.append(lb)
        right_bin.append(rb)
        left_scores.append(1.0 if lb else 0.0)
        right_scores.append(1.0 if rb else 0.0)

    if not left_scores:
        raise ValidationError("no paired scorable questions after policy filters")

    bootstrap = paired_bootstrap_ci(
        left_scores, right_scores, n_bootstrap=n_bootstrap, seed=seed
    )
    effect = paired_effect_size(left_scores, right_scores)
    mcnemar = mcnemar_exact(left_bin, right_bin)

    # Holm correction placeholder for single primary metric (identity)
    p = float(mcnemar["p_value"])
    holm_adjusted = min(1.0, p)  # single test

    return {
        "schema_version": "nexus-paired-comparison-v1",
        "metric_name": metric_name,
        "left_system": left.get("system_id"),
        "right_system": right.get("system_id"),
        "dataset_id": left.get("dataset_id"),
        "dataset_sha256": left.get("dataset_sha256"),
        "paired_n": len(left_scores),
        "questions_total": len(left_rows),
        "excluded_question_ids": excluded,
        "error_timeout_policy": (
            "FAILED/TIMED_OUT/INVALID_INPUT count as incorrect (0) when the "
            "primary metric slot is absent; NOT_RUN forbids comparison"
        ),
        "bootstrap": bootstrap,
        "effect_size": effect,
        "mcnemar": mcnemar,
        "multiple_comparison": {
            "method": "holm",
            "n_tests": 1,
            "adjusted_p_value": holm_adjusted,
        },
        "superiority_verdict": (
            "LEFT_BETTER"
            if (
                bootstrap.get("ci_low") is not None
                and float(bootstrap["ci_low"]) > 0
                and holm_adjusted < 0.05
            )
            else (
                "RIGHT_BETTER"
                if (
                    bootstrap.get("ci_high") is not None
                    and float(bootstrap["ci_high"]) < 0
                    and holm_adjusted < 0.05
                )
                else "INCONCLUSIVE"
            )
        ),
        "resource_pareto": _resource_pareto(left, right),
    }


def _resource_pareto(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    la = (left.get("aggregates") or {}).get("latency_ms") or {}
    ra = (right.get("aggregates") or {}).get("latency_ms") or {}
    lr = (left.get("aggregates") or {}).get("peak_rss_mb") or {}
    rr = (right.get("aggregates") or {}).get("peak_rss_mb") or {}
    return {
        "left_latency_p50_ms": la.get("p50"),
        "right_latency_p50_ms": ra.get("p50"),
        "left_peak_rss_mb_max": lr.get("max"),
        "right_peak_rss_mb_max": rr.get("max"),
        "note": (
            "Pareto points are descriptive; dominance claims require both "
            "quality CI and resource deltas under the same comparison mode"
        ),
    }
