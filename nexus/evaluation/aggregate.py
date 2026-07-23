"""Canonical aggregation from schema-valid per-question records only."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from nexus.evaluation.metrics import summarize_metrics
from nexus.evaluation.schema import RESULT_SCHEMA_VERSION, normalize_terminal_outcome
from nexus.evaluation.validate import ValidationError, validate_result_artifact


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def aggregate_question_records(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute all Phase-2 aggregates from per-question records alone."""
    n = len(rows)
    outcomes = Counter(
        normalize_terminal_outcome(r.get("terminal_outcome", "failed")).value
        for r in rows
    )
    latencies = [
        float(r["latency_ms"])
        for r in rows
        if r.get("latency_ms") is not None and math.isfinite(float(r["latency_ms"]))
    ]
    rss = [
        float(r["peak_rss_mb"])
        for r in rows
        if r.get("peak_rss_mb") is not None and math.isfinite(float(r["peak_rss_mb"]))
    ]

    failure_categories = Counter(
        str(r.get("failure_category") or "none") for r in rows
    )

    metric_names = set()
    for r in rows:
        metric_names.update((r.get("metrics") or {}).keys())

    metric_aggs = {
        name: summarize_metrics(rows, name).to_dict() for name in sorted(metric_names)
    }

    # Abstention P/R/F1 from terminal outcomes vs should_abstain in metrics/env
    tp = fp = fn = tn = 0
    for r in rows:
        predicted = normalize_terminal_outcome(
            r.get("terminal_outcome", "failed")
        ).value == "abstained" or bool(r.get("abstention"))
        # Gold signal if present in structured_evidence or metrics reason tags
        gold_abstain = False
        env = r.get("execution_environment") or {}
        if "should_abstain" in env:
            gold_abstain = bool(env["should_abstain"])
        metrics = r.get("metrics") or {}
        gc = metrics.get("grounded_correct") or {}
        if gc.get("reason") == "abstain_required":
            gold_abstain = True
        if predicted and gold_abstain:
            tp += 1
        elif predicted and not gold_abstain:
            fp += 1
        elif (not predicted) and gold_abstain:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        None
        if prec is None or rec is None or (prec + rec) == 0
        else 2 * prec * rec / (prec + rec)
    )

    by_domain: dict[str, dict[str, Any]] = {}
    by_qtype: dict[str, dict[str, Any]] = {}
    domain_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    type_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        domain_groups[str(r.get("domain") or "unknown")].append(r)
        type_groups[str(r.get("question_type") or "unknown")].append(r)
    for key, group in domain_groups.items():
        by_domain[key] = {
            "questions_total": len(group),
            "grounded_correct": summarize_metrics(group, "grounded_correct").to_dict(),
            "terminal_outcomes": dict(
                Counter(
                    normalize_terminal_outcome(r.get("terminal_outcome", "failed")).value
                    for r in group
                )
            ),
        }
    for key, group in type_groups.items():
        by_qtype[key] = {
            "questions_total": len(group),
            "grounded_correct": summarize_metrics(group, "grounded_correct").to_dict(),
            "terminal_outcomes": dict(
                Counter(
                    normalize_terminal_outcome(r.get("terminal_outcome", "failed")).value
                    for r in group
                )
            ),
        }

    answered = outcomes.get("answered", 0)
    return {
        "questions_total": {
            "applicable": True,
            "value": float(n),
            "numerator": float(n),
            "denominator": float(n),
            "reason": "all_input_questions",
        },
        "terminal_outcome_counts": dict(outcomes),
        "answer_coverage": {
            "applicable": True,
            "value": (answered / n) if n else None,
            "numerator": float(answered),
            "denominator": float(n),
            "reason": "answered_over_all_questions",
        },
        "metrics": metric_aggs,
        "grounded_correct": metric_aggs.get(
            "grounded_correct",
            {
                "applicable": False,
                "value": None,
                "numerator": None,
                "denominator": 0.0,
                "reason": "not_computed",
            },
        ),
        "abstention": {
            "precision": {
                "applicable": prec is not None,
                "value": None if prec is None else round(prec, 6),
                "numerator": float(tp),
                "denominator": float(tp + fp),
                "reason": "tp/(tp+fp)",
            },
            "recall": {
                "applicable": rec is not None,
                "value": None if rec is None else round(rec, 6),
                "numerator": float(tp),
                "denominator": float(tp + fn),
                "reason": "tp/(tp+fn)",
            },
            "f1": {
                "applicable": f1 is not None,
                "value": None if f1 is None else round(f1, 6),
                "numerator": None if f1 is None else round(f1, 6),
                "denominator": 1.0,
                "reason": "harmonic_mean",
            },
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "latency_ms": {
            "n_scored": len(latencies),
            "n_total": n,
            "p50": None if not latencies else round(_percentile(latencies, 0.50), 3),
            "p95": None if not latencies else round(_percentile(latencies, 0.95), 3),
            "p99": None if not latencies else round(_percentile(latencies, 0.99), 3),
            "mean": None
            if not latencies
            else round(sum(latencies) / len(latencies), 3),
        },
        "peak_rss_mb": {
            "n_scored": len(rss),
            "n_total": n,
            "max": None if not rss else round(max(rss), 3),
            "mean": None if not rss else round(sum(rss) / len(rss), 3),
        },
        "failure_categories": dict(failure_categories),
        "by_domain": by_domain,
        "by_question_type": by_qtype,
    }


def regenerate_aggregates(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Delete-and-rebuild aggregates from per-question rows; fail closed."""
    errors = validate_result_artifact(dict(artifact))
    if errors:
        raise ValidationError("; ".join(errors))
    if artifact.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValidationError(
            f"regenerate_aggregates requires {RESULT_SCHEMA_VERSION}"
        )
    rows = artifact.get("per_question")
    if not isinstance(rows, list):
        raise ValidationError("per_question must be a list")
    return aggregate_question_records(rows)


def assert_homogeneous_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed when a batch mixes config/dataset/system identities."""
    if not rows:
        return
    keys = (
        "dataset_id",
        "dataset_sha256",
        "system_id",
        "profile",
        "config_hash",
        "source_commit",
    )
    first = rows[0]
    for key in keys:
        expected = first.get(key)
        for i, row in enumerate(rows):
            if row.get(key) != expected:
                raise ValidationError(
                    f"mixed identity at per_question[{i}].{key}: "
                    f"{row.get(key)!r} != {expected!r}"
                )
    ids = [str(r.get("question_id")) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValidationError("duplicate question_id in per_question")
