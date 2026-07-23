"""Primary and secondary evaluation metrics with explicit denominators.

The primary metric is *grounded correct answer rate*: every question
contributes to the denominator. An answer passes only when the conclusion is
correct, material claims are supported, citations/evidence entail those
claims, temporal constraints are respected, and abstention is used when
evidence is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MetricValue:
    """A single metric with explicit numerator/denominator."""

    name: str
    applicable: bool
    value: float | None
    numerator: float | None
    denominator: float | None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "applicable": self.applicable,
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "reason": self.reason,
        }


def _is_abstain_text(answer: str) -> bool:
    text = (answer or "").casefold()
    return (not text.strip()) or ("insufficient evidence" in text)


def compute_grounded_correct(
    *,
    answer: str,
    gold_answer: str,
    should_abstain: bool,
    answer_correct: bool | None,
    material_claims_supported: bool | None,
    citations_entail: bool | None,
    temporal_ok: bool | None,
    evidence_sufficient: bool | None = None,
) -> MetricValue:
    """Compute the primary grounded-correct binary for one question.

    When structured adjudication fields are omitted, falls back to a
    deterministic proxy: abstain questions must abstain; otherwise the
    answer must be non-abstaining and ``answer_correct`` must be True.
    Missing structured fields are treated as fail-closed (not grounded).
    """
    predicted_abstain = _is_abstain_text(answer)

    if should_abstain:
        ok = predicted_abstain
        return MetricValue(
            name="grounded_correct",
            applicable=True,
            value=1.0 if ok else 0.0,
            numerator=1.0 if ok else 0.0,
            denominator=1.0,
            reason="abstain_required",
        )

    if predicted_abstain:
        # Answering is required; abstaining when evidence exists is incorrect
        # unless evidence_sufficient is explicitly False.
        if evidence_sufficient is False:
            return MetricValue(
                name="grounded_correct",
                applicable=True,
                value=1.0,
                numerator=1.0,
                denominator=1.0,
                reason="abstain_correct_insufficient_evidence",
            )
        return MetricValue(
            name="grounded_correct",
            applicable=True,
            value=0.0,
            numerator=0.0,
            denominator=1.0,
            reason="incorrect_abstention",
        )

    checks = [
        answer_correct,
        material_claims_supported,
        citations_entail,
        temporal_ok,
    ]
    if any(c is None for c in checks):
        # Primary grounded_correct is not applicable without full adjudication.
        # Callers that need an exploratory proxy must use
        # ``compute_proxy_key_fact_correct`` — never treat that proxy as
        # primary grounded correctness.
        return MetricValue(
            name="grounded_correct",
            applicable=False,
            value=None,
            numerator=None,
            denominator=1.0,
            reason="incomplete_adjudication_not_primary",
        )

    ok = all(bool(c) for c in checks)
    return MetricValue(
        name="grounded_correct",
        applicable=True,
        value=1.0 if ok else 0.0,
        numerator=1.0 if ok else 0.0,
        denominator=1.0,
        reason="full_grounded_criteria",
    )


def compute_proxy_key_fact_correct(
    *,
    answer: str,
    gold_answer: str,
    should_abstain: bool,
    answer_correct: bool | None,
) -> MetricValue:
    """Exploratory key-fact proxy — MUST NOT be reported as grounded_correct."""
    predicted_abstain = _is_abstain_text(answer)
    if should_abstain:
        ok = predicted_abstain
        return MetricValue(
            name="proxy_key_fact_correct",
            applicable=True,
            value=1.0 if ok else 0.0,
            numerator=1.0 if ok else 0.0,
            denominator=1.0,
            reason="proxy_abstain_required",
        )
    if predicted_abstain:
        return MetricValue(
            name="proxy_key_fact_correct",
            applicable=True,
            value=0.0,
            numerator=0.0,
            denominator=1.0,
            reason="proxy_incorrect_abstention",
        )
    if answer_correct is None:
        return MetricValue(
            name="proxy_key_fact_correct",
            applicable=False,
            value=None,
            numerator=None,
            denominator=1.0,
            reason="proxy_unscored",
        )
    ok = bool(answer_correct)
    return MetricValue(
        name="proxy_key_fact_correct",
        applicable=True,
        value=1.0 if ok else 0.0,
        numerator=1.0 if ok else 0.0,
        denominator=1.0,
        reason="proxy_key_fact_only_not_grounded_correct",
    )


def summarize_metrics(
    records: Iterable[Mapping[str, Any]],
    metric_name: str,
) -> MetricValue:
    """Aggregate a per-question metric; denominator is all applicable rows."""
    rows = list(records)
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        metrics = row.get("metrics") or {}
        slot = metrics.get(metric_name) or {}
        if not slot.get("applicable", False):
            continue
        denominator += 1.0
        value = slot.get("value")
        if value is None:
            continue
        numerator += float(value)
    if denominator == 0.0:
        return MetricValue(
            name=metric_name,
            applicable=False,
            value=None,
            numerator=None,
            denominator=0.0,
            reason="no_applicable_rows",
        )
    return MetricValue(
        name=metric_name,
        applicable=True,
        value=round(numerator / denominator, 6),
        numerator=numerator,
        denominator=denominator,
        reason="aggregate_over_applicable",
    )
