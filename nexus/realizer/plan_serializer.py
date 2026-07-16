"""Compact, lossless serialization of a verified AnswerPlan."""

from __future__ import annotations

from typing import Any

from .answer_plan import validate_answer_plan


SERIALIZER_VERSION = "nexus-answer-plan-serializer-v1"
MODEL_SERIALIZER_VERSION = "nexus-answer-plan-model-input-v1"


def serialize_answer_plan(plan: dict[str, Any]) -> str:
    """Serialize facts and provenance without including raw evidence paragraphs."""
    errors = validate_answer_plan(plan)
    if errors:
        raise ValueError("invalid AnswerPlan: " + ",".join(errors))
    answer = plan["resolved_answer"]
    provenance = "; ".join(
        f"{item['evidence_id']}@{item['source_locator']}"
        for item in plan["provenance"]
    )
    immutable = " | ".join(answer["immutable_values"])
    aliases = " | ".join(answer.get("aliases", [])) or "-"
    return "\n".join((
        f"[SCHEMA] {SERIALIZER_VERSION}",
        f"[LANGUAGE] {plan['language']}",
        f"[OPERATOR] {plan['operator']}",
        f"[DECISION] {plan['decision']}",
        f"[QUESTION] {plan['question']}",
        f"[RESOLVED_ANSWER] {answer['canonical_text']}",
        f"[ALIASES] {aliases}",
        f"[IMMUTABLE] {immutable}",
        f"[PROVENANCE] {provenance}",
        "[INSTRUCTION] Realize the resolved answer naturally; preserve every immutable value.",
    ))


def serialize_answer_plan_for_model(plan: dict[str, Any]) -> str:
    """Drop audit-only hashes while retaining the complete linguistic contract."""
    errors = validate_answer_plan(plan)
    if errors:
        raise ValueError("invalid AnswerPlan: " + ",".join(errors))
    answer = plan["resolved_answer"]
    aliases = " | ".join(answer.get("aliases", [])) or "-"
    immutable = " | ".join(answer["immutable_values"]) or "-"
    return "\n".join((
        f"[SCHEMA] {MODEL_SERIALIZER_VERSION}",
        f"[LANGUAGE] {plan['language']}",
        f"[OPERATOR] {plan['operator']}",
        f"[DECISION] {plan['decision']}",
        f"[QUESTION] {plan['question']}",
        f"[FACT] {answer['canonical_text']}",
        f"[ALIASES] {aliases}",
        f"[IMMUTABLE] {immutable}",
        f"[PROVENANCE_COUNT] {len(plan['provenance'])}",
        "[INSTRUCTION] Produce the natural answer in the requested language.",
    ))
