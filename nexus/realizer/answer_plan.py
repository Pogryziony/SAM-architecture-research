"""Verified input contract for a NEXUS surface Realizer.

The plan contains the answer selected by upstream graph reasoning.  A Realizer
may change wording, but it is never allowed to select evidence or alter facts.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = "nexus-answer-plan-v1"
OPERATORS = {"extract", "compose_path", "compare", "abstain"}
DECISIONS = {"answer", "abstain"}
_IMMUTABLE_RE = re.compile(r"(?<!\w)(?:[+-]?\d+(?:[.,]\d+)?|[A-Z0-9]{2,})(?!\w)")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def compile_answer_plan(record: dict[str, Any]) -> dict[str, Any]:
    """Compile one validated corpus-v2 record into a resolved AnswerPlan."""
    evidence = record["evidence"]
    surface_target = str(record["answer"]).strip()
    operator = record["semantic_plan"]["operator"]
    source_aliases = [
        str(item).strip() for item in record.get("answer_aliases", [])
        if str(item).strip() and str(item).strip().casefold() != surface_target.casefold()
    ]
    answer = (
        "__ABSTAIN__" if operator == "abstain"
        else (source_aliases[0] if source_aliases else surface_target)
    )
    immutable_values = list(dict.fromkeys(_IMMUTABLE_RE.findall(answer)))
    evidence_ids = [str(item["id"]) for item in evidence]
    provenance = [
        {
            "evidence_id": str(item["id"]),
            "title": str(item.get("title", "")).strip(),
            "source_locator": str(item["source_locator"]),
            "evidence_sha256": str(item["text_sha256"]),
        }
        for item in evidence
    ]
    claim = {
        "id": "claim-1",
        "canonical_value": answer,
        "evidence_ids": evidence_ids,
        "grounding": "gold_support_annotation",
    }
    body = {
        "source_record_id": record["id"],
        "language": record["language"],
        "operator": operator,
        "decision": "abstain" if operator == "abstain" else "answer",
        "question": str(record["question"]).strip(),
        "claims": [claim],
        "resolved_answer": {
            "canonical_text": answer,
            "aliases": source_aliases[1:] if source_aliases else [],
            "immutable_values": [] if operator == "abstain" else immutable_values,
            "answer_sha256": _sha256(answer),
        },
        "required_evidence_ids": evidence_ids,
        "provenance": provenance,
        "provenance_coverage": 1.0,
        "verified": True,
        "reasoning_owner": "nexus_upstream",
        "realizer_may_change_facts": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "ap1_" + _sha256(body)[:32],
        **body,
    }


def validate_answer_plan(plan: dict[str, Any], record: dict[str, Any] | None = None) -> list[str]:
    """Return stable validation errors; an empty list means the plan is usable."""
    errors: list[str] = []
    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported_schema")
    if plan.get("language") not in {"en", "pl"}:
        errors.append("invalid_language")
    if not str(plan.get("question", "")).strip():
        errors.append("empty_question")
    if plan.get("operator") not in OPERATORS:
        errors.append("invalid_operator")
    if plan.get("decision") not in DECISIONS:
        errors.append("invalid_decision")
    if (plan.get("operator") == "abstain") != (plan.get("decision") == "abstain"):
        errors.append("operator_decision_mismatch")
    if plan.get("verified") is not True:
        errors.append("unverified")
    if plan.get("reasoning_owner") != "nexus_upstream":
        errors.append("invalid_reasoning_owner")
    if plan.get("realizer_may_change_facts") is not False:
        errors.append("realizer_can_change_facts")
    resolved = plan.get("resolved_answer")
    if not isinstance(resolved, dict) or not str(resolved.get("canonical_text", "")).strip():
        errors.append("missing_resolved_answer")
        resolved = {}
    canonical = str(resolved.get("canonical_text", "")).strip()
    if resolved.get("answer_sha256") != _sha256(canonical):
        errors.append("invalid_answer_hash")
    immutable = resolved.get("immutable_values", [])
    immutable_valid = isinstance(immutable, list) and not any(
        not str(value).strip() for value in immutable
    )
    if plan.get("operator") == "abstain":
        immutable_valid = immutable_valid and immutable == [] and canonical == "__ABSTAIN__"
    else:
        immutable_valid = immutable_valid and all(value in canonical for value in immutable)
    if not immutable_valid:
        errors.append("canonical_answer_not_immutable")
    if not isinstance(resolved.get("aliases", []), list):
        errors.append("invalid_aliases")
    required = plan.get("required_evidence_ids", [])
    if not isinstance(required, list) or not required or len(required) != len(set(required)):
        errors.append("invalid_required_evidence")
        required = []
    provenance = plan.get("provenance", [])
    provenance_ids = [item.get("evidence_id") for item in provenance if isinstance(item, dict)]
    if provenance_ids != required:
        errors.append("provenance_mismatch")
    for item in provenance if isinstance(provenance, list) else []:
        if not isinstance(item, dict) or not str(item.get("source_locator", "")).strip():
            errors.append("invalid_provenance")
            continue
        evidence_hash = str(item.get("evidence_sha256", ""))
        if len(evidence_hash) != 64 or any(char not in "0123456789abcdef" for char in evidence_hash):
            errors.append("invalid_provenance_hash")
    if plan.get("provenance_coverage") != 1.0:
        errors.append("incomplete_provenance")
    claims = plan.get("claims", [])
    if not isinstance(claims, list) or len(claims) != 1:
        errors.append("invalid_claims")
    else:
        claim = claims[0]
        if claim.get("id") != "claim-1" or claim.get("grounding") != "gold_support_annotation":
            errors.append("invalid_claim_contract")
        if claim.get("canonical_value") != canonical:
            errors.append("claim_answer_mismatch")
        if claim.get("evidence_ids") != required:
            errors.append("claim_evidence_mismatch")
    body = {key: value for key, value in plan.items() if key not in {"schema_version", "id"}}
    if plan.get("id") != "ap1_" + _sha256(body)[:32]:
        errors.append("unstable_plan_id")
    if record is not None:
        if plan.get("source_record_id") != record.get("id"):
            errors.append("source_record_mismatch")
        if plan.get("language") != record.get("language"):
            errors.append("record_language_mismatch")
        if plan.get("operator") != record.get("semantic_plan", {}).get("operator"):
            errors.append("record_operator_mismatch")
        record_answer = str(record.get("answer", "")).strip()
        record_aliases = [str(item).strip() for item in record.get("answer_aliases", [])]
        valid_canonical = (
            canonical == "__ABSTAIN__" if plan.get("operator") == "abstain"
            else canonical in [record_answer, *record_aliases]
        )
        if not valid_canonical:
            errors.append("record_answer_mismatch")
        record_ids = [str(item.get("id")) for item in record.get("evidence", [])]
        if required != record_ids:
            errors.append("record_evidence_mismatch")
        expected_provenance = [
            (str(item.get("id")), str(item.get("text_sha256")), str(item.get("source_locator")))
            for item in record.get("evidence", [])
        ]
        actual_provenance = [
            (str(item.get("evidence_id")), str(item.get("evidence_sha256")), str(item.get("source_locator")))
            for item in provenance if isinstance(item, dict)
        ]
        if actual_provenance != expected_provenance:
            errors.append("record_provenance_mismatch")
    return sorted(set(errors))
