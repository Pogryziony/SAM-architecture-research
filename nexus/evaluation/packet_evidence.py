"""Structured evidence packs for human adjudication items."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def evidence_from_system_row(row: Mapping[str, Any] | None) -> list[str]:
    """Extract citation / structured evidence lines for a rater packet."""
    if not row:
        return []
    lines: list[str] = []
    structured = row.get("structured_evidence")
    if isinstance(structured, dict) and structured:
        lines.append("structured_evidence=" + json.dumps(structured, ensure_ascii=False)[:2000])
    for cite in row.get("citations") or []:
        lines.append(f"citation:{cite}")
    for doc in row.get("retrieved_documents") or []:
        lines.append(f"retrieved:{doc}")
    env = row.get("execution_environment") or {}
    if env.get("prompt"):
        # Include a truncated evidence section from the prompt when present
        prompt = str(env["prompt"])
        if "Evidence:" in prompt:
            ev = prompt.split("Evidence:", 1)[1].strip()
            lines.append("prompt_evidence=" + ev[:2500])
    as_of = row.get("as_known_at") or row.get("as_valid_at") or env.get("as_known_at")
    if as_of:
        lines.append(f"temporal_context:{as_of}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def build_permitted_evidence_map(
    questions: Sequence[Mapping[str, Any]],
    system_answers: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, list[str]]:
    """Union evidence across systems for each human-dependent question."""
    out: dict[str, list[str]] = {}
    for q in questions:
        qid = str(q.get("id") or q.get("question_id") or "")
        gold = str(q.get("gold_answer") or "")
        entities = [str(x) for x in (q.get("gold_entities") or [])]
        base = []
        if gold:
            base.append(f"gold_answer_reference_held_from_annotator=hash_only")
        if entities:
            base.append("gold_entities=" + ",".join(entities[:12]))
        if q.get("as_known_at") or q.get("as_valid_at"):
            base.append(
                "question_temporal="
                + json.dumps(
                    {
                        "as_known_at": q.get("as_known_at"),
                        "as_valid_at": q.get("as_valid_at"),
                    },
                    sort_keys=True,
                )
            )
        for _sid, by_q in system_answers.items():
            base.extend(evidence_from_system_row(by_q.get(qid)))
        # Raters need *some* evidence for support/citation dimensions
        if not any(x.startswith(("citation:", "retrieved:", "prompt_evidence=", "structured_evidence=")) for x in base):
            base.append(
                "WARNING:no_system_evidence_available; "
                "material_claim_support and citation_entailment must be marked unscorable"
            )
        # Dedup
        seen: set[str] = set()
        cleaned: list[str] = []
        for line in base:
            if line not in seen:
                seen.add(line)
                cleaned.append(line)
        out[qid] = cleaned
    return out
