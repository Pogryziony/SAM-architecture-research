"""Deterministic Pointer/Copy Realizer for extractive NEXUS evidence.

The current Realizer dataset contains complete answer-bearing evidence.  This
module selects one candidate using question/evidence features and copies its
text verbatim.  It never regenerates paths, identifiers, numbers, or config
values token by token.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from .grounded import EvidenceCandidate, evidence_candidates


_TOKEN = re.compile(r"[A-Za-z0-9_./%+-]+")
_KIND_BASE = {"node_fact": 4.0, "snippet": 3.0, "path_node": 2.0, "fact": 1.0}


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(str(text)) if len(token) > 1}


def candidate_id(candidate: EvidenceCandidate) -> str:
    payload = "\x1f".join((candidate.kind, candidate.source, candidate.text))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def candidate_selection_score(question: str, candidate: EvidenceCandidate) -> float:
    """Score a candidate without labels or list-position features."""
    question_tokens = _tokens(question)
    source_tokens = _tokens(candidate.source)
    source_overlap = (
        len(question_tokens & source_tokens) / len(source_tokens)
        if source_tokens else 0.0
    )
    source_exact = (
        0.5
        if candidate.source
        and candidate.source.casefold() in str(question).casefold()
        else 0.0
    )
    return (
        _KIND_BASE.get(candidate.kind, 0.0)
        + 2.0 * candidate.question_overlap
        + candidate.confidence
        + source_overlap
        + source_exact
    )


@dataclass(frozen=True)
class PointerCopyConfig:
    minimum_score: float = 1.0
    minimum_margin: float = 0.25


def pointer_copy_config_from_dict(payload: dict[str, Any]) -> PointerCopyConfig:
    if payload.get("schema_version") != "nexus-pointer-copy-realizer-v3":
        raise ValueError("unsupported Pointer/Copy config schema")
    if payload.get("score_version") != "question_evidence_v1":
        raise ValueError("unsupported Pointer/Copy score version")
    if payload.get("candidate_ordering") != "score_then_confidence_then_candidate_id":
        raise ValueError("unsupported Pointer/Copy candidate ordering")
    config = PointerCopyConfig(
        minimum_score=float(payload["minimum_score"]),
        minimum_margin=float(payload["minimum_margin"]),
    )
    if config.minimum_score < 0 or config.minimum_margin < 0:
        raise ValueError("Pointer/Copy thresholds must be non-negative")
    return config


@dataclass(frozen=True)
class PointerCopyResult:
    answer: str
    strategy: str
    selected_candidate_id: str
    selected_candidate_kind: str
    evidence_source: str
    selection_score: float
    selection_margin: float
    candidate_count: int
    fallback_used: bool
    rejection_reason: str
    grounding_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def realize_pointer_copy(
    record: dict[str, Any],
    *,
    config: PointerCopyConfig | None = None,
) -> PointerCopyResult:
    """Select and copy one supported candidate, failing closed on ambiguity."""
    config = config or PointerCopyConfig()
    question = str(record.get("question") or "")
    ranked = sorted(
        evidence_candidates(record),
        key=lambda candidate: (
            candidate_selection_score(question, candidate),
            candidate.confidence,
            candidate_id(candidate),
        ),
        reverse=True,
    )
    if not ranked:
        return PointerCopyResult(
            answer="Insufficient evidence to answer.",
            strategy="insufficient_evidence",
            selected_candidate_id="",
            selected_candidate_kind="",
            evidence_source="",
            selection_score=0.0,
            selection_margin=0.0,
            candidate_count=0,
            fallback_used=True,
            rejection_reason="no_evidence_candidate",
            grounding_score=0.0,
        )

    selected = ranked[0]
    score = candidate_selection_score(question, selected)
    runner_up_score = (
        candidate_selection_score(question, ranked[1]) if len(ranked) > 1 else 0.0
    )
    margin = score - runner_up_score
    if score < config.minimum_score:
        rejection = "selection_score_below_threshold"
    elif len(ranked) > 1 and margin < config.minimum_margin:
        rejection = "ambiguous_evidence_candidates"
    else:
        rejection = ""

    if rejection:
        return PointerCopyResult(
            answer="Insufficient evidence to answer.",
            strategy="insufficient_evidence",
            selected_candidate_id="",
            selected_candidate_kind="",
            evidence_source="",
            selection_score=round(score, 6),
            selection_margin=round(margin, 6),
            candidate_count=len(ranked),
            fallback_used=True,
            rejection_reason=rejection,
            grounding_score=0.0,
        )

    return PointerCopyResult(
        answer=selected.text,
        strategy="pointer_copy",
        selected_candidate_id=candidate_id(selected),
        selected_candidate_kind=selected.kind,
        evidence_source=selected.source,
        selection_score=round(score, 6),
        selection_margin=round(margin, 6),
        candidate_count=len(ranked),
        fallback_used=False,
        rejection_reason="",
        grounding_score=1.0,
    )


__all__ = [
    "PointerCopyConfig",
    "PointerCopyResult",
    "candidate_id",
    "candidate_selection_score",
    "pointer_copy_config_from_dict",
    "realize_pointer_copy",
]
