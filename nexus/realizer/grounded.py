"""Fail-closed grounded realization from structured NEXUS evidence.

The distillation target is a language rendering of evidence, not hidden
knowledge.  When the evidence already contains a complete answer, copying the
best supported fact is both more accurate and more faithful to NEXUS than
asking an autoregressive model to recreate the same bytes.  A neural answer is
used only when it remains strongly supported; otherwise this module falls back
to the best evidence sentence and records that decision.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


_CLAIM_PREFIX = re.compile(
    r"^(?:TrainClaim|Claim)_[0-9A-Fa-f]+:\s*", re.IGNORECASE
)
_TOKEN = re.compile(r"[A-Za-z0-9_./%+-]+")
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?%?")


@dataclass(frozen=True)
class EvidenceCandidate:
    text: str
    source: str
    kind: str
    confidence: float
    question_overlap: float


@dataclass(frozen=True)
class GroundedRealization:
    answer: str
    strategy: str
    fallback_used: bool
    grounding_score: float
    neural_grounding_score: float | None
    candidate_count: int
    evidence_source: str
    rejection_reason: str
    selected_candidate_id: str = ""
    selected_candidate_kind: str = ""
    selection_score: float = 0.0
    selection_margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundingDiagnostics:
    score: float
    continuous_support_score: float
    best_token_f1: float
    best_similarity: float
    unsupported_numbers: tuple[str, ...]
    unsupported_tokens: tuple[str, ...]
    unsupported_identifiers: tuple[str, ...]
    readable: bool
    rejection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_evidence_text(text: str) -> str:
    """Remove graph-internal claim IDs without rewriting the evidence."""
    return _CLAIM_PREFIX.sub("", str(text).strip()).strip()


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(text) if len(token) > 1}


def _overlap(question: str, text: str) -> float:
    question_tokens = _tokens(question)
    if not question_tokens:
        return 0.0
    return len(question_tokens & _tokens(text)) / len(question_tokens)


def _append_candidate(
    output: list[EvidenceCandidate],
    seen: set[str],
    *,
    text: Any,
    source: Any,
    kind: str,
    confidence: Any,
    question: str,
) -> None:
    cleaned = clean_evidence_text(str(text or ""))
    key = cleaned.casefold()
    if not cleaned or key in seen:
        return
    seen.add(key)
    try:
        numeric_confidence = float(confidence)
    except (TypeError, ValueError):
        numeric_confidence = 0.5
    output.append(EvidenceCandidate(
        text=cleaned,
        source=str(source or ""),
        kind=kind,
        confidence=max(0.0, min(1.0, numeric_confidence)),
        question_overlap=_overlap(question, cleaned),
    ))


def evidence_candidates(record: dict[str, Any]) -> list[EvidenceCandidate]:
    """Return deduplicated answer candidates ranked without using labels."""
    evidence = record.get("evidence_pack", {})
    if not isinstance(evidence, dict):
        return []
    question = str(record.get("question") or evidence.get("question") or "")
    candidates: list[EvidenceCandidate] = []
    seen: set[str] = set()

    for item in evidence.get("node_facts", []):
        if isinstance(item, dict):
            _append_candidate(
                candidates, seen, text=item.get("text"),
                source=item.get("source"), kind="node_fact",
                confidence=item.get("confidence", 1.0), question=question,
            )
    for item in evidence.get("snippets", []):
        if isinstance(item, dict):
            _append_candidate(
                candidates, seen, text=item.get("text"),
                source=item.get("source"), kind="snippet",
                confidence=item.get("confidence", 0.9), question=question,
            )
    for path in evidence.get("paths", []):
        if not isinstance(path, dict):
            continue
        for node in path.get("nodes", []):
            if not isinstance(node, dict):
                continue
            _append_candidate(
                candidates, seen,
                text=node.get("key_finding") or node.get("description"),
                source=(node.get("sources") or [""])[0], kind="path_node",
                confidence=path.get("score", 0.7), question=question,
            )
    for fact in evidence.get("facts", []):
        _append_candidate(
            candidates, seen, text=fact, source="", kind="fact",
            confidence=0.6, question=question,
        )

    kind_priority = {"node_fact": 4, "snippet": 3, "path_node": 2, "fact": 1}
    return sorted(
        candidates,
        key=lambda item: (
            kind_priority.get(item.kind, 0),
            item.question_overlap,
            item.confidence,
            len(item.text),
            item.text,
        ),
        reverse=True,
    )


def answer_similarity(answer: str, reference: str) -> float:
    """Stable character similarity for diagnostics and validation scoring."""
    left = " ".join(str(answer).casefold().split())
    right = " ".join(str(reference).casefold().split())
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def token_f1(answer: str, reference: str) -> float:
    left = _tokens(answer)
    right = _tokens(reference)
    if not left or not right:
        return 0.0
    common = len(left & right)
    precision = common / len(left)
    recall = common / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def grounding_score(answer: str, candidates: Iterable[EvidenceCandidate]) -> float:
    """Measure final fail-closed support against evidence."""
    return grounding_diagnostics(answer, candidates).score


def _readable(text: str) -> bool:
    text = str(text).strip()
    if len(text) < 3 or "\ufffd" in text or not any(char.isalpha() for char in text):
        return False
    printable = sum(char.isalnum() or char.isspace() or char in ".,:;!?%_-/()[]\"'" for char in text)
    return printable / len(text) >= 0.9


def grounding_diagnostics(
    answer: str, candidates: Iterable[EvidenceCandidate]
) -> GroundingDiagnostics:
    """Explain grounding failure without collapsing every failure to zero."""
    answer = str(answer).strip()
    values = list(candidates)
    readable = _readable(answer)
    if not answer or not values:
        reason = "empty_answer" if not answer else "no_evidence_candidate"
        return GroundingDiagnostics(
            score=0.0, continuous_support_score=0.0,
            best_token_f1=0.0, best_similarity=0.0,
            unsupported_numbers=(), unsupported_tokens=(),
            unsupported_identifiers=(), readable=readable,
            rejection_reason=reason,
        )

    best_f1 = max(token_f1(answer, item.text) for item in values)
    best_similarity = max(answer_similarity(answer, item.text) for item in values)
    continuous = max(
        0.65 * token_f1(answer, item.text)
        + 0.35 * answer_similarity(answer, item.text)
        for item in values
    )
    evidence_text = " ".join(item.text for item in values)
    answer_numbers = set(_NUMBER.findall(answer))
    evidence_numbers = set(_NUMBER.findall(evidence_text))
    unsupported_numbers = tuple(sorted(answer_numbers - evidence_numbers))
    answer_tokens = _tokens(answer)
    evidence_tokens = _tokens(evidence_text)
    unsupported_tokens = tuple(sorted(answer_tokens - evidence_tokens))
    unsupported_identifiers = tuple(
        token for token in unsupported_tokens
        if any(marker in token for marker in ("/", ".", "_"))
    )
    unsupported_ratio = (
        len(unsupported_tokens) / len(answer_tokens) if answer_tokens else 0.0
    )
    if not readable:
        reason = "neural_answer_unreadable"
    elif unsupported_numbers:
        reason = "unsupported_number"
    elif unsupported_identifiers:
        reason = "unsupported_identifier"
    elif unsupported_ratio > 0.10:
        reason = "unsupported_token_ratio"
    else:
        reason = ""
    score = 0.0 if reason else continuous
    return GroundingDiagnostics(
        score=score,
        continuous_support_score=continuous,
        best_token_f1=best_f1,
        best_similarity=best_similarity,
        unsupported_numbers=unsupported_numbers,
        unsupported_tokens=unsupported_tokens,
        unsupported_identifiers=unsupported_identifiers,
        readable=readable,
        rejection_reason=reason,
    )


def realize_grounded(
    record: dict[str, Any],
    neural_answer: str | None = None,
    *,
    neural_support_threshold: float = 0.72,
) -> GroundedRealization:
    """Use a supported neural answer or fall back to evidence extraction."""
    candidates = evidence_candidates(record)
    if not candidates:
        return GroundedRealization(
            answer="Insufficient evidence to answer.", strategy="insufficient_evidence",
            fallback_used=True, grounding_score=0.0,
            neural_grounding_score=None, candidate_count=0,
            evidence_source="", rejection_reason="no_evidence_candidate",
        )

    neural_score = None
    rejection = "neural_answer_missing"
    if neural_answer is not None:
        diagnostics = grounding_diagnostics(neural_answer, candidates)
        neural_score = diagnostics.score
        if diagnostics.rejection_reason:
            rejection = diagnostics.rejection_reason
        elif neural_score < neural_support_threshold:
            rejection = "neural_answer_not_grounded"
        else:
            return GroundedRealization(
                answer=neural_answer.strip(), strategy="neural_grounded",
                fallback_used=False, grounding_score=neural_score,
                neural_grounding_score=neural_score,
                candidate_count=len(candidates), evidence_source="multiple",
                rejection_reason="",
            )

    # Import lazily to keep candidate extraction independent of the selector.
    from .pointer_copy import realize_pointer_copy

    pointer = realize_pointer_copy(record)
    return GroundedRealization(
        answer=pointer.answer,
        strategy=(
            "evidence_copy"
            if pointer.strategy == "pointer_copy"
            else "insufficient_evidence"
        ),
        fallback_used=True, grounding_score=pointer.grounding_score,
        neural_grounding_score=neural_score,
        candidate_count=len(candidates), evidence_source=pointer.evidence_source,
        rejection_reason=(pointer.rejection_reason or rejection),
        selected_candidate_id=pointer.selected_candidate_id,
        selected_candidate_kind=pointer.selected_candidate_kind,
        selection_score=pointer.selection_score,
        selection_margin=pointer.selection_margin,
    )


__all__ = [
    "EvidenceCandidate", "GroundedRealization", "GroundingDiagnostics",
    "answer_similarity", "clean_evidence_text", "evidence_candidates",
    "grounding_diagnostics", "grounding_score",
    "realize_grounded", "token_f1",
]
