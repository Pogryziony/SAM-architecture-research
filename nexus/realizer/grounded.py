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
    """Measure support against evidence while fail-closing numeric claims."""
    answer = str(answer).strip()
    if not answer:
        return 0.0
    values = list(candidates)
    if not values:
        return 0.0
    evidence_text = " ".join(item.text for item in values)
    answer_numbers = set(_NUMBER.findall(answer))
    evidence_numbers = set(_NUMBER.findall(evidence_text))
    if not answer_numbers <= evidence_numbers:
        return 0.0
    # The current Realizer dataset is extractive.  Do not accept a fluent
    # continuation that appends unsupported entities or claims to an otherwise
    # correct answer.  A small allowance keeps punctuation/casing differences
    # harmless while failing closed on materially new content.
    answer_tokens = _tokens(answer)
    evidence_tokens = _tokens(evidence_text)
    if answer_tokens and len(answer_tokens - evidence_tokens) / len(answer_tokens) > 0.10:
        return 0.0
    return max(
        0.65 * token_f1(answer, item.text)
        + 0.35 * answer_similarity(answer, item.text)
        for item in values
    )


def _readable(text: str) -> bool:
    text = str(text).strip()
    if len(text) < 3 or "\ufffd" in text or not any(char.isalpha() for char in text):
        return False
    printable = sum(char.isalnum() or char.isspace() or char in ".,:;!?%_-/()[]\"'" for char in text)
    return printable / len(text) >= 0.9


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
        neural_score = grounding_score(neural_answer, candidates)
        if not _readable(neural_answer):
            rejection = "neural_answer_unreadable"
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

    selected = candidates[0]
    return GroundedRealization(
        answer=selected.text, strategy="evidence_copy",
        fallback_used=True, grounding_score=1.0,
        neural_grounding_score=neural_score,
        candidate_count=len(candidates), evidence_source=selected.source,
        rejection_reason=rejection,
    )


__all__ = [
    "EvidenceCandidate", "GroundedRealization", "answer_similarity",
    "clean_evidence_text", "evidence_candidates", "grounding_score",
    "realize_grounded", "token_f1",
]
