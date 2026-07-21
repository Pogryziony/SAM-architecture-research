"""Stage 5 conflict policy tests."""

from __future__ import annotations

from nexus.reasoning.conflict_policy import (
    ConflictClass,
    apply_conflict_policy,
    classify_graph_conflicts,
)


def test_unresolved_contradiction_blocks_unconditional_answer():
    conflicts = classify_graph_conflicts(
        contradicts=[("A", "contradicts", "B")],
    )
    decision = apply_conflict_policy(conflicts, base_recommendation="answer")
    assert decision.allow_unconditional_answer is False
    assert decision.recommendation == "conditional_answer"
    assert decision.conflicts[0].conflict_class == ConflictClass.CONTRADICTION


def test_resolved_or_empty_allows_base_answer():
    decision = apply_conflict_policy([], base_recommendation="answer")
    assert decision.allow_unconditional_answer is True
    assert decision.recommendation == "answer"
