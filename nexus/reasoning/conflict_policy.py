"""Stage 5 contradiction / conflict policy (partial).

Classifies unresolved conflicts and blocks unconditional answers when any
unresolved conflict is present. Full contradiction F1 remains prereg-gated
(see EXPERIMENT_CONTRADICTION_POLICY_V1.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class ConflictClass(str, Enum):
    CONTRADICTION = "contradiction"
    SUPERSESSION = "supersession"
    VALIDITY_MISMATCH = "validity_mismatch"
    SOURCE_DISAGREEMENT = "source_disagreement"


@dataclass(frozen=True)
class Conflict:
    conflict_class: ConflictClass
    source: str
    relation: str
    target: str
    resolved: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_class": self.conflict_class.value,
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "resolved": self.resolved,
            "note": self.note,
        }


@dataclass(frozen=True)
class ConflictPolicyDecision:
    allow_unconditional_answer: bool
    conflicts: tuple[Conflict, ...]
    recommendation: str  # answer | conditional_answer | abstain

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_unconditional_answer": self.allow_unconditional_answer,
            "recommendation": self.recommendation,
            "conflicts": [item.to_dict() for item in self.conflicts],
        }


def classify_graph_conflicts(
    *,
    contradicts: Sequence[tuple[str, str, str]] = (),
    replaces: Sequence[tuple[str, str, str]] = (),
    validity_mismatches: Sequence[tuple[str, str, str]] = (),
    source_disagreements: Sequence[tuple[str, str, str]] = (),
) -> list[Conflict]:
    """Build conflict records from explicit graph signals."""
    conflicts: list[Conflict] = []
    for source, relation, target in contradicts:
        conflicts.append(
            Conflict(
                ConflictClass.CONTRADICTION,
                source,
                relation or "contradicts",
                target,
            )
        )
    for source, relation, target in replaces:
        conflicts.append(
            Conflict(
                ConflictClass.SUPERSESSION,
                source,
                relation or "replaces",
                target,
                resolved=False,
                note="supersession requires explicit resolution policy",
            )
        )
    for source, relation, target in validity_mismatches:
        conflicts.append(
            Conflict(ConflictClass.VALIDITY_MISMATCH, source, relation, target)
        )
    for source, relation, target in source_disagreements:
        conflicts.append(
            Conflict(ConflictClass.SOURCE_DISAGREEMENT, source, relation, target)
        )
    return conflicts


def apply_conflict_policy(
    conflicts: Sequence[Conflict],
    *,
    base_recommendation: str = "answer",
) -> ConflictPolicyDecision:
    """Unresolved conflicts never yield an unconditional answer."""
    unresolved = [item for item in conflicts if not item.resolved]
    if not unresolved:
        return ConflictPolicyDecision(
            allow_unconditional_answer=base_recommendation == "answer",
            conflicts=tuple(conflicts),
            recommendation=base_recommendation,
        )
    recommendation = base_recommendation
    if recommendation == "answer":
        recommendation = "conditional_answer"
    return ConflictPolicyDecision(
        allow_unconditional_answer=False,
        conflicts=tuple(conflicts),
        recommendation=recommendation,
    )
