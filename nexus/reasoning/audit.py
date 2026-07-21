"""Deterministic reasoning audit for NEXUS answers.

The audit is deliberately model-free.  It turns the paths and evidence already
selected by NEXUS into a replayable proof trace, surfaces contradictory graph
edges, and computes a transparent readiness diagnostic.  It does not rewrite
or suppress answers; callers decide how to use the recommendation.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from nexus.graph import Path, PathStep
from nexus.graph.store import InMemoryGraphStore


_INSUFFICIENT_MARKERS = (
    "insufficient evidence",
    "not enough evidence",
    "cannot answer from the evidence",
    "no evidence",
    "unable to determine",
)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _structured_locators(node: Any) -> list[str]:
    """Extract non-empty locators from ``properties['provenance']`` records."""
    raw = (getattr(node, "properties", None) or {}).get("provenance")
    if not isinstance(raw, list):
        return []
    locators: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            locator = str(item.get("locator", "")).strip()
            if locator:
                locators.append(locator)
        else:
            locator = str(getattr(item, "locator", "")).strip()
            if locator:
                locators.append(locator)
    return locators


def _step_sources(step: PathStep, graph: InMemoryGraphStore) -> list[str]:
    sources: set[str] = set()
    if step.edge.evidence:
        sources.add(step.edge.evidence)
    for node_id in (step.edge.source, step.edge.target):
        node = graph.get_node(node_id)
        if node:
            sources.update(source for source in node.sources if source)
            sources.update(_structured_locators(node))
    return sorted(sources)


def _step_has_structured_provenance(step: PathStep, graph: InMemoryGraphStore) -> bool:
    """True when both endpoints carry at least one structured locator."""
    for node_id in (step.edge.source, step.edge.target):
        node = graph.get_node(node_id)
        if node is None or not _structured_locators(node):
            return False
    return True


@dataclass(frozen=True)
class ProofStep:
    """One replayable graph step supporting an answer."""

    step_id: str
    from_node: str
    relation: str
    to_node: str
    confidence: float
    reversed: bool
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = list(self.sources)
        return data


@dataclass(frozen=True)
class CounterEvidence:
    """A graph edge that challenges at least one node in the proof."""

    counter_id: str
    source: str
    relation: str
    target: str
    confidence: float
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = list(self.sources)
        return data


@dataclass
class ReasoningAudit:
    """Serializable audit record attached to a canonical NEXUS result."""

    proof_steps: list[ProofStep] = field(default_factory=list)
    counter_evidence: list[CounterEvidence] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    readiness_score: float = 0.0
    recommended_action: str = "abstain"
    proof_valid: bool = False
    provenance_coverage: float = 0.0
    structured_provenance_coverage: float = 0.0
    traversal_truncated: bool = False
    traversal_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_steps": [step.to_dict() for step in self.proof_steps],
            "counter_evidence": [item.to_dict() for item in self.counter_evidence],
            "components": dict(self.components),
            "readiness_score": self.readiness_score,
            "recommended_action": self.recommended_action,
            "proof_valid": self.proof_valid,
            "provenance_coverage": self.provenance_coverage,
            "structured_provenance_coverage": self.structured_provenance_coverage,
            "traversal_truncated": self.traversal_truncated,
            "traversal_stats": dict(self.traversal_stats),
            "errors": list(self.errors),
        }


def _build_proof_steps(
    paths: Sequence[Path], graph: InMemoryGraphStore,
) -> tuple[list[ProofStep], list[str]]:
    proof_steps: list[ProofStep] = []
    errors: list[str] = []
    seen: set[tuple[str, str, str, bool]] = set()

    for path in paths:
        for step in path.steps:
            key = (
                step.edge.source,
                step.edge.type,
                step.edge.target,
                step.reversed,
            )
            if key in seen:
                continue
            seen.add(key)

            if not graph.has_node(step.edge.source):
                errors.append(f"missing proof source node: {step.edge.source}")
            if not graph.has_node(step.edge.target):
                errors.append(f"missing proof target node: {step.edge.target}")
            stored = any(
                edge.type == step.edge.type and edge.target == step.edge.target
                for edge in graph.get_outgoing(step.edge.source)
            )
            if not stored:
                errors.append(
                    "proof edge absent from graph: "
                    f"{step.edge.source} {step.edge.type} {step.edge.target}"
                )

            proof_steps.append(ProofStep(
                step_id=_stable_id(
                    step.edge.source,
                    step.edge.type,
                    step.edge.target,
                    "reverse" if step.reversed else "forward",
                ),
                from_node=step.from_node,
                relation=step.edge.type,
                to_node=step.to_node,
                confidence=round(_clip(step.edge.confidence), 4),
                reversed=step.reversed,
                sources=tuple(_step_sources(step, graph)),
            ))

    return proof_steps, errors


def _find_counter_evidence(
    proof_steps: Sequence[ProofStep],
    evidence_pack: dict[str, Any],
    graph: InMemoryGraphStore,
) -> list[CounterEvidence]:
    evidence_nodes = {
        node_id
        for step in proof_steps
        for node_id in (step.from_node, step.to_node)
    }
    for path in evidence_pack.get("paths", []):
        for node in path.get("nodes", []):
            node_id = node.get("id")
            if node_id:
                evidence_nodes.add(str(node_id))

    counters: list[CounterEvidence] = []
    seen: set[tuple[str, str, str]] = set()
    for node_id in sorted(evidence_nodes):
        for edge in graph.get_edges(node_id, "both"):
            if edge.type != "contradicts":
                continue
            key = (edge.source, edge.type, edge.target)
            if key in seen:
                continue
            seen.add(key)
            synthetic_step = PathStep(edge=edge)
            counters.append(CounterEvidence(
                counter_id=_stable_id(*key),
                source=edge.source,
                relation=edge.type,
                target=edge.target,
                confidence=round(_clip(edge.confidence), 4),
                sources=tuple(_step_sources(synthetic_step, graph)),
            ))
    counters.sort(key=lambda item: (-item.confidence, item.counter_id))
    return counters


def build_reasoning_audit(
    paths: Sequence[Path],
    graph: InMemoryGraphStore,
    evidence_pack: dict[str, Any],
    verification: Any,
    answer: str,
    *,
    answer_threshold: float = 0.70,
    conditional_threshold: float = 0.40,
    require_structured_provenance: bool = False,
    traversal_truncated: bool = False,
    traversal_stats: dict[str, Any] | None = None,
) -> ReasoningAudit:
    """Build a deterministic, JSON-safe explanation of answer readiness.

    Thresholds are decision-policy inputs, not learned parameters.  The score
    is diagnostic until calibrated on a frozen NEXUS oracle benchmark.
    Truncated traversal and incomplete provenance never yield an unconditional
    ``answer`` recommendation.
    """
    proof_steps, errors = _build_proof_steps(paths, graph)
    counter_evidence = _find_counter_evidence(proof_steps, evidence_pack, graph)
    node_facts = [
        fact for fact in evidence_pack.get("node_facts", [])
        if isinstance(fact, dict)
    ]

    confidence_values = [step.confidence for step in proof_steps]
    confidence_values.extend(
        _clip(fact.get("confidence", 0.0))
        for fact in node_facts
        if isinstance(fact.get("confidence", 0.0), (int, float))
    )
    evidence_quality = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0.0
    )

    provenance_items = [bool(step.sources) for step in proof_steps]
    provenance_items.extend(bool(fact.get("source")) for fact in node_facts)
    provenance_coverage = (
        sum(provenance_items) / len(provenance_items)
        if provenance_items else 0.0
    )

    structured_flags = [
        _step_has_structured_provenance(step, graph)
        for path in paths
        for step in path.steps
    ]
    structured_provenance_coverage = (
        sum(structured_flags) / len(structured_flags)
        if structured_flags else 0.0
    )

    hallucination_rate = _clip(
        getattr(verification, "hallucination_rate", 1.0)
        if verification is not None else 1.0
    )
    verification_support = 1.0 - hallucination_rate
    path_relevance = max(
        (_clip(path.score) for path in paths if path.steps),
        default=0.0,
    )
    opposition_clarity = 1.0 / (1.0 + len(counter_evidence))

    components = {
        "evidence_quality": round(evidence_quality, 4),
        "provenance_coverage": round(provenance_coverage, 4),
        "structured_provenance_coverage": round(structured_provenance_coverage, 4),
        "verification_support": round(verification_support, 4),
        "path_relevance": round(path_relevance, 4),
        "opposition_clarity": round(opposition_clarity, 4),
    }
    readiness_score = round(
        0.25 * evidence_quality
        + 0.20 * provenance_coverage
        + 0.25 * verification_support
        + 0.20 * path_relevance
        + 0.10 * opposition_clarity,
        4,
    )

    has_evidence = bool(proof_steps or node_facts or evidence_pack.get("numbers"))
    proof_valid = has_evidence and not errors
    verification_passed = bool(
        verification is not None and getattr(verification, "passed", False)
    )
    insufficient = any(marker in answer.lower() for marker in _INSUFFICIENT_MARKERS)
    provenance_incomplete = provenance_coverage < 1.0 or (
        require_structured_provenance and structured_provenance_coverage < 1.0
    )

    if insufficient or not proof_valid or not verification_passed:
        action = "abstain"
    elif readiness_score < conditional_threshold:
        action = "abstain"
    elif (counter_evidence
          or provenance_incomplete
          or traversal_truncated
          or readiness_score < answer_threshold):
        action = "conditional_answer"
    else:
        action = "answer"

    return ReasoningAudit(
        proof_steps=proof_steps,
        counter_evidence=counter_evidence,
        components=components,
        readiness_score=readiness_score,
        recommended_action=action,
        proof_valid=proof_valid,
        provenance_coverage=round(provenance_coverage, 4),
        structured_provenance_coverage=round(structured_provenance_coverage, 4),
        traversal_truncated=bool(traversal_truncated),
        traversal_stats=dict(traversal_stats or {}),
        errors=errors,
    )


__all__ = [
    "CounterEvidence",
    "ProofStep",
    "ReasoningAudit",
    "build_reasoning_audit",
]
