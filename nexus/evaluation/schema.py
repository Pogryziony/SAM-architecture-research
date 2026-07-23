"""Schema-versioned per-question evaluation records.

Every evaluated question must produce exactly one terminal outcome. Aggregate
metrics are regenerated from per-question records; incomplete records fail
validation (see ``validate_result_artifact``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


RESULT_SCHEMA_VERSION = "nexus-eval-result-v1"
LEGACY_SCHEMA_VERSIONS = frozenset(
    {
        "nexus-architecture-validation-v1",
        "nexus-oracle-vs-predicted-v2",
        "nexus-stage2-v1",
    }
)


class TerminalOutcome(str, Enum):
    """Exactly one terminal state per evaluated question.

    Phase-2 names are authoritative. ``ERROR`` remains accepted as a legacy
    synonym of ``FAILED`` for Phase-1 fixtures.
    """

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    NOT_RUN = "not_run"
    INVALID_INPUT = "invalid_input"
    # Legacy Phase-1 synonym (maps to FAILED in normalization helpers).
    ERROR = "error"
    UNSCORABLE_LEGACY = "unscorable_legacy"


def normalize_terminal_outcome(value: TerminalOutcome | str) -> TerminalOutcome:
    """Normalize outcome labels; map legacy ``error`` → ``failed``."""
    outcome = (
        value if isinstance(value, TerminalOutcome) else TerminalOutcome(str(value))
    )
    if outcome is TerminalOutcome.ERROR:
        return TerminalOutcome.FAILED
    return outcome


@dataclass
class QuestionOutcome:
    """Complete machine-readable record for one question under one system arm."""

    question_id: str
    domain: str
    question_type: str
    dataset_id: str
    dataset_sha256: str
    system_id: str
    profile: str
    config_hash: str
    config_identity_schema: str
    model_id: str
    checkpoint_id: str
    source_commit: str
    executed_at_utc: str
    terminal_outcome: TerminalOutcome
    question: str = ""
    final_answer: str = ""
    citations: list[str] = field(default_factory=list)
    retrieved_candidates: list[str] = field(default_factory=list)
    retrieved_documents: list[str] = field(default_factory=list)
    graph_paths: list[dict[str, Any]] = field(default_factory=list)
    structured_evidence: dict[str, Any] = field(default_factory=dict)
    abstention: bool = False
    verifier: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    latency_ms: float | None = None
    peak_rss_mb: float | None = None
    token_cost: dict[str, Any] = field(default_factory=dict)
    execution_environment: dict[str, Any] = field(default_factory=dict)
    # Phase-2 identity / diagnostics (optional; empty defaults preserve v1)
    run_id: str = ""
    system_version: str = ""
    domain_pack_id: str = ""
    domain_pack_version: str = ""
    graph_snapshot_id: str = ""
    resolver_identity: dict[str, Any] = field(default_factory=dict)
    resolved_entities: list[dict[str, Any]] = field(default_factory=list)
    entry_nodes: list[str] = field(default_factory=list)
    traversal_stats: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    reasoning_audit: dict[str, Any] = field(default_factory=dict)
    failure_category: str = ""
    exception_class: str = ""
    diagnostic_message: str = ""
    comparison_mode: str = ""  # system_level | controlled | ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terminal_outcome"] = normalize_terminal_outcome(
            self.terminal_outcome
        ).value
        return payload


def empty_metric_applicability() -> dict[str, dict[str, Any]]:
    """Default metric slots; every metric declares applicability and value."""
    names = (
        "grounded_correct",
        "answer_correctness",
        "unsupported_material_claim",
        "citation_correctness",
        "citation_completeness",
        "retrieval_recall",
        "evidence_precision",
        "path_validity",
        "entry_recall",
        "proof_validity",
        "abstention_precision",
        "abstention_recall",
        "fact_fuzzy_accuracy",
    )
    return {
        name: {
            "applicable": False,
            "value": None,
            "numerator": None,
            "denominator": None,
            "reason": "not_computed",
        }
        for name in names
    }


def build_question_record(
    *,
    question_id: str,
    domain: str,
    question_type: str,
    dataset_id: str,
    dataset_sha256: str,
    system_id: str,
    profile: str,
    config_hash: str,
    config_identity_schema: str,
    model_id: str,
    checkpoint_id: str,
    source_commit: str,
    executed_at_utc: str,
    terminal_outcome: TerminalOutcome | str,
    question: str = "",
    final_answer: str = "",
    citations: list[str] | None = None,
    retrieved_candidates: list[str] | None = None,
    retrieved_documents: list[str] | None = None,
    graph_paths: list[dict[str, Any]] | None = None,
    structured_evidence: dict[str, Any] | None = None,
    abstention: bool = False,
    verifier: dict[str, Any] | None = None,
    failure_reason: str = "",
    metrics: dict[str, dict[str, Any]] | None = None,
    latency_ms: float | None = None,
    peak_rss_mb: float | None = None,
    token_cost: dict[str, Any] | None = None,
    execution_environment: dict[str, Any] | None = None,
    run_id: str = "",
    system_version: str = "",
    domain_pack_id: str = "",
    domain_pack_version: str = "",
    graph_snapshot_id: str = "",
    resolver_identity: dict[str, Any] | None = None,
    resolved_entities: list[dict[str, Any]] | None = None,
    entry_nodes: list[str] | None = None,
    traversal_stats: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    reasoning_audit: dict[str, Any] | None = None,
    failure_category: str = "",
    exception_class: str = "",
    diagnostic_message: str = "",
    comparison_mode: str = "",
) -> QuestionOutcome:
    """Construct a schema-complete per-question record."""
    return QuestionOutcome(
        question_id=question_id,
        domain=domain,
        question_type=question_type,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        system_id=system_id,
        profile=profile,
        config_hash=config_hash,
        config_identity_schema=config_identity_schema,
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        source_commit=source_commit,
        executed_at_utc=executed_at_utc,
        terminal_outcome=normalize_terminal_outcome(terminal_outcome),
        question=question,
        final_answer=final_answer,
        citations=list(citations or []),
        retrieved_candidates=list(retrieved_candidates or []),
        retrieved_documents=list(retrieved_documents or []),
        graph_paths=list(graph_paths or []),
        structured_evidence=dict(structured_evidence or {}),
        abstention=abstention,
        verifier=dict(verifier or {}),
        failure_reason=failure_reason,
        metrics=metrics if metrics is not None else empty_metric_applicability(),
        latency_ms=latency_ms,
        peak_rss_mb=peak_rss_mb,
        token_cost=dict(token_cost or {}),
        execution_environment=dict(execution_environment or {}),
        run_id=run_id,
        system_version=system_version,
        domain_pack_id=domain_pack_id,
        domain_pack_version=domain_pack_version,
        graph_snapshot_id=graph_snapshot_id,
        resolver_identity=dict(resolver_identity or {}),
        resolved_entities=list(resolved_entities or []),
        entry_nodes=list(entry_nodes or []),
        traversal_stats=dict(traversal_stats or {}),
        provenance=dict(provenance or {}),
        reasoning_audit=dict(reasoning_audit or {}),
        failure_category=failure_category,
        exception_class=exception_class,
        diagnostic_message=diagnostic_message,
        comparison_mode=comparison_mode,
    )
