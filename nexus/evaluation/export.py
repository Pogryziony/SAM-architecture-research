"""Export NEXUSRunner QuestionResult records into eval-result-v1."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from nexus.evaluation.metrics import compute_grounded_correct
from nexus.evaluation.schema import (
    RESULT_SCHEMA_VERSION,
    TerminalOutcome,
    build_question_record,
    empty_metric_applicability,
    normalize_terminal_outcome,
)
from nexus.evaluation.validate import assert_valid_result_artifact
from nexus.pipeline.config import CONFIG_IDENTITY_SCHEMA, ProductionNEXUSConfig
from nexus.pipeline.runner import PipelineResult, QuestionResult


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def classify_terminal_outcome(qr: QuestionResult) -> TerminalOutcome:
    """Map runner diagnostics to a mutually exclusive terminal outcome."""
    category = (qr.failure_category or "").strip()
    answer = (qr.answer or "").strip()
    abstain_text = (not answer) or ("insufficient" in answer.casefold())

    if category.startswith("exception:"):
        return TerminalOutcome.FAILED
    if category == "timed_out":
        return TerminalOutcome.TIMED_OUT
    if category == "invalid_input":
        return TerminalOutcome.INVALID_INPUT
    if category == "not_run":
        return TerminalOutcome.NOT_RUN
    if category == "no_entities_resolved" and not answer:
        return TerminalOutcome.FAILED
    if category == "no_graph_paths" and abstain_text and not qr.evidence_pack:
        return TerminalOutcome.FAILED
    if abstain_text or category == "insufficient_answer":
        return TerminalOutcome.ABSTAINED
    if answer:
        return TerminalOutcome.ANSWERED
    return TerminalOutcome.FAILED


def classify_failure_category(qr: QuestionResult) -> str:
    """Derive failure_category with zero-hop answers not treated as path failures.

    A valid zero-hop grounded answer has entry entities and an answer (or
    structured evidence) even when ``graph_paths_count == 0``.
    """
    if qr.failure_category.startswith("exception:"):
        return qr.failure_category

    entities = qr.selected_entry_nodes or qr.predicted_entities
    answer = (qr.answer or "").strip()
    abstain = (not answer) or ("insufficient" in answer.casefold())
    has_evidence = bool(qr.evidence_pack) or bool(qr.evidence_pack_keys)

    if not entities:
        return "no_entities_resolved"
    if abstain:
        return "insufficient_answer"
    if not qr.verifier_passed and answer:
        return "verifier_failed"
    if qr.graph_paths_count == 0 and not has_evidence and not answer:
        return "no_graph_paths"
    # Zero-hop success: paths may be zero while answer/evidence exist.
    return ""


def question_result_to_outcome(
    qr: QuestionResult,
    *,
    config: ProductionNEXUSConfig,
    dataset_id: str,
    dataset_sha256: str,
    system_id: str,
    profile: str,
    source_commit: str,
    executed_at_utc: str,
    run_id: str,
    domain: str = "",
    question_type: str = "",
    domain_pack_id: str = "",
    domain_pack_version: str = "",
    graph_snapshot_id: str = "",
    model_id: str = "",
    checkpoint_id: str = "",
    gold: Mapping[str, Any] | None = None,
    comparison_mode: str = "",
    peak_rss_mb: float | None = None,
) -> dict[str, Any]:
    """Convert one QuestionResult into a schema-v1 per-question dict."""
    category = classify_failure_category(qr)
    qr.failure_category = category
    outcome = classify_terminal_outcome(qr)

    exception_class = ""
    diagnostic = ""
    if category.startswith("exception:"):
        exception_class = category.split(":", 1)[1]
        diagnostic = category

    latency = None
    if qr.per_stage_latency_ms:
        latency = round(sum(float(v) for v in qr.per_stage_latency_ms.values()), 3)

    metrics = empty_metric_applicability()
    gold = dict(gold or {})
    should_abstain = bool(gold.get("should_abstain", False))
    if gold.get("gold_answer") is not None or should_abstain:
        grounded = compute_grounded_correct(
            answer=qr.answer,
            gold_answer=str(gold.get("gold_answer") or ""),
            should_abstain=should_abstain,
            answer_correct=None,
            material_claims_supported=None,
            citations_entail=None,
            temporal_ok=None,
        )
        # Prefer fuzzy proxy when available
        try:
            from benchmarks.scoring import compute_fact_score

            fact = compute_fact_score(
                qr.answer, str(gold.get("gold_answer") or "")
            )
            fuzzy = fact.get("fuzzy_accuracy")
            if fuzzy is not None:
                metrics["fact_fuzzy_accuracy"] = {
                    "applicable": True,
                    "value": float(fuzzy),
                    "numerator": float(fuzzy),
                    "denominator": 1.0,
                    "reason": "key_fact_scorer",
                }
                grounded = compute_grounded_correct(
                    answer=qr.answer,
                    gold_answer=str(gold.get("gold_answer") or ""),
                    should_abstain=should_abstain,
                    answer_correct=float(fuzzy) >= 0.5,
                    material_claims_supported=None,
                    citations_entail=None,
                    temporal_ok=None,
                )
        except Exception:
            pass
        metrics["grounded_correct"] = grounded.to_dict()
        metrics["path_validity"] = {
            "applicable": True,
            "value": 1.0 if qr.proof_valid else 0.0,
            "numerator": 1.0 if qr.proof_valid else 0.0,
            "denominator": 1.0,
            "reason": "reasoning_audit.proof_valid",
        }

    citations: list[str] = []
    ep = qr.evidence_pack if isinstance(qr.evidence_pack, dict) else {}
    sources = ep.get("sources") or ep.get("Sources") or []
    if isinstance(sources, list):
        citations = [str(s) for s in sources]

    paths = [
        {"score": score, "rank": i}
        for i, score in enumerate(qr.path_scores)
    ]

    record = build_question_record(
        question_id=qr.question_id,
        domain=domain or str(gold.get("domain") or "unknown"),
        question_type=question_type or str(gold.get("question_type") or qr.parsed_intent or "unknown"),
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        system_id=system_id,
        profile=profile,
        config_hash=config.config_hash,
        config_identity_schema=getattr(
            config, "identity_schema", CONFIG_IDENTITY_SCHEMA
        ),
        model_id=model_id or "unspecified",
        checkpoint_id=checkpoint_id
        or config.realizer_checkpoint_sha256
        or "",
        source_commit=source_commit,
        executed_at_utc=executed_at_utc,
        terminal_outcome=outcome,
        question=qr.question,
        final_answer=qr.answer,
        citations=citations,
        retrieved_candidates=[
            str(c.get("entity_id", c)) if isinstance(c, dict) else str(c)
            for c in qr.resolution_candidates
        ],
        graph_paths=paths,
        structured_evidence=ep if isinstance(ep, dict) else {},
        abstention=outcome is TerminalOutcome.ABSTAINED,
        verifier={
            "passed": qr.verifier_passed,
            "hallucination_rate": qr.hallucination_rate,
        },
        failure_reason=category,
        metrics=metrics,
        latency_ms=latency,
        peak_rss_mb=peak_rss_mb,
        execution_environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cascade_level": qr.cascade_level,
            "per_stage_latency_ms": qr.per_stage_latency_ms,
            "allow_synth_fallback": bool(
                getattr(config, "allow_synth_fallback", True)
            ),
            "realizer_backend": getattr(config, "realizer_backend", ""),
            "selected_realizer": (qr.reasoning_audit or {}).get(
                "selected_realizer", getattr(config, "realizer_backend", "")
            ),
            "fallback_considered": bool(
                (qr.reasoning_audit or {}).get("fallback_considered", False)
            ),
            "fallback_permitted": bool(
                (qr.reasoning_audit or {}).get(
                    "fallback_permitted",
                    getattr(config, "allow_synth_fallback", True),
                )
            ),
            "fallback_reason": str(
                (qr.reasoning_audit or {}).get("fallback_reason") or ""
            ),
            "fallback_terminal_outcome": str(
                (qr.reasoning_audit or {}).get("fallback_terminal_outcome") or ""
            ),
        },
        run_id=run_id,
        system_version="nexus-v1",
        domain_pack_id=domain_pack_id or config.pipeline_id.domain_pack_id,
        domain_pack_version=domain_pack_version
        or config.pipeline_id.domain_pack_version,
        graph_snapshot_id=graph_snapshot_id
        or config.pipeline_id.graph_snapshot_id,
        resolver_identity={
            "name": qr.resolver_name or qr.entity_resolution_method,
            "version": qr.resolver_version,
            "threshold": qr.resolver_threshold,
            "rejection_reason": qr.resolver_rejection_reason,
            "latency_ms": qr.resolver_latency_ms,
        },
        resolved_entities=list(qr.resolution_candidates),
        entry_nodes=list(qr.selected_entry_nodes),
        traversal_stats={
            "graph_paths_count": qr.graph_paths_count,
            "path_scores": list(qr.path_scores),
            "lexical_fallback_used": qr.lexical_fallback_used,
        },
        provenance={
            "coverage": qr.provenance_coverage,
            "proof_valid": qr.proof_valid,
            "proof_steps_count": qr.proof_steps_count,
            "counter_evidence_count": qr.counter_evidence_count,
        },
        reasoning_audit=dict(qr.reasoning_audit or {}),
        failure_category=category,
        exception_class=exception_class,
        diagnostic_message=diagnostic,
        comparison_mode=comparison_mode,
    )
    return record.to_dict()


def pipeline_to_eval_artifact(
    pipeline: PipelineResult,
    *,
    config: ProductionNEXUSConfig,
    questions: Sequence[Mapping[str, Any]],
    dataset_id: str,
    dataset_sha256: str,
    system_id: str = "nexus",
    profile: str = "",
    domain_pack_id: str = "",
    domain_pack_version: str = "",
    graph_snapshot_id: str = "",
    model_id: str = "",
    checkpoint_id: str = "",
    comparison_mode: str = "",
    run_id: str | None = None,
    peak_rss_mb: float | None = None,
    status: str = "VALID",
) -> dict[str, Any]:
    """Build a schema-valid run artifact from a PipelineResult."""
    from nexus.evaluation.aggregate import aggregate_question_records

    source_commit = pipeline.source_sha or _git_head()
    executed_at = datetime.now(timezone.utc).isoformat()
    rid = run_id or uuid.uuid4().hex[:12]
    profile_name = profile or config.realizer_backend
    by_id = {str(q.get("id", "")): q for q in questions}

    rows: list[dict[str, Any]] = []
    for qr in pipeline.per_question:
        gold = by_id.get(qr.question_id, {})
        rows.append(
            question_result_to_outcome(
                qr,
                config=config,
                dataset_id=dataset_id,
                dataset_sha256=dataset_sha256,
                system_id=system_id,
                profile=profile_name,
                source_commit=source_commit,
                executed_at_utc=executed_at,
                run_id=rid,
                domain=str(gold.get("domain") or ""),
                question_type=str(gold.get("question_type") or ""),
                domain_pack_id=domain_pack_id,
                domain_pack_version=domain_pack_version,
                graph_snapshot_id=graph_snapshot_id,
                model_id=model_id,
                checkpoint_id=checkpoint_id,
                gold=gold,
                comparison_mode=comparison_mode,
                peak_rss_mb=peak_rss_mb,
            )
        )

    # Config validation errors → INVALID_INPUT / FAILED rows when no per-q results
    if pipeline.errors and not rows:
        for q in questions:
            rows.append(
                build_question_record(
                    question_id=str(q.get("id", "")),
                    domain=str(q.get("domain") or "unknown"),
                    question_type=str(q.get("question_type") or "unknown"),
                    dataset_id=dataset_id,
                    dataset_sha256=dataset_sha256,
                    system_id=system_id,
                    profile=profile_name,
                    config_hash=config.config_hash,
                    config_identity_schema=CONFIG_IDENTITY_SCHEMA,
                    model_id=model_id or "unspecified",
                    checkpoint_id=checkpoint_id,
                    source_commit=source_commit,
                    executed_at_utc=executed_at,
                    terminal_outcome=TerminalOutcome.FAILED,
                    question=str(q.get("question") or ""),
                    failure_reason="; ".join(pipeline.errors),
                    failure_category="config_error",
                    diagnostic_message="; ".join(pipeline.errors),
                    run_id=rid,
                    comparison_mode=comparison_mode,
                ).to_dict()
            )

    aggregates = aggregate_question_records(rows)
    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": executed_at,
        "run_id": rid,
        "source_commit": source_commit,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "system_id": system_id,
        "profile": profile_name,
        "config_hash": config.config_hash,
        "config_identity_schema": CONFIG_IDENTITY_SCHEMA,
        "comparison_mode": comparison_mode,
        "domain_pack_id": domain_pack_id or config.pipeline_id.domain_pack_id,
        "domain_pack_version": domain_pack_version
        or config.pipeline_id.domain_pack_version,
        "graph_snapshot_id": graph_snapshot_id
        or config.pipeline_id.graph_snapshot_id,
        "questions_total": len(rows),
        "per_question": rows,
        "aggregates": aggregates,
        "status": status,
        "pipeline_errors": list(pipeline.errors),
    }
    assert_valid_result_artifact(artifact)
    return artifact


def dataset_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
