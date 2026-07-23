"""Executable baseline adapters emitting nexus-eval-result-v1 records.

Real LLM/RAG calls require credentials. Without them adapters emit schema-valid
NOT_RUN artifacts. Deterministic placeholders remain explicitly labeled and
must not be reported as LLM or modern RAG results.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from nexus.baselines.interface import (
    BaselineRequest,
    BaselineResult,
    BaselineStatus,
    missing_prerequisites,
    run_baseline_or_not_run,
)
from nexus.baselines.registry import get_arm
from nexus.evaluation.aggregate import aggregate_question_records
from nexus.evaluation.dataset_identity import hash_dataset
from nexus.evaluation.schema import (
    RESULT_SCHEMA_VERSION,
    TerminalOutcome,
    build_question_record,
    empty_metric_applicability,
)
from nexus.evaluation.validate import assert_valid_result_artifact
from nexus.pipeline.config import CONFIG_IDENTITY_SCHEMA


def _dataset_hash(questions: Sequence[Mapping[str, Any]]) -> str:
    return hash_dataset(questions)


def llm_decoding_defaults() -> dict[str, Any]:
    return {
        "temperature": float(os.environ.get("NEXUS_LLM_TEMPERATURE", "0")),
        "top_p": float(os.environ.get("NEXUS_LLM_TOP_P", "1")),
        "max_tokens": int(os.environ.get("NEXUS_LLM_MAX_TOKENS", "512")),
        "seed": os.environ.get("NEXUS_LLM_SEED"),
        "timeout_s": float(os.environ.get("NEXUS_LLM_TIMEOUT_S", "60")),
        "retry_max": int(os.environ.get("NEXUS_LLM_RETRY_MAX", "0")),
    }


def run_baseline_eval(
    arm_id: str,
    questions: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    dataset_sha256: str = "",
    comparison_mode: str = "system_level",
    source_commit: str = "UNKNOWN",
    corpus_id: str = "",
    system_prompt: str = "",
) -> dict[str, Any]:
    """Run one baseline arm; always returns a schema-valid artifact."""
    arm = get_arm(arm_id)
    executed_at = datetime.now(timezone.utc).isoformat()
    ds_hash = dataset_sha256 or _dataset_hash(questions)
    decoding = llm_decoding_defaults()
    rows: list[dict[str, Any]] = []
    command = (
        f"python benchmarks/run_fair_baselines.py --arm {arm_id} "
        f"--comparison-mode {comparison_mode}"
    )

    for q in questions:
        req = BaselineRequest(
            arm_id=arm_id,
            question_id=str(q["id"]),
            question=str(q["question"]),
            corpus_id=corpus_id,
            decoding=decoding,
            system_prompt=system_prompt,
        )
        # Prefer registry runner; otherwise honest NOT_RUN
        if arm.run is not None and not missing_prerequisites(arm) and not arm.is_placeholder:
            result = arm.run(req)
        else:
            result = run_baseline_or_not_run(arm, req, command=command)

        if result.status == BaselineStatus.NOT_RUN:
            outcome = TerminalOutcome.NOT_RUN
        elif result.status == BaselineStatus.ERROR:
            outcome = TerminalOutcome.FAILED
        elif "insufficient evidence" in (result.answer or "").casefold():
            outcome = TerminalOutcome.ABSTAINED
        else:
            outcome = TerminalOutcome.ANSWERED

        rows.append(
            build_question_record(
                question_id=str(q["id"]),
                domain=str(q.get("domain") or "unknown"),
                question_type=str(q.get("question_type") or "unknown"),
                dataset_id=dataset_id,
                dataset_sha256=ds_hash,
                system_id=arm_id,
                profile=arm_id,
                config_hash="baseline-adapter-v1",
                config_identity_schema=CONFIG_IDENTITY_SCHEMA,
                model_id=result.model_id
                or os.environ.get("NEXUS_LLM_MODEL", "")
                or arm_id,
                checkpoint_id=result.model_id
                or os.environ.get("NEXUS_LLM_MODEL", ""),
                source_commit=source_commit,
                executed_at_utc=executed_at,
                terminal_outcome=outcome,
                question=str(q["question"]),
                final_answer=result.answer,
                retrieved_documents=result.retrieved_documents,
                failure_reason=result.failure_reason,
                metrics=empty_metric_applicability(),
                latency_ms=result.latency_ms,
                token_cost=result.token_cost,
                execution_environment={
                    "provider": result.provider,
                    "prerequisites": result.prerequisites,
                    "command": result.command or command,
                    "status": result.status.value,
                    "is_placeholder": arm.is_placeholder,
                    "modern_rag": arm.modern_rag,
                    "comparison_mode": comparison_mode,
                    "decoding": decoding,
                    "system_prompt": system_prompt,
                    "corpus_id": corpus_id,
                    "api_version": os.environ.get("NEXUS_LLM_API_VERSION", ""),
                    "base_url": os.environ.get("NEXUS_LLM_BASE_URL", ""),
                    "fixture": "UNIT_TEST_ONLY"
                    if result.provider == "mock-fixture"
                    else "",
                },
                comparison_mode=comparison_mode,
                failure_category=(
                    "not_run"
                    if result.status == BaselineStatus.NOT_RUN
                    else ("exception:BaselineError" if result.status == BaselineStatus.ERROR else "")
                ),
            ).to_dict()
        )

    status = "NOT_RUN" if any(
        r["terminal_outcome"] == "not_run" for r in rows
    ) else "VALID"
    if arm.is_placeholder:
        status = "NOT_RUN"

    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_utc": executed_at,
        "source_commit": source_commit,
        "dataset_id": dataset_id,
        "dataset_sha256": ds_hash,
        "system_id": arm_id,
        "profile": arm_id,
        "config_hash": "baseline-adapter-v1",
        "config_identity_schema": CONFIG_IDENTITY_SCHEMA,
        "comparison_mode": comparison_mode,
        "questions_total": len(rows),
        "per_question": rows,
        "aggregates": aggregate_question_records(rows),
        "status": status,
        "arm_metadata": {
            "family": arm.family,
            "is_placeholder": arm.is_placeholder,
            "modern_rag": arm.modern_rag,
            "description": arm.description,
            "requires_env": list(arm.requires_env),
            "requires_packages": list(arm.requires_packages),
        },
    }
    assert_valid_result_artifact(artifact)
    return artifact


def attach_mock_closed_book_for_unit_tests() -> None:
    """Register a mock runner on closed_book_llm for unit tests only.

    The mock is labeled ``mock-fixture`` and must never be treated as a
    benchmark result.
    """
    from nexus.baselines import registry

    def _mock(req: BaselineRequest) -> BaselineResult:
        return BaselineResult(
            arm_id=req.arm_id,
            question_id=req.question_id,
            status=BaselineStatus.OK,
            answer="MOCK_FIXTURE_ANSWER",
            model_id="mock-llm-fixture-v0",
            provider="mock-fixture",
            decoding=dict(req.decoding),
            latency_ms=1.0,
        )

    arm = registry.BASELINE_ARMS["closed_book_llm"]
    registry.BASELINE_ARMS["closed_book_llm"] = type(arm)(
        arm_id=arm.arm_id,
        family=arm.family,
        description=arm.description + " [UNIT_TEST_MOCK]",
        requires_env=(),  # mock bypasses env for unit tests
        requires_packages=(),
        is_placeholder=False,
        modern_rag=False,
        run=_mock,
    )
