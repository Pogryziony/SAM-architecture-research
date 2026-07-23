"""Fair baseline interfaces — placeholders and missing creds → NOT_RUN."""

from __future__ import annotations

from nexus.baselines import (
    BaselineRequest,
    BaselineStatus,
    get_arm,
    list_arms,
    run_baseline_or_not_run,
)


def test_placeholder_arms_cannot_be_mistaken_for_real_results():
    arm = get_arm("placeholder_evidence_blind")
    assert arm.is_placeholder is True
    result = run_baseline_or_not_run(
        arm,
        BaselineRequest(
            arm_id=arm.arm_id,
            question_id="q0",
            question="Who invented NEXUS?",
        ),
    )
    assert result.status == BaselineStatus.NOT_RUN
    assert "placeholder" in result.failure_reason.lower()


def test_closed_book_llm_is_not_run_without_credentials(monkeypatch):
    monkeypatch.delenv("NEXUS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("NEXUS_LLM_MODEL", raising=False)
    arm = get_arm("closed_book_llm")
    result = run_baseline_or_not_run(
        arm,
        BaselineRequest(arm_id=arm.arm_id, question_id="q0", question="Hi?"),
        command="python -m benchmarks.run_fair_baselines --arm closed_book_llm",
    )
    assert result.status == BaselineStatus.NOT_RUN
    assert "env:NEXUS_LLM_API_KEY" in result.prerequisites
    assert result.command


def test_modern_rag_flags():
    assert get_arm("hybrid_rag_rerank").modern_rag is True
    assert get_arm("bm25_rag").modern_rag is False
    ids = {a.arm_id for a in list_arms(include_placeholders=False)}
    assert "closed_book_llm" in ids
    assert "placeholder_synthesizing_model" not in ids
