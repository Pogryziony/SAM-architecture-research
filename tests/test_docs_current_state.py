"""Documentation consistency — CURRENT_STATE is the source of truth."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_current_state_exists_and_declares_authority():
    text = (ROOT / "docs" / "CURRENT_STATE.md").read_text(encoding="utf-8")
    assert "canonical current-state" in text.lower() or "source of truth" in text.lower()
    assert "VALIDATED_INTERNAL" in text
    assert "SynthesizingModel" in text
    assert "EvidenceBlindModel" in text
    assert "ProductionNEXUSConfig.grounded()" in text


def test_readme_points_to_current_state():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/CURRENT_STATE.md" in readme
    assert "deterministic placeholders" in readme.lower() or "not** evidence" in readme.lower()


def test_verdict_does_not_call_placeholders_real_llms():
    verdict = (
        ROOT / "docs" / "nexus-architecture-validation-verdict.md"
    ).read_text(encoding="utf-8")
    assert "not** a real closed-book LLM" in verdict or "not a real" in verdict.lower()
    assert "VALIDATED_INTERNAL" in verdict
    assert "modern RAG" in verdict


def test_no_current_doc_claims_placeholder_is_llm():
    """Scan active status docs for forbidden unqualified LLM-only wins."""
    paths = [
        ROOT / "docs" / "CURRENT_STATE.md",
        ROOT / "docs" / "nexus-architecture-validation-verdict.md",
        ROOT / "docs" / "production-profiles.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # Forbidden: calling EvidenceBlindModel a real LLM without negation nearby.
        if "EvidenceBlindModel" in text:
            assert "not" in text.lower()
