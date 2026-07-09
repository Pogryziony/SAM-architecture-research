"""
Unit tests for benchmark arm guards (Phase 3).

Tests:
  - Validator catches zero retrieval_tokens for NEXUS arm
  - Validator catches zero retrieval_tokens for RAG arm
  - Validator catches model mismatch between arms
  - Validator passes for correct config
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure benchmarks/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.run_benchmark import validate_benchmark_results  # noqa: E402
from benchmarks.compare_arms import compare_paired  # noqa: E402


# ── Helpers ──

def _make_result(arm_mode, retrieval_tokens, nexus_model="qwen2.5:latest",
                 rag_model="qwen2.5:latest"):
    """Build a minimal result dict for validation."""
    return {
        "question_id": "q001",
        "arm_mode": arm_mode,
        "retrieval_tokens": retrieval_tokens,
        "nexus": {"model": nexus_model},
        "baseline": {"model": rag_model},
    }


def _make_config():
    return {"arm_nexus": "nexus", "arm_rag": "rag_retrieval"}


# ═══════════════════════════════════════════════════════════════════════
# Test 1 — Validator catches zero retrieval_tokens for NEXUS
# ═══════════════════════════════════════════════════════════════════════

def test_validator_catches_zero_nexus_tokens():
    """NEXUS arm with all retrieval_tokens == 0 should fail validation."""
    results = [
        _make_result("nexus", 0),
        _make_result("nexus", 0),
        _make_result("nexus", 0),
    ]
    errors = validate_benchmark_results(results, _make_config())
    assert len(errors) >= 1
    assert any("NEXUS arm" in e for e in errors)


def test_validator_passes_nexus_with_tokens():
    """NEXUS arm with non-zero retrieval_tokens should pass."""
    results = [
        _make_result("nexus", 42),
        _make_result("nexus", 0),  # one zero is fine if others have tokens
    ]
    errors = validate_benchmark_results(results, _make_config())
    nexus_errors = [e for e in errors if "NEXUS" in e]
    assert len(nexus_errors) == 0


# ═══════════════════════════════════════════════════════════════════════
# Test 2 — Validator catches zero retrieval_tokens for RAG
# ═══════════════════════════════════════════════════════════════════════

def test_validator_catches_zero_rag_tokens():
    """RAG arm labeled rag_retrieval with all retrieval_tokens == 0 should fail."""
    results = [
        _make_result("rag_retrieval", 0),
        _make_result("rag_retrieval", 0),
        _make_result("rag_retrieval", 0),
    ]
    errors = validate_benchmark_results(results, _make_config())
    assert len(errors) >= 1
    assert any("RAG arm" in e for e in errors)


def test_validator_passes_rag_with_tokens():
    """RAG arm labeled rag_retrieval with some tokens should pass."""
    results = [
        _make_result("rag_retrieval", 50),
        _make_result("rag_retrieval", 0),
    ]
    errors = validate_benchmark_results(results, _make_config())
    rag_errors = [e for e in errors if "RAG arm" in e]
    assert len(rag_errors) == 0


def test_validator_skips_evidence_blind():
    """Evidence-blind arm is NOT checked for retrieval tokens."""
    results = [
        _make_result("evidence_blind", 0),
        _make_result("evidence_blind", 0),
    ]
    errors = validate_benchmark_results(results, {"arm_rag": "evidence_blind"})
    assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════
# Test 3 — Validator catches model mismatch
# ═══════════════════════════════════════════════════════════════════════

def test_validator_catches_model_mismatch():
    """Different models for NEXUS vs RAG should fail validation."""
    results = [
        _make_result("nexus", 50, nexus_model="qwen2.5:latest",
                     rag_model="llama3:8b"),
        _make_result("nexus", 30, nexus_model="qwen2.5:latest",
                     rag_model="llama3:8b"),
    ]
    errors = validate_benchmark_results(results, _make_config())
    assert len(errors) >= 1
    assert any("Model mismatch" in e for e in errors)


def test_validator_passes_same_model():
    """Same model for both arms should pass."""
    results = [
        _make_result("nexus", 100, nexus_model="qwen2.5:latest",
                     rag_model="qwen2.5:latest"),
        _make_result("rag_retrieval", 50, nexus_model="qwen2.5:latest",
                     rag_model="qwen2.5:latest"),
    ]
    errors = validate_benchmark_results(results, _make_config())
    assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════
# Test 4 — Validator passes for correct config
# ═══════════════════════════════════════════════════════════════════════

def test_validator_passes_correct_config():
    """Mixed NEXUS + rag_retrieval with non-zero tokens and same model."""
    results = [
        _make_result("nexus", 200, nexus_model="qwen2.5:latest",
                     rag_model="qwen2.5:latest"),
        _make_result("nexus", 80, nexus_model="qwen2.5:latest",
                     rag_model="qwen2.5:latest"),
        _make_result("rag_retrieval", 45, nexus_model="qwen2.5:latest",
                     rag_model="qwen2.5:latest"),
        _make_result("rag_retrieval", 150, nexus_model="qwen2.5:latest",
                     rag_model="qwen2.5:latest"),
    ]
    errors = validate_benchmark_results(results, _make_config())
    assert errors == [], f"Expected no errors, got: {errors}"


def test_validator_empty_results():
    """Empty results list should produce no errors (no arm configured)."""
    errors = validate_benchmark_results([], {"arm_rag": "evidence_blind"})
    assert errors == []


def test_validator_none_tokens_handled():
    """None retrieval_tokens should be treated as 0."""
    results = [
        {"question_id": "q001", "arm_mode": "nexus",
         "retrieval_tokens": None,
         "nexus": {"model": "qwen2.5:latest"},
         "baseline": {"model": "qwen2.5:latest"}},
    ]
    errors = validate_benchmark_results(results, _make_config())
    nexus_errors = [e for e in errors if "NEXUS" in e]
    # None treated as 0 via .get(..., 0) → sum == 0 → error
    assert len(nexus_errors) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Test — Guard 1: RAG arm configured but zero result rows
# ═══════════════════════════════════════════════════════════════════════

def test_rag_arm_zero_rows_fails():
    """When arm_rag is 'rag_retrieval' but no rag_retrieval rows exist, fail."""
    results = [
        _make_result("nexus", 100),
        _make_result("nexus", 80),
    ]
    config = {"arm_rag": "rag_retrieval"}
    errors = validate_benchmark_results(results, config)
    assert any("produced 0 result rows" in e for e in errors), \
        f"Expected error about zero RAG rows, got: {errors}"


def test_rag_arm_zero_rows_ok_when_evidence_blind():
    """When arm_rag is 'evidence_blind', zero rag_retrieval rows is fine."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    config = {"arm_rag": "evidence_blind"}
    errors = validate_benchmark_results(results, config)
    rag_zero_errors = [e for e in errors if "produced 0 result rows" in e]
    assert len(rag_zero_errors) == 0, f"Expected no zero-rows error for evidence_blind, got: {rag_zero_errors}"


# ═══════════════════════════════════════════════════════════════════════
# Test — Guard 2: Row count mismatch
# ═══════════════════════════════════════════════════════════════════════

def test_row_count_mismatch_fails():
    """Row count must equal question_count * arm_count."""
    # 3 questions × 2 arms = 6 expected rows. Give only 4.
    results = [
        _make_result("nexus", 100),
        _make_result("rag_retrieval", 50),
        _make_result("nexus", 80),
        _make_result("rag_retrieval", 30),
    ]
    errors = validate_benchmark_results(results, _make_config(), question_count=3)
    assert any("Row count mismatch" in e for e in errors), \
        f"Expected row count mismatch error, got: {errors}"


def test_row_count_match_passes():
    """Correct row count should pass."""
    results = [
        _make_result("nexus", 100),
        _make_result("rag_retrieval", 50),
        _make_result("nexus", 80),
        _make_result("rag_retrieval", 30),
    ]
    errors = validate_benchmark_results(results, _make_config(), question_count=2)
    assert not any("Row count mismatch" in e for e in errors), \
        f"Expected no mismatch error for correct count, got: {errors}"


def test_row_count_400_duplication():
    """Excess rows (duplication) should fail."""
    # 2 questions → expect 4 rows. Give 400 (simulating duplication bug).
    results = []
    for i in range(200):
        results.append(_make_result("nexus", 100))
        results.append(_make_result("rag_retrieval", 50))
    errors = validate_benchmark_results(results, _make_config(), question_count=2)
    assert any("Row count mismatch" in e for e in errors), \
        f"Expected row count mismatch for 400 rows vs 4 expected, got: {errors}"


# ═══════════════════════════════════════════════════════════════════════
# Test — Guard 3: paired_n == 0
# ═══════════════════════════════════════════════════════════════════════

def test_paired_n_zero_is_detected():
    """paired_n == 0 should be detectable via compare_paired (no co-scored questions)."""
    # All NEXUS scores are None, all RAG scores are valid → paired_n = 0
    nexus = [None, None, None, None, None]
    rag = [0.5, 0.3, 0.0, 1.0, 0.3333]
    comp = compare_paired(nexus, rag, "NEXUS", "RAG")
    assert comp["paired_n"] == 0, \
        f"When all NEXUS scores are None, paired_n should be 0, got {comp['paired_n']}"


def test_paired_n_zero_both_none():
    """All scores None → paired_n == 0."""
    nexus = [None, None, None]
    rag = [None, None, None]
    comp = compare_paired(nexus, rag)
    assert comp["paired_n"] == 0


def test_paired_n_nonzero_is_fine():
    """Non-zero paired_n is expected in normal operation."""
    nexus = [1.0, 0.5, 0.0]
    rag = [0.0, 0.5, 0.0]
    comp = compare_paired(nexus, rag)
    assert comp["paired_n"] == 3
    assert comp["sign_test_p"] is not None
