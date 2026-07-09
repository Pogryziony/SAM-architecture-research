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
    errors = validate_benchmark_results(results, _make_config())
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
    """Empty results list should produce no errors."""
    errors = validate_benchmark_results([], _make_config())
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
