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
import json
from pathlib import Path

import pytest

# Ensure benchmarks/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.run_benchmark import (  # noqa: E402
    validate_benchmark_results,
    validate_benchmark_artifact,
    build_benchmark_graph,
)
from benchmarks.compare_arms import compare_paired  # noqa: E402
from benchmarks.stage1b_artifact import validate_stage1b_artifact  # noqa: E402
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG  # noqa: E402


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


def _make_rag_config():
    return {"arm_rag": "rag_retrieval", "arm_nexus": "nexus"}


def _make_default_nexus_config():
    """Return a clean NEXUSConfig with all experimental flags off."""
    return NEXUSConfig(
        enable_cooccurrence_edges=False,
        enable_embedding_er=False,
        enable_associative_encoder=False,
        enable_normalization=False,
    )


def _validate_errors(*args, **kwargs):
    """Call validate_benchmark_results and return only errors (for backward compat with tests)."""
    errors, _warnings = validate_benchmark_results(*args, **kwargs)
    return errors


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
    errors, _warnings = validate_benchmark_results(results, _make_config())
    assert len(errors) >= 1
    assert any("NEXUS arm" in e for e in errors)


def test_validator_passes_nexus_with_tokens():
    """NEXUS arm with non-zero retrieval_tokens should pass."""
    results = [
        _make_result("nexus", 42),
        _make_result("nexus", 0),  # one zero is fine if others have tokens
    ]
    errors, _warnings = validate_benchmark_results(results, _make_config())
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
    errors, _warnings = validate_benchmark_results(results, _make_config())
    assert len(errors) >= 1
    assert any("RAG arm" in e for e in errors)


def test_validator_passes_rag_with_tokens():
    """RAG arm labeled rag_retrieval with some tokens should pass."""
    results = [
        _make_result("rag_retrieval", 50),
        _make_result("rag_retrieval", 0),
    ]
    errors, _warnings = validate_benchmark_results(results, _make_config())
    rag_errors = [e for e in errors if "RAG arm" in e]
    assert len(rag_errors) == 0


def test_validator_skips_evidence_blind():
    """Evidence-blind arm is NOT checked for retrieval tokens."""
    results = [
        _make_result("evidence_blind", 0),
        _make_result("evidence_blind", 0),
    ]
    errors, _warnings = validate_benchmark_results(results, {"arm_rag": "evidence_blind"})
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
    errors, _warnings = validate_benchmark_results(results, _make_config())
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
    errors, _warnings = validate_benchmark_results(results, _make_config())
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
    errors, _warnings = validate_benchmark_results(results, _make_config())
    assert errors == [], f"Expected no errors, got: {errors}"


def test_validator_empty_results():
    """Empty results list should produce no errors (no arm configured)."""
    errors, _warnings = validate_benchmark_results([], {"arm_rag": "evidence_blind"})
    assert errors == []


def test_real_r3_artifact_is_retracted_when_rag_summary_is_empty():
    """The published R3 artifact must be judged from its serialized contents."""
    artifact = _PROJECT_ROOT / "benchmarks" / "results" / "stack_baseline_v2_20260710_091759Z.json"
    errors, _warnings = validate_benchmark_artifact(artifact)
    assert any("provenance incomplete" in error or "summary incomplete" in error for error in errors)


def test_serialized_guard_rejects_invalid_stage1b_artifact(tmp_path):
    """Publication cannot be based on an in-memory result that was not serialized correctly."""
    artifact = tmp_path / "invalid-stage1b.json"
    artifact.write_text(json.dumps({"meta": {}, "metrics": {}}), encoding="utf-8")
    errors = validate_stage1b_artifact(artifact)
    assert errors
    assert any("missing metadata" in error or "missing configuration" in error for error in errors)


def test_serialized_guard_rejects_missing_and_zero_byte_artifacts(tmp_path):
    missing = validate_stage1b_artifact(tmp_path / "missing.json")
    assert any("missing" in error for error in missing)
    empty = tmp_path / "empty.json"
    empty.touch()
    assert any("zero-byte" in error for error in validate_stage1b_artifact(empty))


def test_disabled_cooccurrence_flag_produces_no_related_edges():
    graph, provenance = build_benchmark_graph(_make_default_nexus_config())
    assert provenance["effective_config"]["enable_cooccurrence_edges"] is False
    assert provenance["edge_type_counts"].get("related_to", 0) == 0
    actual_counts = {}
    for node_id in graph._nodes:
        for edge in graph.get_outgoing(node_id):
            actual_counts[edge.type] = actual_counts.get(edge.type, 0) + 1
    assert actual_counts == provenance["edge_type_counts"]
    assert sum(actual_counts.values()) == provenance["edge_count"]
    assert all(edge.type != "related_to" for node_id in graph._nodes for edge in graph.get_outgoing(node_id))


def test_graph_config_mismatch_is_rejected(tmp_path):
    artifact = {
        "config": {"arm_rag": "evidence_blind", "enable_cooccurrence_edges": False},
        "graph_provenance": {
            "effective_config": {"enable_cooccurrence_edges": True},
            "edge_type_counts": {"related_to": 2},
        },
        "summary": {}, "paired_comparison": {}, "results": [],
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    errors, _warnings = validate_benchmark_artifact(path)
    assert any("config mismatch" in error for error in errors)


def test_validator_none_tokens_handled():
    """None retrieval_tokens should be treated as 0."""
    results = [
        {"question_id": "q001", "arm_mode": "nexus",
         "retrieval_tokens": None,
         "nexus": {"model": "qwen2.5:latest"},
         "baseline": {"model": "qwen2.5:latest"}},
    ]
    errors, _warnings = validate_benchmark_results(results, _make_config())
    nexus_errors = [e for e in errors if "NEXUS" in e]
    # None treated as 0 via .get(..., 0) → sum == 0 → error
    assert len(nexus_errors) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Test — Guard 1: RAG arm configured but zero result rows
# ═══════════════════════════════════════════════════════════════════════

def test_nexus_arm_zero_rows_fails():
    """When arm_nexus is 'nexus' but no NEXUS rows exist, fail."""
    results = [_make_result("rag_retrieval", 100)]
    errors, _warnings = validate_benchmark_results(results, {"arm_nexus": "nexus", "arm_rag": "rag_retrieval"})
    assert any("NEXUS arm configured" in e for e in errors), f"Expected missing NEXUS arm error, got: {errors}"


def test_rag_arm_zero_rows_fails():
    """When arm_rag is 'rag_retrieval' but no rag_retrieval rows exist, fail."""
    results = [
        _make_result("nexus", 100),
        _make_result("nexus", 80),
    ]
    config = {"arm_rag": "rag_retrieval"}
    errors, _warnings = validate_benchmark_results(results, config)
    assert any("produced 0 result rows" in e for e in errors), \
        f"Expected error about zero RAG rows, got: {errors}"


def test_rag_arm_zero_rows_ok_when_evidence_blind():
    """When arm_rag is 'evidence_blind', zero rag_retrieval rows is fine."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    config = {"arm_rag": "evidence_blind"}
    errors, _warnings = validate_benchmark_results(results, config)
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
    errors, _warnings = validate_benchmark_results(results, _make_config(), question_count=3)
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
    errors, _warnings = validate_benchmark_results(results, _make_config(), question_count=2)
    assert not any("Row count mismatch" in e for e in errors), \
        f"Expected no mismatch error for correct count, got: {errors}"


def test_row_count_400_duplication():
    """Excess rows (duplication) should fail."""
    # 2 questions → expect 4 rows. Give 400 (simulating duplication bug).
    results = []
    for i in range(200):
        results.append(_make_result("nexus", 100))
        results.append(_make_result("rag_retrieval", 50))
    errors, _warnings = validate_benchmark_results(results, _make_config(), question_count=2)
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


# ═══════════════════════════════════════════════════════════════════════
# R3 Tests — New guards from Phase R3
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Guard 1b — Empty RAG summary
# ═══════════════════════════════════════════════════════════════════════

def test_rag_arm_empty_summary_fails():
    """Smoke test: when rag_retrieval is configured but summary.baseline is empty,
    the validator must report an error. Reproduces the stack_baseline failure."""
    results = [
        _make_result("nexus", 100),
        _make_result("rag_retrieval", 50),
        _make_result("nexus", 80),
        _make_result("rag_retrieval", 30),
    ]
    # Empty baseline summary — simulates the V3 stack_baseline bug
    empty_summary = {
        "nexus": {"answered": 2, "avg_paths_found": 12.79},
        "baseline": {},
    }
    errors, _warnings = validate_benchmark_results(
        results, _make_rag_config(), summary=empty_summary,
        question_count=2, paired_comparison={"paired_n": 2},
    )
    assert any("Empty RAG arm" in e or "baseline is empty" in e for e in errors), \
        f"Expected error about empty RAG summary, got: {errors}"


def test_rag_arm_nonempty_summary_passes():
    """Non-empty baseline summary should pass the guard."""
    results = [
        _make_result("nexus", 100),
        _make_result("rag_retrieval", 50),
        _make_result("nexus", 80),
        _make_result("rag_retrieval", 30),
    ]
    healthy_summary = {
        "nexus": {"answered": 2, "avg_paths_found": 3.5},
        "baseline": {"avg_accuracy": 0.15, "avg_latency_s": 2.0},
    }
    errors, _warnings = validate_benchmark_results(
        results, _make_rag_config(), summary=healthy_summary,
        question_count=2, paired_comparison={"paired_n": 2},
    )
    rag_empty_errors = [e for e in errors if "Empty RAG" in e]
    assert len(rag_empty_errors) == 0, \
        f"Expected no empty-RAG error for healthy summary, got: {rag_empty_errors}"


# ═══════════════════════════════════════════════════════════════════════
# Guard 5 — Config integrity (experimental flags)
# ═══════════════════════════════════════════════════════════════════════

def test_experimental_cooccurrence_fails_without_flag():
    """enable_cooccurrence_edges=True without --allow-experimental must fail."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    bad_config = NEXUSConfig(enable_cooccurrence_edges=True)
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        nexus_config_obj=bad_config, allow_experimental=False,
    )
    assert any("enable_cooccurrence_edges" in e for e in errors), \
        f"Expected config integrity error for cooccurrence edges, got: {errors}"


def test_experimental_embedding_fails_without_flag():
    """enable_embedding_er=True without --allow-experimental must fail."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    bad_config = NEXUSConfig(enable_embedding_er=True)
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        nexus_config_obj=bad_config, allow_experimental=False,
    )
    assert any("enable_embedding_er" in e for e in errors), \
        f"Expected config integrity error for embedding ER, got: {errors}"


def test_experimental_allowed_with_flag():
    """With --allow-experimental, experimental flags are silently accepted."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    bad_config = NEXUSConfig(enable_cooccurrence_edges=True, enable_embedding_er=True)
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        nexus_config_obj=bad_config, allow_experimental=True,
    )
    assert len(errors) == 0, \
        f"Expected no errors when --allow-experimental is set, got: {errors}"


def test_clean_config_passes_without_flag():
    """Clean config with all flags off passes without --allow-experimental."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    clean = _make_default_nexus_config()
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        nexus_config_obj=clean, allow_experimental=False,
    )
    assert len(errors) == 0, \
        f"Expected no errors for clean config, got: {errors}"


# ═══════════════════════════════════════════════════════════════════════
# Guard 6 — Sanity band on avg_paths_found
# ═══════════════════════════════════════════════════════════════════════

def test_high_avg_paths_triggers_sanity_band():
    """avg_paths_found >= 8 with experimental flags off should flag as suspect."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    suspect_summary = {
        "nexus": {"answered": 1, "avg_paths_found": 12.79},
        "baseline": {},
    }
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        summary=suspect_summary, allow_experimental=False,
        nexus_config_obj=_make_default_nexus_config(),
    )
    assert any("avg_paths_found" in w or "Suspect" in w for w in _warnings), \
        f"Expected sanity band warning for high avg_paths, got: {_warnings}"


def test_normal_avg_paths_no_sanity_warning():
    """avg_paths_found < 8 should not trigger sanity band."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    normal_summary = {
        "nexus": {"answered": 1, "avg_paths_found": 3.5},
        "baseline": {},
    }
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        summary=normal_summary, allow_experimental=False,
        nexus_config_obj=_make_default_nexus_config(),
    )
    sanity_warnings = [w for w in _warnings if "Suspect" in w]
    assert len(sanity_warnings) == 0, \
        f"Expected no sanity band warning for normal avg_paths, got: {sanity_warnings}"


def test_high_avg_paths_allowed_with_experimental():
    """With --allow-experimental, high avg_paths does not trigger sanity band."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    suspect_summary = {
        "nexus": {"answered": 1, "avg_paths_found": 12.79},
        "baseline": {},
    }
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        summary=suspect_summary, allow_experimental=True,
        nexus_config_obj=NEXUSConfig(enable_cooccurrence_edges=True),
    )
    sanity_warnings = [w for w in _warnings if "Suspect" in w]
    assert len(sanity_warnings) == 0, \
        f"Expected no sanity band with --allow-experimental, got: {sanity_warnings}"


# ═══════════════════════════════════════════════════════════════════════
# Guard 4 — Answered count == 0
# ═══════════════════════════════════════════════════════════════════════

def test_nexus_answered_zero_fails():
    """NEXUS arm with 0 answered questions should fail."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    zero_answered_summary = {
        "nexus": {"answered": 0, "avg_paths_found": 2.0},
        "baseline": {},
    }
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        summary=zero_answered_summary,
    )
    assert any("answered 0 questions" in e for e in errors), \
        f"Expected error about NEXUS answered=0, got: {errors}"


def test_nexus_answered_nonzero_passes():
    """NEXUS arm with non-zero answered count should pass."""
    results = [
        _make_result("nexus", 100),
        _make_result("evidence_blind", 0),
    ]
    healthy_summary = {
        "nexus": {"answered": 1, "avg_paths_found": 2.0},
        "baseline": {},
    }
    errors, _warnings = validate_benchmark_results(
        results, {"arm_rag": "evidence_blind"},
        summary=healthy_summary,
    )
    nexus_zero_errors = [e for e in errors if "answered 0" in e]
    assert len(nexus_zero_errors) == 0, \
        f"Expected no answered=0 error, got: {nexus_zero_errors}"


# ═══════════════════════════════════════════════════════════════════════
# Integration — full stack_baseline smoke test
# ═══════════════════════════════════════════════════════════════════════

def test_stack_baseline_empty_rag_reproduced():
    """Full reproduction of stack_baseline failure:
    - arm_rag=rag_retrieval
    - Summary has empty baseline/rag
    - Experimental flags are off (clean config)
    This test asserts that the validator catches ALL issues and reports nonzero errors.
    """
    # Simulate a reasonable result set: 2 questions, NEXUS arm has data,
    # RAG arm has rows but summary is empty (the V3 bug)
    results = [
        _make_result("nexus", 120),
        _make_result("rag_retrieval", 0),
        _make_result("nexus", 85),
        _make_result("rag_retrieval", 0),
    ]
    empty_rag_summary = {
        "nexus": {"answered": 2, "avg_paths_found": 12.79},
        "baseline": {},
    }
    clean_config = _make_default_nexus_config()

    errors, _warnings = validate_benchmark_results(
        results, _make_rag_config(),
        question_count=2,
        summary=empty_rag_summary,
        paired_comparison={"paired_n": 0},
        nexus_config_obj=clean_config,
        allow_experimental=False,
    )

    # Should detect at minimum:
    # 1. Empty RAG arm (summary.baseline is {})
    # 2. paired_n == 0
    # 3. RAG arm retrieval_tokens == 0 (summary.baseline is {})
    # 2. paired_n == 0
    # 3. Sanity band (avg_paths_found=12.79 >= 8)
    # 4. RAG arm retrieval_tokens == 0
    assert len(errors) >= 2, \
        f"Expected at least 2 errors reproducing stack_baseline failure, got {len(errors)}: {errors}"
    assert any("Empty RAG" in e or "baseline is empty" in e for e in errors), \
        f"Missing empty RAG error: {errors}"
    assert any("paired_n" in e for e in errors), \
        f"Missing paired_n error: {errors}"
