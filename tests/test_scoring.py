"""
Phase 1 gate tests for the unified scoring module.

These tests verify:
  1. NEXUS and RAG use the SAME scoring function.
  2. None conditions are answer-independent.
  3. "Insufficient evidence" → 0.0.
  4. Regenerated JSON contains paired_n, sign_test_p, win_loss_tie.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make sure benchmarks/ is importable
_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(_BENCHMARKS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR.parent))

from benchmarks.scoring import compute_fact_score  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# Test 1 — SAME scoring function
# ═══════════════════════════════════════════════════════════════════════

def test_nexus_and_rag_use_same_function():
    """Verify no code path differs between NEXUS and RAG — single function."""
    # The scoring module has exactly one public function: compute_fact_score.
    # Both arms must import it.
    from benchmarks import scoring as scoring_mod
    assert hasattr(scoring_mod, "compute_fact_score"), \
        "scoring.py must export compute_fact_score"

    # Verify the function is the canonical one (not a copy in run_benchmark.py)
    score_func = scoring_mod.compute_fact_score
    assert score_func.__module__ == "benchmarks.scoring", \
        f"compute_fact_score must be defined in benchmarks.scoring, got {score_func.__module__}"

    # Smoke test: both "arms" call the same function
    gt = "99.87% accuracy with 1,650 slots"
    nexus_result = compute_fact_score("99.87% accuracy", gt)
    rag_result = compute_fact_score("The accuracy was 99.87% with 1,650 slots", gt)
    # Both should return the same shape
    assert "fuzzy_accuracy" in nexus_result
    assert "fuzzy_accuracy" in rag_result
    assert "exact_accuracy" in nexus_result
    assert "exact_accuracy" in rag_result
    assert "scoring_detail" in nexus_result
    assert "scoring_detail" in rag_result


# ═══════════════════════════════════════════════════════════════════════
# Test 2 — None is answer-independent
# ═══════════════════════════════════════════════════════════════════════

def test_none_is_answer_independent():
    """None must only depend on GT having no extractable facts — never on answer."""
    # GT with NO extractable facts
    gt_no_facts = "Hello world."
    # Falsely verbose answer
    result_verbose = compute_fact_score(
        "99.87% accuracy, 1,650 slots, Exp_0_6_Validation", gt_no_facts
    )
    assert result_verbose["fuzzy_accuracy"] is None, \
        "None expected when GT has no facts, regardless of verbose answer"
    assert result_verbose["exact_accuracy"] is None

    # Empty answer
    result_empty = compute_fact_score("", gt_no_facts)
    assert result_empty["fuzzy_accuracy"] is None, \
        "None expected when GT has no facts, regardless of empty answer"
    assert result_empty["exact_accuracy"] is None

    # Whitespace answer
    result_ws = compute_fact_score("   ", gt_no_facts)
    assert result_ws["fuzzy_accuracy"] is None, \
        "None expected when GT has no facts, regardless of whitespace answer"

    # GT WITH extractable facts — should NOT be None, even for empty answer
    gt_with_facts = "99.87% accuracy"
    result_with_facts_empty = compute_fact_score("", gt_with_facts)
    assert result_with_facts_empty["fuzzy_accuracy"] is not None, \
        "Should get a score (0.0) when GT has facts, even with empty answer"
    assert result_with_facts_empty["fuzzy_accuracy"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Test 3 — "Insufficient evidence" → 0.0
# ═══════════════════════════════════════════════════════════════════════

def test_insufficient_evidence_is_zero():
    """'Insufficient evidence' must yield 0.0 regardless of GT."""
    gt = "99.87% accuracy with 1,650 slots"

    r1 = compute_fact_score("Insufficient evidence to answer.", gt)
    assert r1["fuzzy_accuracy"] == 0.0
    assert r1["exact_accuracy"] == 0.0

    r2 = compute_fact_score("INSUFFICIENT EVIDENCE", gt)
    assert r2["fuzzy_accuracy"] == 0.0
    assert r2["exact_accuracy"] == 0.0

    r3 = compute_fact_score("insufficient evidence", gt)
    assert r3["fuzzy_accuracy"] == 0.0
    assert r3["exact_accuracy"] == 0.0

    # Even with correct numbers in the answer, if it says "insufficient evidence"
    r4 = compute_fact_score("Insufficient evidence (accuracy was 99.87%)", gt)
    assert r4["fuzzy_accuracy"] == 0.0
    assert r4["exact_accuracy"] == 0.0

    # GT with no facts, insufficient evidence → still None (GT-driven)
    gt_no_facts = "Hello world."
    r5 = compute_fact_score("Insufficient evidence", gt_no_facts)
    assert r5["fuzzy_accuracy"] is None, \
        "None takes precedence when GT has no facts (arm-independent)"
    assert r5["exact_accuracy"] is None


# ═══════════════════════════════════════════════════════════════════════
# Test 4 — Regenerated JSON structure
# ═══════════════════════════════════════════════════════════════════════

def test_regenerated_json_has_required_fields():
    """Regenerated JSON must contain paired_n, sign_test_p, win_loss_tie."""
    results_dir = _BENCHMARKS_DIR / "results"
    if not results_dir.is_dir():
        pytest.skip("No regenerated results directory yet — run regenerate_comparison.py first")

    json_files = sorted(results_dir.glob("nexus_vs_rag_*.json"))
    if not json_files:
        pytest.skip("No regenerated JSON files found in benchmarks/results/")

    latest = json_files[-1]
    with open(latest, encoding="utf-8") as fh:
        data = json.load(fh)

    comparison = data.get("comparison", {})

    assert "paired_n" in comparison, \
        f"Regenerated JSON must contain paired_n (in {latest.name})"
    assert isinstance(comparison["paired_n"], int), \
        "paired_n must be an integer"
    assert comparison["paired_n"] > 0, \
        "paired_n must be > 0 (some questions should be co-scored)"

    assert "sign_test_p" in comparison, \
        f"Regenerated JSON must contain sign_test_p (in {latest.name})"
    assert isinstance(comparison["sign_test_p"], (float, int)), \
        "sign_test_p must be a number"

    assert "win_loss_tie" in comparison, \
        f"Regenerated JSON must contain win_loss_tie (in {latest.name})"
    wlt = comparison["win_loss_tie"]
    assert "NEXUS_wins" in wlt
    assert "RAG_wins" in wlt
    assert "ties" in wlt


# ═══════════════════════════════════════════════════════════════════════
# Test 5 — Paired comparison helper tests
# ═══════════════════════════════════════════════════════════════════════

def test_compare_paired_basic():
    """Basic paired comparison works."""
    from benchmarks.compare_arms import compare_paired

    # 5 questions, 4 co-scored, 1 NEXUS-only
    nexus = [1.0, 0.5, 0.0, 0.3333, 0.0]
    rag = [0.0, 0.5, 0.0, 0.5, None]

    comp = compare_paired(nexus, rag, "NEXUS", "RAG")

    assert comp["paired_n"] == 4
    assert comp["sign_test_p"] is not None
    assert comp["nexus"]["mean_accuracy"] == round((1.0 + 0.5 + 0.0 + 0.3333) / 4, 4)
    assert comp["rag"]["mean_accuracy"] == round((0.0 + 0.5 + 0.0 + 0.5) / 4, 4)

    wlt = comp["win_loss_tie"]
    # NEXUS wins: 1.0>0.0, 0.3333>0.0? No: 0.3333 < 0.5
    # 1.0>0.0 → NEXUS win
    # 0.5=0.5 → tie
    # 0.0=0.0 → tie
    # 0.3333<0.5 → RAG win
    assert wlt["NEXUS_wins"] == 1
    assert wlt["RAG_wins"] == 1
    assert wlt["ties"] == 2

    sec = comp["secondary_unpaired"]
    assert sec["nexus_only_scored"]["n"] == 1
    assert sec["rag_only_scored"]["n"] == 0


def test_compare_paired_all_ties():
    """All ties → sign test p=1.0."""
    from benchmarks.compare_arms import compare_paired

    nexus = [0.5, 0.5, 0.5]
    rag = [0.5, 0.5, 0.5]
    comp = compare_paired(nexus, rag)

    assert comp["paired_n"] == 3
    assert comp["sign_test_p"] == 1.0
    assert comp["win_loss_tie"]["NEXUS_wins"] == 0
    assert comp["win_loss_tie"]["RAG_wins"] == 0
    assert comp["win_loss_tie"]["ties"] == 3


def test_sign_test_small_n():
    """Exact binomial sign test for small n."""
    from benchmarks.compare_arms import sign_test

    # 0 wins vs 5 — H0: p=0.5
    r = sign_test(0, 5, 0)
    # Two-sided: P(X=0) + P(X=5) = 2 * 0.5^5 = 2/32 = 0.0625
    assert abs(r["p_value"] - 0.0625) < 0.01

    # 3 wins vs 7
    r2 = sign_test(3, 7, 0)
    # n=10, k=3: P(X<=3) + P(X>=7) = 2 * sum_{i=0}^3 C(10,i)*0.5^10
    assert r2["p_value"] > 0.1  # should not be significant


def test_fact_extraction():
    """Test that fact extraction works correctly."""
    from benchmarks.scoring import _extract_key_facts, _extract_numbers

    text = "99.87% accuracy with 1,650 slots and Exp_0_6_Validation"
    facts = _extract_key_facts(text)
    assert "99.87%" in facts
    assert any("1650" in f and "slots" in f for f in facts)
    assert "exp_0_6_validation" in facts

    nums = _extract_numbers(text)
    assert 0.9987 in nums
    assert 1650.0 in nums


def test_fuzzy_number_match():
    """Test fuzzy number matching with 5% tolerance."""
    from benchmarks.scoring import _fuzzy_number_match

    gt = {0.9987, 1.0}
    pred = {0.9987}  # exact match
    matches, total = _fuzzy_number_match(pred, gt)
    assert matches == 1
    assert total == 2

    gt2 = {1.0}
    pred2 = {0.97}  # within 5% of 1.0
    matches2, total2 = _fuzzy_number_match(pred2, gt2)
    assert matches2 == 1
    assert total2 == 1

    gt3 = {1.0}
    pred3 = {0.92}  # 8% off — outside 5% tolerance
    matches3, total3 = _fuzzy_number_match(pred3, gt3)
    assert matches3 == 0
    assert total3 == 1


def test_empty_answer_is_zero():
    """Empty or None predicted answer → 0.0 (when GT has facts)."""
    gt = "99.87% accuracy"

    r1 = compute_fact_score("", gt)
    assert r1["fuzzy_accuracy"] == 0.0
    assert r1["exact_accuracy"] == 0.0

    r2 = compute_fact_score(None, gt)
    assert r2["fuzzy_accuracy"] == 0.0
    assert r2["exact_accuracy"] == 0.0

    r3 = compute_fact_score("   \n  ", gt)
    assert r3["fuzzy_accuracy"] == 0.0
    assert r3["exact_accuracy"] == 0.0


def test_zero_preserves_facts():
    """Even a 0.0 score should still have correct scoring_detail."""
    from benchmarks.scoring import _extract_numbers

    gt = "99.87% accuracy"
    result = compute_fact_score("Nothing relevant here.", gt)
    assert result["fuzzy_accuracy"] == 0.0
    assert result["exact_accuracy"] == 0.0
    detail = result["scoring_detail"]
    # GT numbers should be extracted
    assert detail["gt_numbers"] == sorted(list(_extract_numbers(gt)))
    # Entity overlap should be empty
    assert detail["entity_overlap"] == []
