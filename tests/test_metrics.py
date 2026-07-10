"""Unit tests for metric definitions in eval_gates.py.

Proves that entity_precision, entity_recall, entity_f1, and
exact_entity_accuracy are computed correctly with hand-calculated examples.
"""

from __future__ import annotations

import pytest

from stack.encoder.eval_gates import compute_stage_recall


def _compute_metrics(resolved: list[list[str]], gt: list[list[str]]) -> dict:
    """Compute metrics from resolved and GT entity ID lists, one per question.

    Args:
        resolved: Resolved entity IDs per question.
        gt: GT entity IDs per question.

    Returns:
        Dict with entity_precision, entity_recall, entity_f1, exact_entity_accuracy.
    """
    n = len(resolved)
    total_correct = 0
    total_resolved = 0
    total_gt = 0
    exact_matches = 0

    for r_ids, g_ids in zip(resolved, gt):
        r_set = set(r_ids)
        g_set = set(g_ids)

        total_resolved += len(r_set)
        total_gt += len(g_set)

        for eid in r_set:
            if eid in g_set:
                total_correct += 1

        if g_set and g_set.issubset(r_set):
            exact_matches += 1

    precision = total_correct / max(total_resolved, 1)
    recall = total_correct / max(total_gt, 1)
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-8)
    )
    exact_acc = exact_matches / n if n > 0 else 0.0

    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "exact_entity_accuracy": exact_acc,
        "total_correct": total_correct,
        "total_resolved": total_resolved,
        "total_gt": total_gt,
        "exact_matches": exact_matches,
    }


class TestStageRecallDenominators:
    def test_pair_level_stage_recall(self):
        assert compute_stage_recall(231, 275) == pytest.approx(231 / 275)
        assert compute_stage_recall(0, 275) == 0.0
        assert compute_stage_recall(1, 0) == 0.0


class TestHandCalculatedMetrics:
    """Verify metrics match hand-calculated values on known examples."""

    def test_five_questions_hand_calculated(self):
        """5 questions, 1 GT entity each.
        GT = [A, A, B, C, E], resolved = [A, A, D, C, F]
        Exact calculation:
          - correct = 3 (q0: A, q1: A, q3: C)
          - resolved = 5, GT = 5
          - precision = 3/5 = 0.6
          - recall = 3/5 = 0.6
          - f1 = 2*0.6*0.6 / 1.2 = 0.6
          - exact = 3/5 = 0.6 (q0, q1, q3 have all GT in resolved;
            q2 [B] ⊄ [D], q4 [E] ⊄ [F])
        """
        gt = [["A"], ["A"], ["B"], ["C"], ["E"]]
        resolved = [["A"], ["A"], ["D"], ["C"], ["F"]]
        result = _compute_metrics(resolved, gt)

        assert result["entity_precision"] == pytest.approx(0.6)
        assert result["entity_recall"] == pytest.approx(0.6)
        assert result["entity_f1"] == pytest.approx(0.6)
        assert result["exact_entity_accuracy"] == pytest.approx(0.6)

    def test_perfect_match(self):
        """All GT entities resolved exactly → precision=1, recall=1, f1=1, exact=1."""
        gt = [["X", "Y"], ["Z"], ["W"]]
        resolved = [["X", "Y"], ["Z"], ["W"]]
        result = _compute_metrics(resolved, gt)

        assert result["entity_precision"] == pytest.approx(1.0)
        assert result["entity_recall"] == pytest.approx(1.0)
        assert result["entity_f1"] == pytest.approx(1.0)
        assert result["exact_entity_accuracy"] == pytest.approx(1.0)

    def test_no_match(self):
        """No GT entities resolved → precision=0, recall=0, f1=0, exact=0."""
        gt = [["A"], ["B"], ["C"]]
        resolved = [["D"], ["E"], ["F"]]
        result = _compute_metrics(resolved, gt)

        assert result["entity_precision"] == pytest.approx(0.0)
        assert result["entity_recall"] == pytest.approx(0.0)
        assert result["entity_f1"] == pytest.approx(0.0)
        assert result["exact_entity_accuracy"] == pytest.approx(0.0)

    def test_partial_overlap_multi_entity(self):
        """Mix of partial and exact matches across questions.
        GT:      q0=[A,B], q1=[C],   q2=[D,E]
        Resolved: q0=[A,C], q1=[C,X], q2=[D,E,F]
        
        q0: correct=1 (A), resolved=2, gt=2, exact: no (B missing)
        q1: correct=1 (C), resolved=2, gt=1, exact: yes (C in [C,X])
        q2: correct=2 (D,E), resolved=3, gt=2, exact: yes (D,E in [D,E,F])
        Total: correct=4, resolved=7, gt=5
        precision = 4/7 ≈ 0.5714
        recall = 4/5 = 0.8
        f1 = 2*0.5714*0.8/(0.5714+0.8) ≈ 0.6667
        exact = 2/3 (q1 and q2)
        """
        gt = [["A", "B"], ["C"], ["D", "E"]]
        resolved = [["A", "C"], ["C", "X"], ["D", "E", "F"]]
        result = _compute_metrics(resolved, gt)

        assert result["entity_precision"] == pytest.approx(4 / 7, rel=1e-4)
        assert result["entity_recall"] == pytest.approx(0.8, rel=1e-4)
        assert result["entity_f1"] == pytest.approx(0.6667, abs=0.001)
        assert result["exact_entity_accuracy"] == pytest.approx(2 / 3, rel=1e-4)

    def test_empty_gt_question(self):
        """Questions with no GT entities don't affect exact match (gt_ids is empty)."""
        gt = [["A"], [], ["B"]]
        resolved = [["A"], ["X"], ["B"]]
        result = _compute_metrics(resolved, gt)

        # q0: correct=1, resolved=1, gt=1, exact
        # q1: correct=0, resolved=1, gt=0, not counted for exact (gt empty)
        # q2: correct=1, resolved=1, gt=1, exact
        # Total: correct=2, resolved=3, gt=2
        # precision=2/3, recall=2/2=1.0, f1=2*0.667*1/(1.667)=0.8, exact=2/3
        assert result["entity_precision"] == pytest.approx(2 / 3, rel=1e-4)
        assert result["entity_recall"] == pytest.approx(1.0)
        assert result["entity_f1"] == pytest.approx(0.8, rel=1e-4)
        assert result["exact_entity_accuracy"] == pytest.approx(2 / 3, rel=1e-4)

    def test_no_questions(self):
        """Empty input → all metrics 0."""
        result = _compute_metrics([], [])
        assert result["entity_precision"] == pytest.approx(0.0)
        assert result["entity_recall"] == pytest.approx(0.0)
        assert result["entity_f1"] == pytest.approx(0.0)
        assert result["exact_entity_accuracy"] == pytest.approx(0.0)


class TestConsistentMetricNaming:
    """Verify that precision/recall/f1 are consistently defined across functions.

    Precision = correct / total_predictions
    Recall = correct / total_GT_entities
    F1 = 2 * P * R / (P + R)
    """

    def test_precision_is_correct_over_resolved(self):
        """entity_precision = correct predictions / total predictions made."""
        # 3 questions, 5 GT, 6 resolved, 4 correct
        gt = [["A", "B"], ["C"], ["D", "E"]]
        resolved = [["A", "B", "X"], ["C", "Y"], ["D"]]
        result = _compute_metrics(resolved, gt)

        # correct: q0: A,B=2, q1: C=1, q2: D=1 → total=4
        # resolved: 3+2+1=6
        # gt: 2+1+2=5
        assert result["entity_precision"] == pytest.approx(4 / 6)
        assert result["entity_recall"] == pytest.approx(4 / 5)
        assert result["entity_f1"] == pytest.approx(
            2 * (4 / 6) * (4 / 5) / ((4 / 6) + (4 / 5))
        )

    def test_recall_is_correct_over_gt_total(self):
        """entity_recall = correct GT matches / total GT entities."""
        gt = [["A"], ["B"], ["C"], ["D"], ["E"]]
        resolved = [["A"], ["B"], ["X"], ["Y"], ["Z"]]
        result = _compute_metrics(resolved, gt)

        # correct=2, gt=5, resolved=5
        assert result["entity_recall"] == pytest.approx(2 / 5)
        # recall = 0.4, precision = 0.4, f1 = 0.4
        assert result["entity_precision"] == pytest.approx(0.4)
        assert result["entity_f1"] == pytest.approx(0.4)
