from __future__ import annotations

import json
from pathlib import Path

import pytest

from stack.encoder.c2_c3 import (
    DatasetLeakError,
    build_training_groups,
    evaluate_selection_gate,
    freeze_selection,
    normalized_question,
    select_winner,
)


def _row(question: str, entities: list[str]) -> dict:
    return {"id": question, "question": question, "entities": entities, "intent": "factual_lookup"}


def test_leakage_guard_rejects_validation_question_and_never_reads_test(tmp_path: Path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text('{"question":"must not be read"}\n', encoding="utf-8")
    with pytest.raises(DatasetLeakError):
        build_training_groups(
            [_row("Train only", ["A"])],
            [_row("  TRAIN   only ", ["B"])],
            graph=None,
            test_path=test_path,
            candidate_builder=lambda _q: ["A", "B"],
        )


def test_deterministic_groups_and_hard_negative_statistics():
    questions = [_row("Train one", ["A"]), _row("Train two", ["B"])]
    kwargs = dict(graph=None, candidate_builder=lambda q: ["A", "B", "C"], seed=17, hard_negative_k=2)
    first = build_training_groups(questions, [], **kwargs)
    second = build_training_groups(questions, [], **kwargs)
    assert first.groups == second.groups
    assert first.stats["seed"] == 17
    assert first.stats["hard_negative_share"] > 0


def test_identical_candidate_groups_are_required_for_two_rankers():
    groups = [{"question_id": "q", "candidate_ids": ["A", "B"], "positive_ids": ["A"]}]
    assert evaluate_selection_gate(groups, groups)
    assert not evaluate_selection_gate(groups, [{"question_id": "q", "candidate_ids": ["B", "A"], "positive_ids": ["A"]}])


def test_selection_gate_and_tie_break():
    metrics = {"recall@10": 0.55, "recall@5": 0.40}
    assert select_winner({"encoder": metrics, "logistic": {"recall@10": 0.55, "recall@5": 0.35}}) == "encoder"
    assert evaluate_selection_gate([{"question_id": "q", "candidate_ids": ["A"], "positive_ids": ["A"]}], [{"question_id": "q", "candidate_ids": ["A"], "positive_ids": ["A"]}])
    assert freeze_selection(metrics, baseline_recall10=0.3571428571)["gate"] is True
    assert freeze_selection({"recall@10": 0.40, "recall@5": 0.3}, baseline_recall10=0.3571428571)["gate"] is False
