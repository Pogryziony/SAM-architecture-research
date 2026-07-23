"""Canonical dataset identity and forbidden question-only hashes."""

from __future__ import annotations

import pytest

from nexus.evaluation.dataset_identity import (
    assert_primary_dataset_hash,
    hash_dataset,
    question_only_hash,
)


def _q(gold: str = "Answer A", **extra):
    row = {
        "id": "q1",
        "question": "What is A?",
        "gold_answer": gold,
        "gold_entities": ["entity_a"],
        "should_abstain": False,
        "category": "factual",
    }
    row.update(extra)
    return row


def test_gold_mutation_changes_dataset_hash():
    a = hash_dataset([_q("Answer A")])
    b = hash_dataset([_q("Answer B")])
    assert a != b
    assert a != question_only_hash([_q("Answer A")])


def test_question_only_hash_forbidden_for_primary():
    records = [_q()]
    legacy = question_only_hash(records)
    with pytest.raises(ValueError, match="question-only"):
        assert_primary_dataset_hash({"dataset_sha256": legacy}, records)


def test_canonical_hash_accepted():
    records = [_q(), _q(gold="Z", id="q2", question="Z?")]
    # fix second record id
    records[1]["id"] = "q2"
    records[1]["question"] = "What is Z?"
    h = hash_dataset(records)
    assert_primary_dataset_hash({"dataset_sha256": h}, records)


def test_rebind_script_refuses():
    from benchmarks._rebind_nexus_dataset import main

    assert main() == 2
