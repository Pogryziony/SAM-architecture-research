from __future__ import annotations

import pytest

from stack.encoder.loader import select_entity_candidates


def test_threshold_boundaries_are_strict():
    selected, ranked = select_entity_candidates(["equal", "above"], [0.5, 0.5001], 0.5)
    assert selected == ["above"]
    assert ranked[0][0] == "above"


def test_equal_scores_preserve_candidate_order():
    selected, ranked = select_entity_candidates(["first", "second"], [0.8, 0.8], 0.1)
    assert selected == ["first", "second"]
    assert [item[0] for item in ranked] == ["first", "second"]


def test_missing_scores_do_not_create_candidates():
    selected, ranked = select_entity_candidates(["first", "second"], [0.9], 0.1)
    assert selected == ["first"]
    assert [item[0] for item in ranked] == ["first"]


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_threshold_bounds_rejected(threshold):
    with pytest.raises(ValueError):
        select_entity_candidates(["x"], [0.5], threshold)


def test_rejected_candidates_are_not_returned():
    selected, ranked = select_entity_candidates(["low", "high"], [0.1, 0.9], 0.5)
    assert selected == ["high"]
    assert all(item[0] != "low" for item in ranked)
