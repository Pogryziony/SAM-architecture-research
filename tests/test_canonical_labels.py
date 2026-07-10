"""Unit tests for canonical intent label normalization.

Proves that dataset loader, rule classifier, and eval script all agree
on the canonical label set — this is a metric bug fix, not a model change.
"""

from __future__ import annotations

import json

import pytest

from nexus.utils.canonical_labels import (
    CANONICAL_INTENT_LABELS,
    CANONICAL_LABEL_SET,
    canonicalize_intent,
    canonicalize_question_type,
)
from stack.encoder.intent_rules import RuleIntentClassifier


def _load_test_jsonl(path: str) -> list[dict]:
    import os
    # __file__ = tests/test_canonical_labels.py → repo root is one dir up from tests/
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, path), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TestCanonicalLabelMapping:
    """Verify canonical mapping covers all known raw values."""

    def test_factual_maps_to_factual_lookup(self):
        assert canonicalize_intent("factual") == "factual_lookup"
        assert canonicalize_intent("factual_lookup") == "factual_lookup"

    def test_multihop_variants_map_to_multi_hop(self):
        assert canonicalize_intent("multihop") == "multi_hop"
        assert canonicalize_intent("multi-hop") == "multi_hop"
        assert canonicalize_intent("multi_hop") == "multi_hop"

    def test_comparison_variants_map_to_comparison(self):
        assert canonicalize_intent("comparative") == "comparison"
        assert canonicalize_intent("comparison") == "comparison"

    def test_diagnostic_variants_map_to_diagnostic(self):
        assert canonicalize_intent("causal_explanation") == "diagnostic"
        assert canonicalize_intent("dependency_chain") == "diagnostic"
        assert canonicalize_intent("impact_analysis") == "diagnostic"
        assert canonicalize_intent("diagnostic") == "diagnostic"

    def test_unknown_label_passthrough(self):
        """Unknown labels pass through unchanged for debugging."""
        assert canonicalize_intent("bogus_label") == "bogus_label"

    def test_canonical_label_set_has_four_labels(self):
        assert CANONICAL_LABEL_SET == frozenset({
            "factual_lookup", "comparison", "multi_hop", "diagnostic",
        })

    def test_all_canonical_values_are_in_set(self):
        """Every value in CANONICAL_INTENT_LABELS must be in CANONICAL_LABEL_SET."""
        for raw, canonical in CANONICAL_INTENT_LABELS.items():
            assert canonical in CANONICAL_LABEL_SET, (
                f"Canonical value '{canonical}' (from raw '{raw}') "
                f"not in CANONICAL_LABEL_SET"
            )


class TestTestSplitLabelsAreCanonical:
    """Verify the frozen test split only uses canonical labels."""

    def test_test_jsonl_intents_are_canonical(self):
        questions = _load_test_jsonl("stack/encoder/data/test.jsonl")
        for q in questions:
            intent = q.get("intent", "")
            assert intent in CANONICAL_LABEL_SET, (
                f"Question {q['id']}: intent '{intent}' not in canonical set"
            )


class TestRuleClassifierOutputsAreCanonical:
    """Verify the rule classifier only outputs canonical labels."""

    def test_all_rule_outputs_are_canonical(self):
        classifier = RuleIntentClassifier()
        for _, intent, _ in classifier.RULES:
            assert intent in CANONICAL_LABEL_SET, (
                f"Rule intent '{intent}' not in canonical set"
            )


class TestDatasetQuestionTypesMapCorrectly:
    """Verify questions.jsonl question_type values all map to canonical."""

    def test_all_question_types_map_to_canonical(self):
        sample_questions = _load_test_jsonl("benchmarks/qa-dataset/questions.jsonl")
        for q in sample_questions:
            qt = q.get("question_type", "")
            canonical = canonicalize_question_type(qt)
            assert canonical in CANONICAL_LABEL_SET, (
                f"Question {q['id']}: question_type '{qt}' maps to "
                f"'{canonical}' — not in canonical set"
            )


class TestLabelAgreement:
    """Prove all three components agree on the canonical label set."""

    def test_rule_classifier_only_uses_canonical_labels(self):
        """Rule classifier RULES all output canonical labels (subset of canonical)."""
        classifier = RuleIntentClassifier()
        rule_labels = {intent for _, intent, _ in classifier.RULES}
        assert rule_labels <= CANONICAL_LABEL_SET, (
            f"Rule labels {rule_labels} not subset of canonical {CANONICAL_LABEL_SET}"
        )

    def test_frozen_test_split_only_uses_canonical_labels(self):
        """Frozen test split only uses canonical labels (may not cover all 4)."""
        questions = _load_test_jsonl("stack/encoder/data/test.jsonl")
        dataset_labels = {q.get("intent", "") for q in questions}
        assert dataset_labels <= CANONICAL_LABEL_SET, (
            f"Test split labels {dataset_labels} not subset of "
            f"canonical {CANONICAL_LABEL_SET}"
        )

    def test_questions_jsonl_maps_all_to_canonical(self):
        """All question_type values in questions.jsonl map to canonical set."""
        questions = _load_test_jsonl("benchmarks/qa-dataset/questions.jsonl")
        for q in questions:
            qt = q.get("question_type", "")
            canonical = canonicalize_question_type(qt)
            assert canonical in CANONICAL_LABEL_SET, (
                f"Question {q['id']}: '{qt}' → '{canonical}' not canonical"
            )
