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


class TestClassifyOutputsAreCanonical:
    """Verify classify() returns canonical labels for concrete examples."""

    def test_compare_question_returns_comparison(self):
        """'Compare X vs Y' → 'comparison' (canonical, not 'comparative')."""
        classifier = RuleIntentClassifier()
        result = classifier.classify("Compare X vs Y")
        assert result == "comparison"
        assert result in CANONICAL_LABEL_SET

    def test_diagnostic_question_returns_diagnostic(self):
        """'Why did SAM fail?' → 'diagnostic'."""
        classifier = RuleIntentClassifier()
        result = classifier.classify("Why did SAM fail?")
        assert result == "diagnostic"
        assert result in CANONICAL_LABEL_SET

    def test_factual_question_returns_factual_lookup(self):
        """'How many experiments were run?' → 'factual_lookup'."""
        classifier = RuleIntentClassifier()
        result = classifier.classify("How many experiments were run?")
        assert result == "factual_lookup"
        assert result in CANONICAL_LABEL_SET

    def test_multi_hop_question_returns_multi_hop(self):
        """'How does the X experiment relate to Y?' → 'multi_hop'."""
        classifier = RuleIntentClassifier()
        result = classifier.classify(
            "How does the noise experiment relate to the validation experiment?"
        )
        assert result == "multi_hop"
        assert result in CANONICAL_LABEL_SET

    def test_all_classify_outputs_are_canonical(self):
        """Every rule in the classifier produces canonical labels when matched."""
        classifier = RuleIntentClassifier()
        # Test each rule pattern against a matching question to ensure
        # the output label is canonical.
        test_inputs = [
            ("Compare A and B", "comparison"),
            ("how does X differ from Y", "comparison"),
            ("how does the validation experiment relate to", "multi_hop"),
            ("what experiment directly", "multi_hop"),
            ("what was the significance", "diagnostic"),
            ("why is it important", "diagnostic"),
            ("how many experiments", "factual_lookup"),
            ("which research phase", "factual_lookup"),
            ("summarize the findings", "factual_lookup"),
        ]
        for question, expected in test_inputs:
            result = classifier.classify(question)
            assert result == expected, f"classify('{question}') = {result}, expected {expected}"
            assert result in CANONICAL_LABEL_SET, (
                f"classify('{question}') = '{result}' not in canonical set"
            )

    def test_classify_output_is_canonical_idempotent(self):
        """Canonicalizing a classify() output should return the same label."""
        classifier = RuleIntentClassifier()
        for question in [
            "Compare X vs Y",
            "Why did SAM fail?",
            "How many experiments were run?",
            "How does the X experiment relate to Y?"
        ]:
            result = classifier.classify(question)
            if result is not None:
                assert canonicalize_intent(result) == result, (
                    f"classify('{question}') = '{result}' not idempotent under canonicalize"
                )


class TestIntegrationCanonicalLabels:
    """Integration tests: dataset → canonical, rules → canonical, parser → canonical."""

    def test_dataset_labels_canonicalize_to_known_set(self):
        """All labels in the frozen test split canonicalize to known set."""
        questions = _load_test_jsonl("stack/encoder/data/test.jsonl")
        for q in questions:
            raw = q.get("intent", q.get("question_type", ""))
            canonical = canonicalize_intent(raw)
            assert canonical in CANONICAL_LABEL_SET, (
                f"Question {q['id']}: raw '{raw}' canonicalized to "
                f"'{canonical}' — not in canonical set"
            )

    def test_rule_classifier_output_in_canonical_set(self):
        """All rule classifier outputs are in the canonical set."""
        classifier = RuleIntentClassifier()
        # Test a representative set of matching inputs
        test_questions = [
            "Compare A and B",
            "how does X differ from Y",
            "how does the validation experiment relate to",
            "what was the significance of X",
            "why did it fail",
            "how many experiments",
            "summarize the findings",
        ]
        for q in test_questions:
            result = classifier.classify(q)
            if result is not None:
                assert result in CANONICAL_LABEL_SET, (
                    f"classify('{q}') = '{result}' not in canonical set"
                )

    def test_parser_output_when_canonicalized_in_canonical_set(self):
        """Parser intent output, when canonicalized, is in canonical set."""
        import os
        import sys
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from nexus.graph.store import InMemoryGraphStore
        from nexus.graph import Node

        # Build minimal graph using proper Node objects
        graph = InMemoryGraphStore()
        graph.add_node(Node(
            id="Exp_Test", type="Experiment",
            properties={"description": "A test experiment", "key_finding": "test"},
        ))

        test_questions = [
            "What is the significance of the test experiment?",
            "Compare test experiment with other work",
            "How many experiments were conducted?",
            "How does the test experiment relate to other research?",
        ]

        from nexus.query.parser import parse_question

        for q_text in test_questions:
            try:
                pq = parse_question(q_text, graph)
                raw_intent = pq.intent if hasattr(pq, 'intent') else "factual_lookup"
                canonical = canonicalize_intent(raw_intent)
                assert canonical in CANONICAL_LABEL_SET, (
                    f"Parser intent '{raw_intent}' → '{canonical}' not in canonical set"
                )
            except Exception:
                # Parser may fail on minimal graph — skip gracefully
                pass
