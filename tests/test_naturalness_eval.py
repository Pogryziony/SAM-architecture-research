"""
Unit tests for benchmarks/naturalness_eval.py — Stage 2 naturalness metric.

Tests the five components with hand-written good/bad examples.
Also scores the current SynthesizingModel output as baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.naturalness_eval import (
    score_naturalness,
    score_naturalness_batch,
    _split_sentences,
    _score_aggregation,
    _score_connectors,
    _score_referring,
    _score_repetition,
    _score_sentence_length,
    EDGE_CONNECTORS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Sentence splitting tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSentenceSplitting:
    def test_single_sentence(self):
        result = _split_sentences("The experiment achieved 99.87% accuracy.")
        assert len(result) == 1
        assert "99.87%" in result[0]

    def test_two_sentences(self):
        result = _split_sentences(
            "The experiment achieved 99.87% accuracy. "
            "It also had high precision."
        )
        assert len(result) == 2

    def test_question_mark(self):
        result = _split_sentences("Why did this fail? The encoder was undertrained.")
        assert len(result) == 2

    def test_empty_text(self):
        result = _split_sentences("")
        assert len(result) == 0

    def test_no_punctuation(self):
        result = _split_sentences("The experiment achieved 99.87% accuracy")
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Component 1: Aggregation rate (max 25)
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregationRate:
    def test_robot_style_one_fact_per_sentence(self):
        """One fact per sentence = 0 aggregation score (robot-like)."""
        answer = (
            "Experiment X achieved 99.87% accuracy. "
            "Experiment X had 50% precision. "
            "Experiment X used 1000 examples."
        )
        score = _score_aggregation(answer, num_facts=3)
        assert score == 0.0, f"Expected 0 for 3 facts in 3 sentences, got {score}"

    def test_good_aggregation_two_facts_per_sentence(self):
        """Two facts merged into one sentence = 12.5."""
        answer = "Experiment X achieved 99.87% accuracy with 50% precision."
        score = _score_aggregation(answer, num_facts=2)
        # 2 facts / 1 sentence = 2.0. Score: 25 * (2-1)/2 = 12.5
        assert score == 12.5, f"Expected 12.5, got {score}"

    def test_excellent_aggregation_three_facts_per_sentence(self):
        """Three facts in one sentence = full 25."""
        answer = "Experiment X achieved 99.87% accuracy, 50% precision, and 90% recall."
        score = _score_aggregation(answer, num_facts=3)
        assert score == 25.0, f"Expected 25.0, got {score}"

    def test_no_facts(self):
        score = _score_aggregation("Some text.", num_facts=0)
        assert score == 0.0

    def test_single_sentence_below_threshold(self):
        """1 fact in 1 sentence = 1.0 ratio → 0 score."""
        score = _score_aggregation("Exp X achieved 99.87%.", num_facts=1)
        assert score == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Component 2: Connector presence (max 20)
# ═══════════════════════════════════════════════════════════════════════════

class TestConnectorPresence:
    def test_caused_by_connector(self):
        answer = "The experiment failed because the encoder was undertrained."
        edge_types = ["caused_by"]
        score = _score_connectors(answer, edge_types)
        assert score == 20.0, f"Expected 20 for matched caused_by, got {score}"

    def test_depends_on_connector(self):
        answer = "The chain-set retrieval, which depends on the BCE retriever, achieved high accuracy."
        edge_types = ["depends_on"]
        score = _score_connectors(answer, edge_types)
        assert score == 20.0

    def test_no_connector_when_expected(self):
        answer = "The experiment failed. The encoder was undertrained."
        edge_types = ["caused_by"]
        score = _score_connectors(answer, edge_types)
        assert score == 0.0, f"Expected 0 for missing connector, got {score}"

    def test_no_edge_types_gives_full(self):
        score = _score_connectors("Any answer.", edge_types=None)
        assert score == 20.0

    def test_empty_edge_types(self):
        score = _score_connectors("Any answer.", edge_types=[])
        assert score == 20.0

    def test_polish_connector_poniewaz(self):
        answer = "Eksperyment nie powiódł się, ponieważ enkoder był niedotrenowany."
        edge_types = ["caused_by"]
        score = _score_connectors(answer, edge_types)
        assert score == 20.0

    def test_contradicts_connector(self):
        answer = "The model improved accuracy; however, precision decreased."
        edge_types = ["contradicts"]
        score = _score_connectors(answer, edge_types)
        assert score == 20.0

    def test_multiple_edge_types_partial_match(self):
        answer = "The experiment failed because of undertraining."
        edge_types = ["caused_by", "depends_on", "validates"]
        score = _score_connectors(answer, edge_types)
        # 1 matched out of 3 expected → 20 * 1/3 ≈ 6.7
        assert 5.0 <= score <= 10.0, f"Expected partial score, got {score}"

    def test_all_edge_connectors_defined(self):
        """Every edge type in EDGE_CONNECTORS has at least one connector."""
        for etype, connectors in EDGE_CONNECTORS.items():
            assert len(connectors) > 0, f"Edge type '{etype}' has no connectors"


# ═══════════════════════════════════════════════════════════════════════════
# Component 3: Referring expressions (max 20)
# ═══════════════════════════════════════════════════════════════════════════

class TestReferringExpressions:
    def test_good_referring_full_then_short(self):
        """Full name first (with entity ID), then short reference."""
        answer = (
            "Exp_0_11_ChainRetrieval, the chain-set retrieval experiment, "
            "achieved 99.87% accuracy. This experiment also demonstrated "
            "strong recall on multi-hop queries."
        )
        score = _score_referring(answer)
        assert score == 20.0, f"Expected 20 for full+short, got {score}"

    def test_no_referring_variety(self):
        """Monolithic text without referring variety."""
        answer = "The experiment achieved 99.87% accuracy with high precision."
        score = _score_referring(answer)
        # Single sentence: neutral score 12.5
        assert score <= 12.5, f"Expected low/neutral score for single sentence, got {score}"

    def test_anaphor_without_long_form(self):
        """Has 'this' but no clear long-form introduction."""
        answer = (
            "It achieved 99.87% accuracy. This was better than the baseline. "
            "That result confirmed the hypothesis."
        )
        score = _score_referring(answer)
        # Has anaphor but no long form → 16
        assert score == 16.0, f"Expected 16, got {score}"

    def test_long_form_without_anaphor(self):
        """Has entity ID but no short-form reference."""
        answer = (
            "Exp_0_11_ChainRetrieval achieved 99.87% accuracy. "
            "Exp_0_11_ChainRetrieval also had 50% precision."
        )
        score = _score_referring(answer)
        # Has long form but no anaphor → 12
        assert score == 12.0, f"Expected 12, got {score}"

    def test_polish_anaphor(self):
        answer = (
            "Eksperyment ChainRetrieval osiągnął dokładność 99.87%. "
            "Ten eksperyment wykazał również wysoką precyzję."
        )
        score = _score_referring(answer)
        assert score == 20.0, f"Expected 20 for Polish full+short, got {score}"


# ═══════════════════════════════════════════════════════════════════════════
# Component 4: Repetition penalty (max 20)
# ═══════════════════════════════════════════════════════════════════════════

class TestRepetitionPenalty:
    def test_no_repetition(self):
        """Unique trigrams throughout → full score."""
        answer = (
            "The chain-set retrieval experiment achieved exceptional accuracy "
            "across all benchmark categories without any degradation."
        )
        score = _score_repetition(answer)
        assert score >= 18.0, f"Expected high score for no repetition, got {score}"

    def test_heavy_repetition(self):
        """Repeated identical phrases → low score."""
        answer = (
            "The experiment achieved high accuracy. "
            "The experiment achieved high accuracy. "
            "The experiment achieved high accuracy."
        )
        score = _score_repetition(answer)
        assert score < 10.0, f"Expected low score for heavy repetition, got {score}"

    def test_short_text(self):
        """Very short text → full score (no meaningful repetition possible)."""
        score = _score_repetition("High accuracy.")
        assert score == 20.0

    def test_some_repetition(self):
        """Moderate repetition → intermediate score."""
        answer = (
            "The experiment achieved 99.87% accuracy each time. "
            "The experiment achieved 99.87% accuracy consistently. "
            "The experiment achieved 99.87% accuracy repeatedly."
        )
        score = _score_repetition(answer)
        # "the experiment achieved" repeated 3 times → moderate repetition
        assert 5.0 <= score <= 18.0, f"Expected intermediate score, got {score}"


# ═══════════════════════════════════════════════════════════════════════════
# Component 5: Mean sentence length (max 15)
# ═══════════════════════════════════════════════════════════════════════════

class TestSentenceLength:
    def test_ideal_length(self):
        """A sentence with ~15 words in the 10-25 band."""
        answer = (
            "The chain set retrieval experiment achieved ninety nine point eight "
            "seven percent accuracy across all dataset splits."
        )
        score = _score_sentence_length(answer)
        # ~16 words → in band → 15
        assert score == 15.0, f"Expected 15 for in-band length, got {score}"

    def test_too_short(self):
        """A 3-word sentence — well below band."""
        score = _score_sentence_length("High accuracy achieved.")
        assert score == 0.0

    def test_too_long(self):
        """A very long sentence > 40 words."""
        words = "very " * 45 + "long."
        score = _score_sentence_length(words)
        assert score == 0.0

    def test_below_band(self):
        """6 words → (6-5)/5 * 15 = 3."""
        answer = "This experiment achieved very high accuracy."
        score = _score_sentence_length(answer)
        assert score == 3.0, f"Expected 3.0, got {score}"

    def test_above_band(self):
        """30 words → (40-30)/15 * 15 = 10."""
        words = "word " * 30
        score = _score_sentence_length(words.strip())
        assert score == 10.0, f"Expected 10.0, got {score}"

    def test_multi_sentence_average(self):
        """Average of short and normal gives intermediate."""
        answer = "Short. This is a normal length sentence with exactly ten words here."
        score = _score_sentence_length(answer)
        # Sentence 1: 1 word → 0, Sentence 2: ~10 words → just in band → 15
        # Average: ~5.5 words → (5.5-5)/5 * 15 = 1.5
        assert 0.0 <= score <= 5.0, f"Expected low-intermediate, got {score}"


# ═══════════════════════════════════════════════════════════════════════════
# Composite score tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositeScore:
    def test_full_score_with_all_components(self):
        """A well-written answer with aggregation, connectors, and referring."""
        answer = (
            "The chain-set retrieval experiment achieved 99.87% accuracy with "
            "50% precision, validating the NEXUS architecture assumptions "
            "because the BCE retriever provided strong candidates. "
            "This experiment also confirmed prior hypotheses."
        )
        facts = [
            "chain-set retrieval achieved 99.87% accuracy",
            "chain-set retrieval achieved 50% precision",
            "BCE retriever provided strong candidates",
        ]
        edge_types = ["caused_by", "validates"]
        result = score_naturalness(answer, facts, edge_types)
        assert "total" in result
        assert result["total"] >= 50.0, f"Expected >= 50 for good answer, got {result['total']}"
        assert result["aggregation"] > 0  # 3 facts in 2 sentences > 1:1
        assert result["connectors"] > 0
        assert result["referring"] > 10

    def test_robot_answer_low_score(self):
        """Robot-like bullet style should score low."""
        answer = (
            "Experiment X achieved 99.87% accuracy. "
            "Experiment X achieved 50% precision. "
            "Experiment X achieved 90% recall. "
            "Experiment X achieved 95% F1."
        )
        facts = [
            "accuracy 99.87%",
            "precision 50%",
            "recall 90%",
            "F1 95%",
        ]
        # Pass edge types to penalize missing connectors in robot output
        result = score_naturalness(answer, facts, edge_types=["caused_by", "validates"])
        assert result["total"] < 40.0, f"Expected < 40 for robot answer, got {result['total']}"
        assert result["aggregation"] == 0.0  # 4 facts, 4 sentences = 1:1

    def test_no_facts_no_edges(self):
        """Answer without facts metadata still gets non-zero from other components."""
        answer = (
            "The experiment demonstrated that the proposed architecture "
            "outperforms existing baselines by a significant margin."
        )
        result = score_naturalness(answer)
        assert result["total"] > 0
        assert result["aggregation"] == 0.0  # No facts

    def test_batch_scoring(self):
        answers = [
            "The experiment achieved 99.87% accuracy.",
            "Good result but needs more testing because the data was noisy.",
        ]
        result = score_naturalness_batch(answers)
        assert result["n"] == 2
        assert "mean_total" in result
        assert "component_means" in result

    def test_empty_batch(self):
        result = score_naturalness_batch([])
        assert result["n"] == 0
        assert result["mean_total"] == 0.0

    def test_total_capped_at_100(self):
        """Score should never exceed 100."""
        # Craft an answer that would score well on everything
        answer = (
            "The chain-set retrieval experiment (Exp_0_11_ChainRetrieval) "
            "achieved 99.87% accuracy, 95% precision, and 98% recall across "
            "all datasets, because the BCE-based retriever provided strong "
            "candidate selection. This experiment validated the architecture, "
            "which depends on the multi-hop retrieval pipeline. It also "
            "confirmed prior hypotheses about memory efficiency and "
            "demonstrated consistent performance improvements."
        )
        result = score_naturalness(answer, facts=["f1", "f2", "f3"], edge_types=["caused_by", "depends_on", "validates"])
        assert result["total"] <= 100.0, f"Score capped at 100, got {result['total']}"


# ═══════════════════════════════════════════════════════════════════════════
# Baseline: Score current SynthesizingModel output
# ═══════════════════════════════════════════════════════════════════════════

class TestCurrentSynthesizingModelBaseline:
    """Score the current SynthesizingModel style to establish baseline."""

    def test_typical_synthesizer_output_style(self):
        """Typical SynthesizingModel output: one sentence, no connectors,
        no referring variety, short length."""
        # This matches the typical output format of the current SynthesizingModel
        # which produces single-sentence answers like:
        # "The Exp_0_6 achieved 99.87% accuracy."
        # Even with edge types present, current output uses no connectors.
        answer = "The Exp_0_6_Validation achieved 99.87% accuracy."
        result = score_naturalness(
            answer,
            facts=["accuracy 99.87%"],
            edge_types=["caused_by", "validates"],
        )
        # Single sentence, minimal features → low score
        assert result["total"] < 45.0, (
            f"Current synthesizer baseline should be low (<45), got {result['total']}. "
            f"Components: agg={result['aggregation']}, conn={result['connectors']}, "
            f"ref={result['referring']}, rep={result['repetition']}, "
            f"slen={result['sentence_length']}"
        )

    def test_baseline_known_low_aggregation(self):
        """Current synthesizer rarely aggregates multiple facts."""
        answer = "The Exp_0_6 achieved 99.87% accuracy."
        result = score_naturalness(answer, facts=["accuracy 99.87%", "precision 50%"])
        # 2 facts, 1 sentence → aggregation (2-1)/2 * 25 = 12.5
        # But referring will be neutral and connectors absent
        assert result["aggregation"] <= 12.5
        assert result["connectors"] <= 20.0  # No edge info = full credit by default

    def test_baseline_no_connectors_in_output(self):
        """Current synthesizer doesn't use edge-type connectors in factual answers."""
        answer = "The Exp_0_6_Validation achieved 99.87% accuracy."
        # Even with edge types, there are no connectors in the answer
        result = score_naturalness(answer, facts=["f1"], edge_types=["caused_by", "validates"])
        # Connectors should be low — answer has no "because" or "validating"
        assert result["connectors"] <= 5.0, (
            f"Expected low connectors for no-connector answer, got {result['connectors']}"
        )
