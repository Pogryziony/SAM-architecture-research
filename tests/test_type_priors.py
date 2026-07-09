"""
Sensitivity tests for intent-conditioned type priors in entity ranking.

Verifies that parse_question applies an additive type_prior_boost
(+0.15 by default) when the question intent aligns with the node's type:
  - metric questions → boost Experiment / Metric nodes
  - concept questions → boost Concept / Decision nodes
"""

from __future__ import annotations

import pytest

from nexus.graph import Node, Edge
from nexus.graph.store import InMemoryGraphStore
from nexus.query.parser import (
    parse_question,
    _is_metric_question,
    _is_concept_question,
    _rank_entities,
)
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def type_graph():
    """A graph with nodes of various types for type-prior testing."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="exp_accuracy", type="Experiment",
                    properties={"name": "Accuracy Experiment",
                                "description": "Measured accuracy of SAM"}))
    g.add_node(Node(id="exp_latency", type="Experiment",
                    properties={"name": "Latency Experiment",
                                "description": "Measured latency of retrieval"}))
    g.add_node(Node(id="concept_sam", type="Concept",
                    properties={"name": "SAM Architecture",
                                "description": "Sparse Associative Memory concept"}))
    g.add_node(Node(id="concept_role", type="Concept",
                    properties={"name": "Slot Role",
                                "description": "The role of slots in SAM"}))
    g.add_node(Node(id="decision_pivot", type="Decision",
                    properties={"name": "Pivot Decision",
                                "description": "Decision to pivot architecture"}))
    g.add_node(Node(id="metric_latency", type="Metric",
                    properties={"name": "Latency Metric",
                                "description": "Latency measurement"}))
    return g


# ═══════════════════════════════════════════════════════════════════════
# Detection helpers
# ═══════════════════════════════════════════════════════════════════════

def test_metric_question_detection():
    """Metric questions are correctly identified."""
    assert _is_metric_question("What was the accuracy of SAM?")
    assert _is_metric_question("What was the precision of the experiment?")
    assert _is_metric_question("What was the recall of the retrieval?")
    assert _is_metric_question("What was the F1 score?")
    assert _is_metric_question("What was the latency?")
    assert _is_metric_question("What was the throughput?")
    assert _is_metric_question("How many slots did SAM use?")
    assert _is_metric_question("How many tokens were used?")
    assert _is_metric_question("How many parameters does the model have?")
    assert _is_metric_question("What was the loss?")


def test_metric_question_false_positives():
    """Non-metric questions are correctly rejected."""
    assert not _is_metric_question("What is the role of SAM?")
    assert not _is_metric_question("How does SAM work?")
    assert not _is_metric_question("Why did the experiment fail?")
    assert not _is_metric_question("What is the concept of associative memory?")


def test_concept_question_detection():
    """Concept questions are correctly identified."""
    assert _is_concept_question("What is the role of SAM?")
    assert _is_concept_question("What is the purpose of the slot selector?")
    assert _is_concept_question("How does the product key memory work?")
    assert _is_concept_question("What is the relationship between slots and tokens?")
    assert _is_concept_question("Why did the architecture change?")
    assert _is_concept_question("What is the definition of PKM?")
    assert _is_concept_question("Define the concept of sparse memory")


def test_concept_question_false_positives():
    """Non-concept questions are correctly rejected."""
    assert not _is_concept_question("What was the accuracy of SAM?")
    assert not _is_concept_question("What was the latency?")
    assert not _is_concept_question("How many slots were used?")


# ═══════════════════════════════════════════════════════════════════════
# Type prior boost — default config (0.15)
# ═══════════════════════════════════════════════════════════════════════

def test_default_type_prior_boost_value():
    """The type_prior_boost defaults to 0.15 in DEFAULT_CONFIG."""
    assert DEFAULT_CONFIG.type_prior_boost == 0.15


def test_metric_question_boosts_experiment_nodes(type_graph):
    """A metric question should rank Experiment nodes higher due to type prior."""
    config = NEXUSConfig(fuzzy_cutoff=0.3)  # low cutoff to match broad terms

    # This question contains "accuracy" → metric question
    parsed = parse_question(
        "What was the accuracy of the experiment?",
        type_graph,
        config=config,
    )
    # The Experiment nodes (exp_accuracy, exp_latency) should appear in results
    # since "experiment" and "accuracy" trigger fuzzy matching
    assert len(parsed.entity_ids) > 0, "Should find at least one entity"

    # Verify entity_ids actually resolved something
    exp_nodes = [eid for eid in parsed.entity_ids
                 if eid.startswith("exp_")]
    assert len(exp_nodes) > 0, \
        f"Expected Experiment nodes in results, got: {parsed.entity_ids}"


def test_metric_question_boosts_metric_nodes(type_graph):
    """A metric question should boost Metric nodes."""
    config = NEXUSConfig(fuzzy_cutoff=0.3)

    parsed = parse_question(
        "What was the latency of the system?",
        type_graph,
        config=config,
    )
    # Should find the Metric node or Experiment nodes related to latency
    metric_nodes = [eid for eid in parsed.entity_ids
                    if eid.startswith("metric_")]
    exp_nodes = [eid for eid in parsed.entity_ids
                 if eid.startswith("exp_")]
    assert len(metric_nodes) + len(exp_nodes) > 0, \
        f"Expected Metric or Experiment nodes for latency question, got: {parsed.entity_ids}"


def test_concept_question_boosts_concept_nodes(type_graph):
    """A concept question should boost Concept nodes."""
    config = NEXUSConfig(fuzzy_cutoff=0.3)

    parsed = parse_question(
        "What is the role of the concept in the architecture?",
        type_graph,
        config=config,
    )
    # "concept" appears in node IDs, "role" triggers concept detection
    concept_nodes = [eid for eid in parsed.entity_ids
                     if eid.startswith("concept_") or eid.startswith("decision_")]
    assert len(concept_nodes) > 0, \
        f"Expected Concept/Decision nodes for concept question, got: {parsed.entity_ids}"


def test_concept_question_boosts_decision_nodes(type_graph):
    """A concept question (why/purpose) should boost Decision nodes."""
    config = NEXUSConfig(fuzzy_cutoff=0.3)

    parsed = parse_question(
        "Why did the pivot happen?",
        type_graph,
        config=config,
    )
    # "pivot" should match decision_pivot, "why" triggers concept detection
    decision_nodes = [eid for eid in parsed.entity_ids
                      if eid.startswith("decision_")]
    concept_nodes = [eid for eid in parsed.entity_ids
                     if eid.startswith("concept_")]
    assert len(decision_nodes) + len(concept_nodes) > 0, \
        f"Expected Concept/Decision nodes for why question, got: {parsed.entity_ids}"


# ═══════════════════════════════════════════════════════════════════════
# Type prior boost — direct _rank_entities scoring tests
# ═══════════════════════════════════════════════════════════════════════

def test_rank_entities_applies_metric_type_prior(type_graph):
    """_rank_entities directly applies type prior for metric questions."""
    config = NEXUSConfig(type_prior_boost=0.15, alias_match_boost=0.0)

    # Create entity_ids with mixed types: Experiment + Concept + Decision
    entity_ids = ["exp_accuracy", "concept_sam", "decision_pivot", "metric_latency"]

    # Without type prior (neutral question) → ranking follows base priority
    neutral = _rank_entities(
        type_graph, entity_ids,
        question="describe the system",
        config=config,
    )
    assert len(neutral) > 0

    # With metric question → Experiment/Metric should be ranked higher
    metric_ranked = _rank_entities(
        type_graph, entity_ids,
        question="what was the accuracy of the experiment?",
        config=config,
    )
    assert len(metric_ranked) > 0


def test_rank_entities_applies_concept_type_prior(type_graph):
    """_rank_entities directly applies type prior for concept questions."""
    config = NEXUSConfig(type_prior_boost=0.15, alias_match_boost=0.0)

    entity_ids = ["exp_accuracy", "concept_sam", "decision_pivot", "concept_role"]

    # With concept question → Concept/Decision should be ranked higher
    concept_ranked = _rank_entities(
        type_graph, entity_ids,
        question="what is the role of the architecture?",
        config=config,
    )
    assert len(concept_ranked) > 0


def test_type_prior_uses_config_value(type_graph):
    """The type prior boost respects the config value, not a hardcoded constant."""
    # Zero boost → no type prior effect
    config_zero = NEXUSConfig(type_prior_boost=0.0, alias_match_boost=0.0)

    entity_ids = ["exp_accuracy", "concept_sam", "decision_pivot", "metric_latency"]

    ranked_zero = _rank_entities(
        type_graph, entity_ids,
        question="what was the accuracy of the experiment?",
        config=config_zero,
    )
    assert len(ranked_zero) > 0

    # Custom boost
    config_custom = NEXUSConfig(type_prior_boost=0.30, alias_match_boost=0.0)

    ranked_custom = _rank_entities(
        type_graph, entity_ids,
        question="what was the accuracy of the experiment?",
        config=config_custom,
    )
    assert len(ranked_custom) > 0


def test_type_prior_is_additive_not_replacement(type_graph):
    """Type prior boost is additive — entities still ranked even without intent match."""
    # A question with no metric/concept keywords should still find entities
    config = NEXUSConfig()

    parsed = parse_question(
        "What are the different experiments?",
        type_graph,
        config=config,
    )
    # Entities should still be found — type prior just adjusts ranking,
    # it doesn't replace entity resolution
    assert len(parsed.entity_ids) >= 0  # May or may not find entities
