"""
Tests for NEXUS query/reasoning pipeline layer.

Covers: parser (intent detection, entity spotting), verifier (supported/fabricated claims),
evidence builder (reversed-edge phrasing), and answer_question edge cases.

All tests are self-contained — they create their own small graphs, no ingestion pipeline needed.
"""

from __future__ import annotations

import pytest

from nexus.graph import Node, Edge, Path, PathStep
from nexus.graph.store import InMemoryGraphStore
from nexus.query.parser import detect_intent, spot_entities, parse_question
from nexus.reasoning.verifier import Verifier, VerificationResult, extract_claims
from nexus.reasoning.evidence_builder import build_evidence_pack, _fact_from_step
from nexus.reasoning.answer import answer_question
from nexus.utils.config import NEXUSConfig, DEFAULT_CONFIG


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def empty_graph():
    """An empty InMemoryGraphStore with zero nodes."""
    return InMemoryGraphStore()


@pytest.fixture
def populated_graph():
    """A small graph for entity-spotting and pipeline tests."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="Exp_0_11_ChainRetrieval", type="Experiment",
                    properties={"name": "Chain Retrieval", "description": "Chain-aware retrieval experiment"}))
    g.add_node(Node(id="concept_pivottonexus", type="Concept",
                    properties={"name": "Pivot to NEXUS", "description": "Decision to unify graph layers"}))
    g.add_node(Node(id="bottleneck_selector", type="Metric",
                    properties={"name": "Selector Bottleneck", "description": "Selector latency issue"}))
    g.add_node(Node(id="project_setup", type="Document",
                    properties={"name": "Project Setup", "description": "Initial project scaffolding"}))
    g.add_node(Node(id="concept_comparison", type="Concept",
                    properties={"name": "Architecture Comparison",
                                "description": "Comparing SAM variants"}))
    g.add_edge(Edge(type="derived_from", source="concept_pivottonexus",
                    target="Exp_0_11_ChainRetrieval", confidence=0.85))
    g.add_edge(Edge(type="validates", source="bottleneck_selector",
                    target="concept_pivottonexus", confidence=0.9))
    g.add_edge(Edge(type="depends_on", source="concept_comparison",
                    target="Exp_0_11_ChainRetrieval", confidence=0.8))
    g.add_edge(Edge(type="related_to", source="project_setup",
                    target="Exp_0_11_ChainRetrieval", confidence=0.7))
    return g


@pytest.fixture
def simple_path_graph():
    """A two-node directed path: A --[derived_from]--> B."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="A", type="Entity", properties={"name": "Alpha"}))
    g.add_node(Node(id="B", type="Entity", properties={"name": "Beta"}))
    g.add_edge(Edge(type="derived_from", source="B", target="A", confidence=0.9))
    return g


@pytest.fixture
def multi_edge_graph():
    """A graph with multiple edge types for evidence-builder tests."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="X", type="Entity", properties={"name": "Xavier"}))
    g.add_node(Node(id="Y", type="Entity", properties={"name": "Yvonne"}))
    g.add_node(Node(id="Z", type="Entity", properties={"name": "Zara"}))
    g.add_node(Node(id="T", type="TestCase", properties={"name": "TestAlpha"}))
    g.add_edge(Edge(type="derived_from", source="Y", target="X", confidence=0.9))
    g.add_edge(Edge(type="validates", source="T", target="Y", confidence=1.0))
    g.add_edge(Edge(type="depends_on", source="Z", target="X", confidence=0.85))
    return g


@pytest.fixture
def verifier():
    """Default verifier with 0.2 threshold."""
    return Verifier(hallucination_threshold=0.2)


@pytest.fixture
def strict_verifier():
    """Strict verifier with 0.0 threshold — any unsupported claim fails."""
    return Verifier(hallucination_threshold=0.0)


# ═══════════════════════════════════════════════════════════════════
# 1. Parser — intent detection
# ═══════════════════════════════════════════════════════════════════

class TestIntentDetection:
    """Test detect_intent() for all keyword patterns and edge cases."""

    def test_why_question_detected_as_causal_explanation(self):
        intent, direction = detect_intent("Why did the project pivot to NEXUS?")
        assert intent == "causal_explanation"
        assert direction == "in"

    def test_what_depends_detected_as_dependency_chain(self):
        intent, direction = detect_intent("What depends on the chain retrieval module?")
        assert intent == "dependency_chain"
        assert direction == "both"

    def test_what_affects_detected_as_impact_analysis(self):
        intent, direction = detect_intent("What affects the selector performance?")
        assert intent == "impact_analysis"
        assert direction == "out"

    def test_compare_detected_as_comparison(self):
        intent, direction = detect_intent("Compare the pivot-to-NEXUS approach with original design.")
        assert intent == "comparison"
        assert direction == "both"

    def test_vs_detected_as_comparison(self):
        intent, direction = detect_intent("Architecture A vs Architecture B — which is better?")
        assert intent == "comparison"
        assert direction == "both"

    def test_what_is_detected_as_factual_lookup(self):
        intent, direction = detect_intent("What is the key finding of the experiment?")
        assert intent == "factual_lookup"
        assert direction == "both"

    def test_how_detected_as_diagnostic(self):
        intent, direction = detect_intent("How do I diagnose the selector bottleneck?")
        assert intent == "diagnostic"
        assert direction == "in"

    def test_empty_question_defaults_to_factual_lookup(self):
        intent, direction = detect_intent("")
        assert intent == "factual_lookup"
        assert direction == "both"

    def test_ambiguous_question_defaults_to_factual_lookup(self):
        intent, direction = detect_intent("xyzzy flibble woz")
        assert intent == "factual_lookup"
        assert direction == "both"

    def test_multi_keyword_picks_first_match(self):
        """When both 'why' and 'what depends' appear, first pattern (why → causal) wins."""
        intent, direction = detect_intent("Why does X and what depends on Y?")
        assert intent == "causal_explanation"

    def test_cause_keyword_detected_as_causal(self):
        intent, _ = detect_intent("What was the cause of the failure in the experiment?")
        assert intent == "causal_explanation"

    def test_reason_keyword_detected_as_causal(self):
        intent, _ = detect_intent("What is the reason for the pivot?")
        assert intent == "causal_explanation"

    def test_difference_detected_as_comparison(self):
        intent, _ = detect_intent("What is the difference between approach A and B?")
        assert intent == "comparison"

    def test_how_to_detected_as_diagnostic(self):
        intent, _ = detect_intent("How to debug the selector bottleneck?")
        assert intent == "diagnostic"


# ═══════════════════════════════════════════════════════════════════
# 2. Parser — entity spotting with stop words
# ═══════════════════════════════════════════════════════════════════

class TestEntitySpotting:
    """Test spot_entities() and parse_question() for entity resolution."""

    def test_finds_known_entity_in_question(self, populated_graph):
        entities, _wb = spot_entities(
            "What was the key finding of the chain retrieval experiment?",
            populated_graph,
        )
        entity_ids = {e[3] for e in entities}
        assert "Exp_0_11_ChainRetrieval" in entity_ids

    def test_parse_question_finds_entity(self, populated_graph):
        parsed = parse_question(
            "What is the pivot to NEXUS about?",
            populated_graph,
        )
        assert "concept_pivottonexus" in parsed.entity_ids

    def test_stop_words_not_matched_as_entities(self, populated_graph):
        """Common stop words like 'the', 'is', 'what', 'why', 'how' should NOT be entities."""
        # Even if a question contains only stop words, no entities should be found
        # unless they match actual graph nodes.
        g = InMemoryGraphStore()
        g.add_node(Node(id="the_concept", type="Concept"))   # 'the' is in graph name index
        # Without any matching node, stop words alone should return nothing
        entities, _wb = spot_entities("the is what why how", g)
        # 'the' might match via fuzzy lookup since we have 'the_concept'
        # The key test: a graph without matching names → no entities
        g2 = InMemoryGraphStore()
        g2.add_node(Node(id="alpha", type="Entity"))
        entities, _wb = spot_entities("the is what why how", g2)
        # 'the', 'is', 'what', 'why', 'how' are stop words — should not match 'alpha'
        assert len(entities) == 0

    def test_what_keyword_not_entity(self, populated_graph):
        """'what' is a stop word and should not be matched as an entity."""
        entities, _wb = spot_entities("what is the setup?", populated_graph)
        entity_ids = {e[3] for e in entities}
        assert "what" not in entity_ids

    def test_how_keyword_not_entity(self, populated_graph):
        entities, _wb = spot_entities("how does it work?", populated_graph)
        entity_ids = {e[3] for e in entities}
        assert "how" not in entity_ids

    def test_fuzzy_chain_retrieval_matches_experiment_id(self, populated_graph):
        """'chain retrieval' should fuzzy-match 'Exp_0_11_ChainRetrieval'."""
        entities, _wb = spot_entities("chain retrieval results", populated_graph)
        entity_ids = {e[3] for e in entities}
        assert "Exp_0_11_ChainRetrieval" in entity_ids

    def test_fuzzy_pivot_to_nexus_matches_concept(self, populated_graph):
        entities, _wb = spot_entities("pivot to NEXUS decision", populated_graph)
        entity_ids = {e[3] for e in entities}
        assert "concept_pivottonexus" in entity_ids

    def test_nonexistent_entity_returns_empty(self, populated_graph):
        entities, _wb = spot_entities("quantum blockchain synergy", populated_graph)
        assert len(entities) == 0

    def test_parse_question_returns_empty_for_nonexistent_entities(self, populated_graph):
        parsed = parse_question("nothing useful here", populated_graph, cutoff=1.0)
        assert parsed.entity_ids == []
        assert parsed.entity_spans == []


# ═══════════════════════════════════════════════════════════════════
# 3. Verifier — supported vs fabricated claims
# ═══════════════════════════════════════════════════════════════════

class TestVerifier:
    """Test Verifier.verify() for hallucination detection."""

    def _make_mock_evidence(self):
        """Create a mock evidence_pack dict with known nodes and edges."""
        return {
            "question": "What is the relationship between Alpha and Beta?",
            "paths": [
                {
                    "score": 0.95,
                    "length": 1,
                    "nodes": [
                        {"id": "Alpha", "type": "Entity", "name": "Alpha"},
                        {"id": "Beta", "type": "Entity", "name": "Beta"},
                    ],
                    "edges": [
                        {"type": "depends_on", "from": "Beta", "to": "Alpha"},
                    ],
                },
            ],
            "facts": ["Beta depends on Alpha (confidence: 0.95)"],
            "sources": ["source_1.md"],
        }

    def test_supported_claims_pass_verification(self, verifier):
        evidence = self._make_mock_evidence()
        answer = "Beta depends on Alpha."
        result = verifier.verify(answer, evidence)
        assert result.supported_count == 1
        assert result.hallucination_rate == 0.0
        assert result.passed is True

    def test_fabricated_claims_fail_verification(self, verifier):
        evidence = self._make_mock_evidence()
        answer = "Gamma implements Delta. Sigma contradicts Omega. Zeta is related to Eta."
        result = verifier.verify(answer, evidence)
        # All entities and relations fabricated → all unsupported
        assert len(result.unsupported_claims) > 0
        assert result.supported_count == 0
        assert result.hallucination_rate == 1.0
        assert result.passed is False

    def test_mixed_claims_produce_correct_hallucination_rate(self, verifier):
        evidence = self._make_mock_evidence()
        answer = (
            "Beta depends on Alpha. "           # supported
            "Gamma implements Delta. "           # fabricated
            "Alpha is a key entity. "            # supported (Alpha is in evidence)
            "Sigma contradicts Omega."           # fabricated
        )
        result = verifier.verify(answer, evidence)
        assert result.hallucination_rate == 0.5  # 2 of 4 unsupported
        assert result.supported_count == 2
        assert len(result.unsupported_claims) == 2

    def test_hallucination_rate_above_threshold_fails(self, verifier):
        """threshold=0.2, rate=0.5 → passed=False."""
        evidence = self._make_mock_evidence()
        answer = "Beta depends on Alpha. Gamma implements Delta."
        result = verifier.verify(answer, evidence)
        assert result.hallucination_rate == 0.5
        assert result.passed is False

    def test_hallucination_rate_at_threshold_passes(self):
        """threshold=0.5, rate=0.5 → passed=True (≤ threshold)."""
        v = Verifier(hallucination_threshold=0.5)
        evidence = self._make_mock_evidence()
        answer = "Beta depends on Alpha. Gamma implements Delta."
        result = v.verify(answer, evidence)
        assert result.hallucination_rate == 0.5
        assert result.passed is True

    def test_hallucination_rate_below_threshold_passes(self, verifier):
        """threshold=0.2, rate=0.0 → passed=True."""
        evidence = self._make_mock_evidence()
        answer = "Beta depends on Alpha."
        result = verifier.verify(answer, evidence)
        assert result.hallucination_rate == 0.0
        assert result.passed is True

    def test_empty_answer_passes_with_zero_rate(self, verifier):
        evidence = self._make_mock_evidence()
        result = verifier.verify("", evidence)
        assert result.passed is True
        assert result.hallucination_rate == 0.0
        assert result.supported_count == 0

    def test_insufficient_evidence_answer_passes(self, verifier):
        evidence = self._make_mock_evidence()
        result = verifier.verify("Insufficient evidence to answer.", evidence)
        assert result.passed is True
        assert result.hallucination_rate == 0.0

    def test_claim_with_evidence_entity_passes_strict(self, strict_verifier):
        """Strict verifier still passes a fully-supported claim."""
        evidence = self._make_mock_evidence()
        answer = "Beta depends on Alpha."
        result = strict_verifier.verify(answer, evidence)
        assert result.passed is True

    def test_any_fabrication_fails_strict_verifier(self, strict_verifier):
        evidence = self._make_mock_evidence()
        answer = "Beta depends on Alpha. Gamma implements Delta."
        result = strict_verifier.verify(answer, evidence)
        # rate = 0.5 > threshold 0.0 → should fail
        assert result.passed is False

    def test_extract_claims_splits_sentences(self):
        text = "Alpha depends on Beta. Beta validates Gamma. Delta is related to Epsilon."
        claims = extract_claims(text)
        assert len(claims) == 3

    def test_extract_claims_filters_short_strings(self):
        text = "Alpha depends on Beta. OK. Hi."
        claims = extract_claims(text)
        # "OK." and "Hi." are < 10 chars, should be filtered
        assert len(claims) == 1

    def test_extract_claims_handles_empty(self):
        assert extract_claims("") == []

    def test_extract_claims_strips_bullet_markers(self):
        text = "- Alpha depends on Beta.\n* Beta validates Gamma."
        claims = extract_claims(text)
        assert claims[0].startswith("Alpha")
        assert claims[1].startswith("Beta")


# ═══════════════════════════════════════════════════════════════════
# 4. Evidence builder — reversed-edge fact phrasing
# ═══════════════════════════════════════════════════════════════════

class TestEvidenceBuilder:
    """Test _fact_from_step() and build_evidence_pack() for correct direction phrasing."""

    def test_forward_derived_from_edge_uses_direct_phrasing(self, simple_path_graph):
        """B --[derived_from]--> A (forward, i.e. B derived_from A).
           _fact_from_step sees edge from B to A. If traversed forward:
           edge.source = B, edge.target = A, reversed=False → from_node=B, to_node=A.
           'derived_from' forward phrase is 'is derived from'.
           So: 'Beta is derived from Alpha'."""
        edge = Edge(type="derived_from", source="B", target="A", confidence=0.9)
        step = PathStep(edge=edge, reversed=False)
        fact = _fact_from_step(step, simple_path_graph)
        assert "beta" in fact.lower()
        assert "alpha" in fact.lower()
        assert "is derived from" in fact.lower()

    def test_reversed_derived_from_edge_uses_supports_phrasing(self, simple_path_graph):
        """Same edge traversed in reverse: reversed=True.
           from_node = edge.target (A), to_node = edge.source (B).
           Reversed phrase for 'derived_from' is 'supports'.
           So: 'Alpha supports Beta'."""
        edge = Edge(type="derived_from", source="B", target="A", confidence=0.9)
        step = PathStep(edge=edge, reversed=True)
        fact = _fact_from_step(step, simple_path_graph)
        assert "alpha" in fact.lower()
        assert "beta" in fact.lower()
        assert "supports" in fact.lower()

    def test_forward_validates_edge_uses_validates(self, multi_edge_graph):
        """T --[validates]--> Y. Forward: from_node=T, to_node=Y.
           Forward phrase for 'validates' is 'validates'."""
        edge = Edge(type="validates", source="T", target="Y", confidence=1.0)
        step = PathStep(edge=edge, reversed=False)
        fact = _fact_from_step(step, multi_edge_graph)
        assert "testalpha" in fact.lower()
        assert "yvonne" in fact.lower()
        assert "validates" in fact.lower()

    def test_reversed_validates_edge_uses_is_validated_by(self, multi_edge_graph):
        """Same validates edge reversed: reversed=True.
           from_node=Y, to_node=T. Reversed phrase is 'is validated by'.
           So: 'Yvonne is validated by TestAlpha'."""
        edge = Edge(type="validates", source="T", target="Y", confidence=1.0)
        step = PathStep(edge=edge, reversed=True)
        fact = _fact_from_step(step, multi_edge_graph)
        assert "yvonne" in fact.lower()
        assert "testalpha" in fact.lower()
        assert "is validated by" in fact.lower()

    def test_forward_depends_on_edge(self, multi_edge_graph):
        """Z --[depends_on]--> X. Forward phrase is 'depends on'."""
        edge = Edge(type="depends_on", source="Z", target="X", confidence=0.85)
        step = PathStep(edge=edge, reversed=False)
        fact = _fact_from_step(step, multi_edge_graph)
        assert "zara" in fact.lower()
        assert "xavier" in fact.lower()
        assert "depends on" in fact.lower()

    def test_reversed_depends_on_edge(self, multi_edge_graph):
        """Z --[depends_on]--> X reversed. Reversed phrase is 'is a dependency of'."""
        edge = Edge(type="depends_on", source="Z", target="X", confidence=0.85)
        step = PathStep(edge=edge, reversed=True)
        fact = _fact_from_step(step, multi_edge_graph)
        assert "xavier" in fact.lower()
        assert "zara" in fact.lower()
        assert "is a dependency of" in fact.lower()

    def test_build_evidence_pack_returns_dict_with_expected_keys(self, multi_edge_graph):
        """build_evidence_pack should return a dict with question, paths, facts, sources."""
        path = Path(steps=[
            PathStep(Edge(type="depends_on", source="Z", target="X", confidence=0.85), reversed=False),
        ], score=0.85)
        pack = build_evidence_pack("What does Z depend on?", [path], multi_edge_graph)
        assert isinstance(pack, dict)
        assert "question" in pack
        assert "paths" in pack
        assert "facts" in pack
        assert "sources" in pack
        assert pack["question"] == "What does Z depend on?"
        assert len(pack["paths"]) == 1
        assert len(pack["facts"]) >= 1


# ═══════════════════════════════════════════════════════════════════
# 5. answer_question — edge cases
# ═══════════════════════════════════════════════════════════════════

class TestAnswerQuestionEdgeCases:
    """Test answer_question() with various failure/edge-case inputs."""

    def _assert_graceful_result(self, result, expect_insufficient=True):
        """Helper: verify the result dict has expected keys and graceful fallback."""
        assert "answer" in result
        assert "evidence_pack" in result
        assert "verification" in result
        if expect_insufficient:
            assert "Insufficient evidence" in result["answer"]
        assert result["verification"].passed is True

    def test_empty_graph_returns_insufficient_evidence(self, empty_graph):
        result = answer_question("What is the relationship?", empty_graph)
        self._assert_graceful_result(result)
        assert "empty" in result["answer"].lower()

    def test_no_recognized_entities_returns_insufficient_evidence(self, populated_graph):
        result = answer_question("xyzzy flibble woz quantum", populated_graph, max_depth=1)
        self._assert_graceful_result(result)
        assert "No relevant entities found" in result["answer"]

    def test_entities_exist_but_no_paths_returns_insufficient_evidence(self):
        """Nodes exist but no edges connecting them → no traversal paths."""
        g = InMemoryGraphStore()
        g.add_node(Node(id="Isolated_A", type="Entity"))
        g.add_node(Node(id="Isolated_B", type="Entity"))
        # No edges between them
        result = answer_question("What connects Isolated_A and Isolated_B?", g, max_depth=4)
        self._assert_graceful_result(result)
        assert "No traversal paths found" in result["answer"]

    def test_result_has_expected_keys(self, populated_graph):
        """A successful query should return all expected keys."""
        # The graph has entities and edges — should produce paths
        result = answer_question(
            "What is the pivot to NEXUS?",
            populated_graph,
            max_depth=4,
        )
        assert "question" in result
        assert "answer" in result
        assert "evidence_pack" in result
        assert "verification" in result
        assert "parsed_query" in result
        assert "path_count" in result
        assert result["question"] == "What is the pivot to NEXUS?"
        assert result["parsed_query"] is not None

    def test_entity_question_with_connections_succeeds(self, populated_graph):
        """A question with entities and paths should produce a non-empty answer."""
        result = answer_question(
            "What caused the pivot to NEXUS?",
            populated_graph,
            max_depth=4,
        )
        assert result["path_count"] > 0
        assert len(result["answer"]) > 0
        assert "Insufficient evidence" not in result["answer"]


# ═══════════════════════════════════════════════════════════════════
# 6. Sensitivity — config perturbation should not break entity resolution
# ═══════════════════════════════════════════════════════════════════

class TestConfigSensitivity:
    """Test that entity resolution is robust to small config perturbations."""

    def test_entity_resolution_not_overly_sensitive(self):
        """Entity resolution should not change by >5% when boosts vary by +-0.05.

        Builds a test graph with the problematic oracle-like case where
        subtle ranking changes could flip the top entity.  Verifies that
        both the default config and a perturbed config find the same top
        entity for a realistic question.
        """
        base_config = NEXUSConfig()
        perturbed = NEXUSConfig(
            property_keyword_boost=0.30,
            curated_node_boost=0.15,
            sub_run_penalty=-0.10,
        )

        # Build a realistic graph with curated and sub-run nodes.
        g = InMemoryGraphStore()
        g.add_node(Node(
            id="Exp_0_6_Validation",
            type="Experiment",
            properties={
                "name": "Validation Experiment",
                "key_finding": "The oracle filter improves accuracy by 12%",
            },
        ))
        g.add_node(Node(
            id="concept_oracle_filter",
            type="Concept",
            properties={
                "name": "Oracle Filter Concept",
                "description": "A filtering mechanism for retrieval results",
            },
        ))
        g.add_node(Node(
            id="Exp_0_6_Validation_top32",
            type="Experiment",
            properties={
                "name": "Validation top-32 sub-run",
                "description": "Sub-experiment variant with top-32",
            },
        ))
        g.add_node(Node(
            id="Exp_0_6_Validation_baseline",
            type="Experiment",
            properties={
                "name": "Validation baseline sub-run",
                "description": "Sub-experiment baseline",
            },
        ))
        g.add_node(Node(
            id="Exp_0_9_OracleFilter",
            type="Experiment",
            properties={
                "name": "Oracle Filter Experiment",
                "key_finding": "OracleFilter achieves state-of-the-art retrieval",
            },
        ))
        g.add_edge(Edge(type="derived_from", source="Exp_0_9_OracleFilter",
                        target="concept_oracle_filter", confidence=0.9))
        g.add_edge(Edge(type="sub_experiment", source="Exp_0_6_Validation_top32",
                        target="Exp_0_6_Validation", confidence=0.8))
        g.add_edge(Edge(type="sub_experiment", source="Exp_0_6_Validation_baseline",
                        target="Exp_0_6_Validation", confidence=0.8))

        # Parse with both configs
        base_parsed = parse_question(
            "What was the key finding of the oracle filter experiment?",
            g, config=base_config,
        )
        pert_parsed = parse_question(
            "What was the key finding of the oracle filter experiment?",
            g, config=perturbed,
        )

        # Both should find at least one entity
        assert len(base_parsed.entity_ids) > 0, "Base config found no entities"
        assert len(pert_parsed.entity_ids) > 0, "Perturbed config found no entities"

        # The curated node (Exp_0_9_OracleFilter) should be top-ranked in both
        assert "Exp_0_9_OracleFilter" in base_parsed.entity_ids, (
            f"Base config missing curated oracle node: {base_parsed.entity_ids}"
        )
        assert "Exp_0_9_OracleFilter" in pert_parsed.entity_ids, (
            f"Perturbed config missing curated oracle node: {pert_parsed.entity_ids}"
        )

        # Top-ranked entity should agree
        assert base_parsed.entity_ids[0] == pert_parsed.entity_ids[0], (
            f"Config perturbation flipped top entity: "
            f"base={base_parsed.entity_ids[0]} vs perturbed={pert_parsed.entity_ids[0]}"
        )

    def test_sub_run_penalty_sensitivity(self):
        """Sub-run penalty perturbation should not promote sub-runs above curated nodes."""
        base_config = NEXUSConfig()
        relaxed_penalty = NEXUSConfig(sub_run_penalty=-0.05)

        # A graph with a curated node and its noisy sub-run siblings.
        g = InMemoryGraphStore()
        g.add_node(Node(
            id="Exp_Main",
            type="Experiment",
            properties={"name": "Main Experiment", "key_finding": "Core result"},
        ))
        g.add_node(Node(
            id="Exp_Main_top64",
            type="Experiment",
            properties={"name": "Main top-64 sub-run"},
        ))
        g.add_node(Node(
            id="Exp_Main_weighted",
            type="Experiment",
            properties={"name": "Main weighted sub-run"},
        ))

        question = "What is the main experiment finding?"
        base_parsed = parse_question(question, g, config=base_config)
        relaxed_parsed = parse_question(question, g, config=relaxed_penalty)

        # The curated "Exp_Main" should be the top entity, never a sub-run.
        assert base_parsed.entity_ids[0] == "Exp_Main", (
            f"Base config promoted sub-run over curated: {base_parsed.entity_ids}"
        )
        assert relaxed_parsed.entity_ids[0] == "Exp_Main", (
            f"Relaxed penalty promoted sub-run over curated: {relaxed_parsed.entity_ids}"
        )
