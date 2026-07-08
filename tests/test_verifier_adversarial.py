"""
Phase 2 — Adversarial verifier tests.

These tests exercise the verifier with crafted evidence packs and answers
designed to expose hallucination detection gaps. After the verifier fix
(commit 231b5cc), both qwen models reported 0.0% hallucination — suspected
overcorrection. These tests validate the FIXED verifier catches fabrications
while still passing honest verbose answers.

Gate requirement: ALL 12+ tests must pass, and at least ONE adversarial
case must produce hallucination_rate > 0 for every backend tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.reasoning.verifier import (
    Verifier,
    extract_factual_claims,
    _collect_evidence_entities,
    _collect_evidence_relations,
    _entity_present,
)
from benchmarks.scoring import _extract_numbers, _fuzzy_number_match


# ── Helper: build minimal evidence pack ─────────────────────────────────

def _make_evidence(
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
    facts: list[str] | None = None,
) -> dict:
    """Build a minimal evidence pack dict for testing the verifier."""
    return {
        "question": "test question",
        "paths": [
            {
                "score": 1.0,
                "length": 1,
                "nodes": nodes or [],
                "edges": edges or [],
            }
        ],
        "facts": facts or [],
        "sources": [],
    }


def _node(id_: str, type_: str = "Experiment", **kwargs) -> dict:
    """Build a minimal node dict."""
    d: dict = {"id": id_, "type": type_}
    d.update(kwargs)
    return d


def _edge(from_: str, to_: str, type_: str = "validates", confidence: float = 1.0) -> dict:
    """Build a minimal edge dict."""
    return {
        "type": type_,
        "from": from_,
        "to": to_,
        "confidence": confidence,
        "reversed": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Adversarial test cases
# ═══════════════════════════════════════════════════════════════════════════

class TestAdversarialCases:
    """Each test verifies a specific hallucination pattern is detected or
    correctly passed."""

    # ── Fabrication cases (must detect) ────────────────────────────────

    def test_fabricated_number(self):
        """Evidence says '99.87% accuracy', answer says '87.9% accuracy'
        — must detect unsupported numeric claim."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_6_Validation", name="oracle memory experiment"),
            ],
            facts=[
                "Exp_0_6_Validation: The oracle memory experiment achieved 99.87% accuracy with core-only achieving 96.6%.",
            ],
        )
        answer = "The oracle memory experiment achieved 87.9% accuracy. This result is significant."

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        assert result.hallucination_rate > 0.0, (
            f"Fabricated number (87.9% vs 99.87%) must produce hallucination_rate > 0, "
            f"got {result.hallucination_rate}"
        )
        assert len(result.unsupported_claims) >= 1, (
            f"Expected at least 1 unsupported claim for wrong number, "
            f"got {len(result.unsupported_claims)}"
        )
        assert not result.passed, "Answer with fabricated number should FAIL verification"

    def test_fabricated_entity(self):
        """Evidence has nodes A, B, C. Answer claims entity 'Z_Phantom' exists
        — must detect unsupported entity claim."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_1", name="experiment one"),
                _node("Concept_Alpha", type_="Concept", name="concept alpha"),
                _node("Bug_42", type_="Bug", name="bug 42"),
            ],
            facts=[
                "Exp_0_1 validates Concept_Alpha (confidence: 1.00)",
                "Bug_42 caused by Concept_Alpha (confidence: 0.95)",
            ],
            edges=[
                _edge("Exp_0_1", "Concept_Alpha", "validates"),
                _edge("Concept_Alpha", "Bug_42", "caused_by"),
            ],
        )
        answer = "Entity Z_Phantom caused the error in experiment one."

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        assert result.hallucination_rate > 0.0, (
            f"Fabricated entity 'Z_Phantom' must produce hallucination_rate > 0, "
            f"got {result.hallucination_rate}"
        )
        # At least one claim should be unsupported
        assert len(result.unsupported_claims) >= 1

    def test_inverted_relation(self):
        """Evidence 'A validates B', answer 'B validates A'
        — must detect inverted relation."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_6", name="oracle experiment"),
                _node("Concept_Bottleneck", type_="Concept", name="selector bottleneck"),
            ],
            edges=[
                _edge("Exp_0_6", "Concept_Bottleneck", "validates"),
            ],
            facts=[
                "Exp_0_6 validates Concept_Bottleneck (confidence: 1.00)",
            ],
        )
        answer = "Concept_Bottleneck validates Exp_0_6."

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        # Inverted relation should be detected as unsupported
        assert result.hallucination_rate > 0.0, (
            f"Inverted relation 'B validates A' (evidence has 'A validates B') "
            f"must produce hallucination_rate > 0, got {result.hallucination_rate}"
        )
        assert len(result.unsupported_claims) >= 1, (
            f"Expected at least 1 unsupported claim for inverted relation, "
            f"got {len(result.unsupported_claims)}"
        )

    def test_mixed_correct_one_fabricated(self):
        """Three correct claims + one fabricated → hallucination_rate > 0, not 0.0."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_6_Validation", name="oracle memory experiment"),
                _node("Concept_SelectorBottleneck", type_="Concept", name="selector bottleneck"),
            ],
            facts=[
                "Exp_0_6_Validation: 99.87% accuracy with 1,650 slots.",
                "Exp_0_6_Validation validates Concept_SelectorBottleneck (confidence: 1.00)",
            ],
            edges=[
                _edge("Exp_0_6_Validation", "Concept_SelectorBottleneck", "validates"),
            ],
        )
        # 4 factual claims: 3 grounded, 1 fabricated
        answer = (
            "The oracle memory experiment achieved 99.87% accuracy. "
            "It also achieved 1,650 slots of capacity. "
            "This experiment validates the selector bottleneck concept. "
            "Additionally, the experiment achieved 92.5% accuracy on unrelated tasks."
        )

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        # 92.5% is not in evidence → this claim should be unsupported
        assert result.hallucination_rate > 0.0, (
            f"Mixed correct+fabricated must have hallucination_rate > 0, "
            f"got {result.hallucination_rate}"
        )
        assert result.hallucination_rate < 1.0, (
            f"Most claims are correct; hallucination_rate should be < 1.0, "
            f"got {result.hallucination_rate}"
        )

    def test_partially_fabricated_multihop(self):
        """Evidence A→B→C, answer says A→B→D — must detect D fabrication."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_1", name="experiment one"),
                _node("Concept_Alpha", type_="Concept", name="concept alpha"),
                _node("Concept_Beta", type_="Concept", name="concept beta"),
            ],
            edges=[
                _edge("Exp_0_1", "Concept_Alpha", "validates"),
                _edge("Concept_Alpha", "Concept_Beta", "depends_on"),
            ],
            facts=[
                "Exp_0_1 validates Concept_Alpha (confidence: 1.00).",
                "Concept_Alpha depends_on Concept_Beta (confidence: 0.90).",
            ],
        )
        answer = "Exp_0_1 validates Concept_Alpha, and Concept_Alpha depends on Concept_Gamma."

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        # "Concept_Gamma" is fabricated → at least one unsupported claim
        assert result.hallucination_rate > 0.0, (
            f"Partially fabricated multi-hop (A→B→D vs A→B→C) "
            f"must have hallucination_rate > 0, got {result.hallucination_rate}"
        )

    def test_all_fabricated(self):
        """Answer with no grounding in evidence at all → high hallucination rate."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_6", name="oracle memory"),
            ],
            facts=["Exp_0_6: oracle memory achieved 99.87% accuracy."],
        )
        answer = "The quantum computing approach achieved 100.0% precision with a novel neural architecture."

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        assert result.hallucination_rate > 0.0, (
            f"All-fabricated answer must have hallucination_rate > 0, "
            f"got {result.hallucination_rate}"
        )
        # All claims should be unsupported
        assert result.supported_count == 0, (
            f"All-fabricated answer should have 0 supported claims, "
            f"got {result.supported_count}"
        )

    def test_false_precision(self):
        """Answer adds detail not in evidence ('the experiment ran for 3 epochs
        on a CPU') when evidence only states the fact → must detect unsupported."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_6_Validation", name="oracle memory experiment"),
            ],
            facts=[
                "Exp_0_6_Validation: The oracle memory experiment demonstrated external memory usage.",
            ],
        )
        answer = "The oracle memory experiment ran for 3 epochs on a single CPU and achieved 99.87% accuracy."

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        # "99.87%" is fabricated (not in evidence), "3 epochs" is fabricated
        assert result.hallucination_rate > 0.0, (
            f"False precision claims (3 epochs, 99.87%) must produce "
            f"hallucination_rate > 0, got {result.hallucination_rate}"
        )

    # ── Honest-pass cases (must NOT detect) ───────────────────────────

    def test_verbose_but_grounded(self):
        """Lots of filler words but all factual claims are in evidence
        — must PASS (0% hallucination)."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_6_Validation", name="oracle memory experiment"),
                _node("Concept_SelectorBottleneck", type_="Concept", name="selector bottleneck"),
            ],
            facts=[
                "Exp_0_6_Validation: The oracle memory experiment achieved 99.87% accuracy.",
                "Exp_0_6_Validation validates Concept_SelectorBottleneck (confidence: 1.00)",
            ],
            edges=[
                _edge("Exp_0_6_Validation", "Concept_SelectorBottleneck", "validates"),
            ],
        )
        answer = (
            "Let me explain the results in detail. Based on the evidence, "
            "I can see that the oracle memory experiment is quite interesting. "
            "It is worth noting that this experiment achieved 99.87% accuracy. "
            "Furthermore, it is important to note that this experiment validates "
            "the selector bottleneck concept. In conclusion, the findings are "
            "quite compelling and significant for the field."
        )

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        assert result.hallucination_rate == 0.0, (
            f"Verbose-but-grounded answer must have 0% hallucination, "
            f"got {result.hallucination_rate}"
        )
        assert result.passed, "Verbose-but-grounded answer must PASS verification"
        # All factual claims should be supported (fillers filtered)
        assert result.supported_count > 0

    def test_template_style_answer(self):
        """Template-style answer must be judged on content, not penalized
        for template phrasing."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_6_Validation", name="oracle memory experiment"),
            ],
            facts=[
                "Exp_0_6_Validation: The oracle memory experiment achieved 99.87% accuracy, "
                "demonstrating that SAM core CAN use external memory.",
            ],
        )
        answer = (
            "Based on the evidence, the oracle memory experiment achieved "
            "99.87% accuracy, demonstrating that SAM core CAN use external memory."
        )

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        assert result.hallucination_rate == 0.0, (
            f"Template-style answer must have 0% hallucination, "
            f"got {result.hallucination_rate}"
        )
        assert result.passed, "Template-style answer must PASS verification"

    def test_empty_answer(self):
        """Empty answer must return hallucination_rate=0, passed=True."""
        evidence = _make_evidence(
            nodes=[_node("Exp_0_6")],
            facts=["Exp_0_6: 99.87% accuracy."],
        )

        verifier = Verifier(hallucination_threshold=0.2)

        for answer in ["", "   \n  ", "Insufficient evidence to answer."]:
            result = verifier.verify(answer, evidence)
            assert result.hallucination_rate == 0.0, (
                f"Empty/insufficient answer '{answer[:30]}' must have 0% hallucination, "
                f"got {result.hallucination_rate}"
            )
            assert result.passed is True, (
                f"Empty/insufficient answer '{answer[:30]}' must pass verification"
            )

    def test_correct_number_different_formatting(self):
        """Evidence '99.87%', answer 'approximately 99.9%'
        — must PASS (within 5% tolerance, same as scoring)."""
        evidence = _make_evidence(
            nodes=[
                _node("Exp_0_6_Validation", name="oracle memory experiment"),
            ],
            facts=["Exp_0_6_Validation: 99.87% accuracy."],
        )
        answer = "The oracle memory experiment achieved approximately 99.9% accuracy."

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        # 99.9% is within 5% of 99.87% → should pass
        assert result.hallucination_rate == 0.0, (
            f"99.9% is within 5% of 99.87% → must have 0% hallucination, "
            f"got {result.hallucination_rate}. Unsupported: {result.unsupported_claims}"
        )
        assert result.passed, "Approximately correct number must PASS verification"

    def test_entity_alias_match(self):
        """Evidence has node 'Exp_0_6_Validation' with alias 'oracle memory
        experiment', answer says 'oracle memory experiment' — must PASS."""
        evidence = _make_evidence(
            nodes=[
                _node(
                    "Exp_0_6_Validation",
                    name="Exp_0_6_Validation",
                    aliases=["oracle memory experiment", "oracle exp"],
                ),
            ],
            facts=[
                "Exp_0_6_Validation: achieved 99.87% accuracy with 1,650 slots.",
            ],
        )
        answer = "The oracle memory experiment achieved 99.87% accuracy."

        verifier = Verifier(hallucination_threshold=0.2)
        result = verifier.verify(answer, evidence)

        assert result.hallucination_rate == 0.0, (
            f"Alias 'oracle memory experiment' must match evidence, "
            f"got hallucination_rate={result.hallucination_rate}"
        )
        assert result.passed, "Alias-matched answer must PASS verification"


# ═══════════════════════════════════════════════════════════════════════════
# Component-level tests (unit tests for specific verifier functions)
# ═══════════════════════════════════════════════════════════════════════════

class TestVerifierComponents:
    """Tests for individual verifier helper functions."""

    def test_collect_evidence_entities_includes_aliases(self):
        """Aliases on nodes must be collected as evidence entities."""
        evidence = {
            "paths": [
                {
                    "nodes": [
                        {"id": "Exp_0_6", "type": "Experiment", "aliases": ["oracle test", "exp6"]},
                    ],
                    "edges": [],
                }
            ],
            "facts": [],
        }
        entities = _collect_evidence_entities(evidence)
        assert "oracle test" in entities, "Alias 'oracle test' must be in evidence entities"
        assert "exp6" in entities, "Alias 'exp6' must be in evidence entities"

    def test_collect_evidence_relations_normalized(self):
        """Evidence relations must be normalized (both canonical and surface forms)."""
        evidence = {
            "paths": [
                {
                    "nodes": [],
                    "edges": [
                        {"type": "validates", "from": "A", "to": "B", "confidence": 1.0},
                    ],
                }
            ],
            "facts": [],
        }
        relations = _collect_evidence_relations(evidence)
        # Should contain canonical form
        assert "validated_by" in relations or "validates" in relations, (
            "Must include canonical relation forms"
        )
        # Surface form "validates" should be included
        has_validates_surface = any("validates" == r or "validates" in r for r in relations)
        assert has_validates_surface, "Surface form 'validates' must be in relations"

    def test_numeric_tolerance_match(self):
        """Verify that _fuzzy_number_match with 5% tolerance works correctly."""
        # 99.9 vs 99.87: 99.9/100=0.999, 99.87/100=0.9987
        # |0.999 - 0.9987| / 0.9987 = 0.0003/0.9987 ≈ 0.0003 < 0.05 → match
        gt = {0.9987}
        pred = {0.999}
        matches, total = _fuzzy_number_match(pred, gt)
        assert matches == 1, f"99.9% should fuzzy-match 99.87% (within 5%): got {matches}/{total}"

        # 87.9 vs 99.87: |0.879 - 0.9987| / 0.9987 = 0.1197/0.9987 ≈ 0.12 > 0.05 → no match
        pred2 = {0.879}
        matches2, total2 = _fuzzy_number_match(pred2, gt)
        assert matches2 == 0, f"87.9% should NOT match 99.87% (12% off): got {matches2}/{total2}"

    def test_factual_claim_filter_filters_filler(self):
        """Filler sentences (transitions, qualifiers) must be filtered out."""
        filler = "Let me explain the results in detail."
        qualifier = "This is important to note for the field."
        transition = "Based on the evidence, the findings are clear."

        claims = extract_factual_claims(filler)
        assert len(claims) == 0, f"Filler sentence must produce 0 factual claims, got {claims}"

        claims2 = extract_factual_claims(qualifier)
        assert len(claims2) == 0, f"Qualifier sentence must produce 0 factual claims"

        claims3 = extract_factual_claims(transition)
        assert len(claims3) == 0, f"Transition sentence must produce 0 factual claims"

    def test_factual_claim_filter_keeps_grounded(self):
        """Sentences with actual entities and facts must be preserved."""
        grounded = "The oracle memory experiment achieved 99.87% accuracy."
        claims = extract_factual_claims(grounded)
        assert len(claims) >= 1, f"Grounded claim must be preserved, got {claims}"

    def test_extract_numbers_from_evidence(self):
        """Numbers extracted from evidence must match scoring.py _extract_numbers."""
        text = "99.87% accuracy with 1,650 slots"
        nums = _extract_numbers(text)
        assert 0.9987 in nums, f"99.87% → 0.9987, got {nums}"
        assert 1650.0 in nums, f"1,650 → 1650.0, got {nums}"
