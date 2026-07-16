"""Safety, position-invariance, and runtime tests for Pointer/Copy v3."""

from __future__ import annotations

from nexus.graph import Node
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import ModelInterface
from nexus.realizer.pointer_copy import realize_pointer_copy
from nexus.realizer.pointer_copy import pointer_copy_config_from_dict
from nexus.realizer.grounded import evidence_candidates, grounding_diagnostics
from nexus.utils.config import NEXUSConfig


def _record(*facts: tuple[str, str, float], answer: str = "") -> dict:
    return {
        "question": "What value is assigned to model.max_seq_len in configs/żółć.yaml?",
        "answer": answer,
        "evidence_pack": {
            "node_facts": [
                {"text": text, "source": source, "confidence": confidence}
                for text, source, confidence in facts
            ],
            "snippets": [], "paths": [], "facts": [],
        },
    }


def test_exactly_copies_paths_keys_numbers_quotes_and_unicode():
    text = (
        'In configs/żółć.yaml, model.max_seq_len is set to 128; '
        'ratio is 0.9983 (99.83%) and mode is "retrieval".'
    )
    result = realize_pointer_copy(_record((text, "model.max_seq_len", 1.0)))
    assert result.answer == text
    assert result.strategy == "pointer_copy"
    assert result.selected_candidate_kind == "node_fact"
    assert result.selected_candidate_id
    assert result.grounding_score == 1.0


def test_candidate_order_does_not_change_selection():
    correct = (
        "In configs/żółć.yaml, model.max_seq_len is set to 128.",
        "model.max_seq_len", 1.0,
    )
    distractor = (
        "In configs/other.yaml, train.epochs is set to 3.",
        "train.epochs", 0.7,
    )
    first = realize_pointer_copy(_record(distractor, correct))
    second = realize_pointer_copy(_record(correct, distractor))
    assert first.answer == second.answer == correct[0]
    assert first.selected_candidate_id == second.selected_candidate_id


def test_conflicting_equal_candidates_fail_closed():
    record = _record(
        ("model.max_seq_len is set to 128.", "model.max_seq_len", 1.0),
        ("model.max_seq_len is set to 256.", "model.max_seq_len", 1.0),
    )
    result = realize_pointer_copy(record)
    assert result.strategy == "insufficient_evidence"
    assert result.rejection_reason == "ambiguous_evidence_candidates"
    assert result.answer == "Insufficient evidence to answer."


def test_missing_evidence_fails_closed():
    result = realize_pointer_copy({"question": "Unknown", "evidence_pack": {}})
    assert result.strategy == "insufficient_evidence"
    assert result.candidate_count == 0


def test_grounding_diagnostics_separate_numbers_and_identifiers():
    record = _record((
        "In configs/żółć.yaml, model.max_seq_len is set to 128.",
        "model.max_seq_len", 1.0,
    ))
    candidates = evidence_candidates(record)
    wrong_number = grounding_diagnostics(
        "In configs/żółć.yaml, model.max_seq_len is set to 256.", candidates,
    )
    wrong_path = grounding_diagnostics(
        "In configs/other.yaml, model.max_seq_len is set to 128.", candidates,
    )
    assert wrong_number.score == 0.0
    assert wrong_number.rejection_reason == "unsupported_number"
    assert wrong_number.unsupported_numbers == ("256",)
    assert wrong_number.continuous_support_score > 0.0
    assert wrong_path.score == 0.0
    assert wrong_path.rejection_reason == "unsupported_identifier"
    assert "configs/other.yaml" in wrong_path.unsupported_identifiers


def test_answer_label_is_not_used_for_selection():
    fact = "model.max_seq_len is set to 128."
    left = _record((fact, "model.max_seq_len", 1.0), answer=fact)
    right = _record(
        (fact, "model.max_seq_len", 1.0),
        answer="Deliberately incorrect scoring label.",
    )
    assert realize_pointer_copy(left) == realize_pointer_copy(right)


def test_production_config_identity_includes_pointer_backend():
    lexical = ProductionNEXUSConfig.lexical_only()
    pointer = ProductionNEXUSConfig.pointer_copy()
    assert pointer.realizer_backend == "pointer_copy"
    assert pointer.config_hash != lexical.config_hash
    assert pointer.to_dict()["nexus_config"]["realizer_backend"] == "pointer_copy"


class _NeverGenerate(ModelInterface):
    def generate(self, prompt: str) -> str:
        raise AssertionError("Pointer/Copy factual path must not call the model")

    @property
    def model_name(self) -> str:
        return "never-generate"


def test_pointer_copy_is_connected_to_answer_question_runtime():
    graph = InMemoryGraphStore()
    graph.add_node(Node(
        "SourceA", "Concept",
        properties={"description": "SourceA achieved 92% accuracy."},
        sources=["reports/source-a.md"],
    ))
    result = answer_question(
        "What was the accuracy of SourceA?",
        graph,
        model=_NeverGenerate(),
        config=NEXUSConfig(realizer_backend="pointer_copy"),
    )
    assert result["answer"] == "SourceA achieved 92% accuracy."
    assert result["realization"]["strategy"] == "pointer_copy"
    assert result["realization"]["evidence_source"]


def test_default_runtime_keeps_registered_synth_semantics():
    assert NEXUSConfig().realizer_backend == "synth"


def test_pointer_config_rejects_unknown_or_negative_policy():
    valid = {
        "schema_version": "nexus-pointer-copy-realizer-v3",
        "score_version": "question_evidence_v1",
        "candidate_ordering": "score_then_confidence_then_candidate_id",
        "minimum_score": 1.0,
        "minimum_margin": 0.25,
    }
    assert pointer_copy_config_from_dict(valid).minimum_margin == 0.25

    for changed in (
        {**valid, "schema_version": "unknown"},
        {**valid, "score_version": "unknown"},
        {**valid, "candidate_ordering": "position"},
        {**valid, "minimum_margin": -0.1},
    ):
        try:
            pointer_copy_config_from_dict(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Pointer/Copy policy was accepted")
