"""Runtime and end-to-end tests for the accepted comparison-plan Realizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus.graph import Node
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig, validate_config
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import ModelInterface
from nexus.realizer.comparison_plan import (
    BACKEND_NAME,
    DEFAULT_WEIGHTS_SHA256,
    HYBRID_BACKEND_NAME,
    build_verified_comparison_plan,
    extract_comparison_slots,
    realize_comparison_plan,
)
from nexus.utils.config import NEXUSConfig


ROOT = Path(__file__).resolve().parents[1]


def _follow_verified_plan(source: str, candidates: tuple[str, str]):
    assert candidates == ("SAME", "DIFFERENT")
    selected = "SAME" if "[VERIFIED_RELATION] SAME\n" in source else "DIFFERENT"
    return selected, {"strategy": "test_plan_follower"}


def _pack(value_1: str = "0.5", value_2: str = "0.5") -> dict:
    return {
        "node_facts": [
            {
                "text": f"In configs/a.yaml, rate is set to {value_1}.",
                "source": "configs/a.yaml",
            },
            {
                "text": f"In configs/b.yaml, rate is set to {value_2}.",
                "source": "configs/b.yaml",
            },
        ]
    }


def _question() -> str:
    return (
        "Compare rate in configs/a.yaml and configs/b.yaml. What value does "
        "each source report, and are the values the same or different?"
    )


def test_symbolic_plan_is_derived_from_immutable_values():
    same = build_verified_comparison_plan(_question(), _pack())
    different = build_verified_comparison_plan(_question(), _pack("0.5", "0.7"))
    assert (same.relation, same.label) == ("the same", "SAME")
    assert (different.relation, different.label) == ("different", "DIFFERENT")


def test_registered_table_fact_form_is_supported():
    pack = {
        "node_facts": [
            {
                "text": "For Inference p50, Threshold is <= 50 ms.",
                "source": "EXPERIMENT_ENTITY_RANKER_V3.md",
            },
            {
                "text": "For resolution_rate, Threshold is >= 100%.",
                "source": "STAGE1B_NEGATIVE.md",
            },
        ]
    }
    question = (
        "Compare Threshold for Inference p50 in EXPERIMENT_ENTITY_RANKER_V3.md "
        "with resolution_rate in STAGE1B_NEGATIVE.md."
    )
    slots = extract_comparison_slots(question, pack)
    assert slots.subject_1 == "Inference p50"
    assert slots.value_1 == "<= 50 ms"
    assert slots.subject_2 == "resolution_rate"
    assert slots.value_2 == ">= 100%"


def test_unknown_or_ambiguous_evidence_fails_closed():
    ambiguous = _pack()
    ambiguous["node_facts"].append({
        "text": "In configs/c.yaml, rate is set to 0.5.",
        "source": "configs/c.yaml",
    })
    result = realize_comparison_plan(
        "Compare rates without naming sources.",
        ambiguous,
        label_selector=_follow_verified_plan,
    )
    assert result.strategy == "insufficient_evidence"
    assert result.answer == "Insufficient evidence to answer."
    assert result.neural_used is False


def test_model_cannot_override_symbolic_relation():
    def contradict(_source: str, _candidates: tuple[str, str]):
        return "DIFFERENT", {"strategy": "malicious_test"}

    result = realize_comparison_plan(
        _question(), _pack(), label_selector=contradict,
    )
    assert result.strategy == "insufficient_evidence"
    assert result.rejection_reason == "comparison_model_contradicted_verified_plan"
    assert result.answer == "Insufficient evidence to answer."


def test_missing_checkpoint_is_reported_without_falling_back_to_generation(tmp_path: Path):
    result = realize_comparison_plan(
        _question(),
        _pack(),
        model_dir=str(tmp_path / "missing"),
        config_path=str(tmp_path / "missing.json"),
        expected_weights_sha256="0" * 64,
    )
    assert result.strategy == "insufficient_evidence"
    assert result.rejection_reason == "comparison_checkpoint_artifact_missing"


class _NeverGenerate(ModelInterface):
    def generate(self, prompt: str) -> str:
        raise AssertionError("comparison backend must not call the synth model")

    @property
    def model_name(self) -> str:
        return "never-generate"


def test_comparison_backend_is_connected_to_answer_question_end_to_end():
    graph = InMemoryGraphStore()
    graph.add_node(Node(
        "ConfigA", "Document", aliases=["configs/a.yaml"],
        properties={"description": "In configs/a.yaml, rate is set to 0.5."},
        sources=["configs/a.yaml"],
    ))
    graph.add_node(Node(
        "ConfigB", "Document", aliases=["configs/b.yaml"],
        properties={"description": "In configs/b.yaml, rate is set to 0.7."},
        sources=["configs/b.yaml"],
    ))
    result = answer_question(
        _question(),
        graph,
        model=_NeverGenerate(),
        config=NEXUSConfig(realizer_backend=BACKEND_NAME),
        entry_nodes_override=["ConfigA", "ConfigB"],
        comparison_label_selector=_follow_verified_plan,
    )
    assert result["answer"] == (
        "configs/a.yaml reports 0.5 for rate, while configs/b.yaml reports "
        "0.7 for rate; the values are different."
    )
    assert result["realization"]["strategy"] == BACKEND_NAME
    assert result["realization"]["relation_plan"] == "DIFFERENT"
    assert result["realization"]["predicted_relation"] == "DIFFERENT"
    assert result["realization"]["slots"]["VALUE_1"] == "0.5"
    assert result["verification"].passed is True


def test_production_factory_binds_checkpoint_identity():
    config = ProductionNEXUSConfig.comparison_plan()
    assert config.realizer_backend == BACKEND_NAME
    assert config.realizer_checkpoint_sha256 == DEFAULT_WEIGHTS_SHA256
    assert config.to_dict()["nexus_config"]["realizer_checkpoint_sha256"] == (
        DEFAULT_WEIGHTS_SHA256
    )
    assert config.config_hash != ProductionNEXUSConfig.lexical_only().config_hash
    assert validate_config(config) == []
    grounded = ProductionNEXUSConfig.grounded()
    assert grounded.realizer_backend == HYBRID_BACKEND_NAME
    assert grounded.realizer_checkpoint_sha256 == DEFAULT_WEIGHTS_SHA256
    assert validate_config(grounded) == []


def test_grounded_profile_routes_factual_qa_to_pointer_copy():
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
        config=ProductionNEXUSConfig.grounded(),
    )
    assert result["answer"] == "SourceA achieved 92% accuracy."
    assert result["realization"]["strategy"] == "pointer_copy"


def test_registered_checkpoint_runs_on_real_validation_record():
    pytest.importorskip("torch")
    record = json.loads(
        (ROOT / "data/distillation/realizer_abstractive_v1/validation.jsonl")
        .read_text(encoding="utf-8").splitlines()[0]
    )
    result = realize_comparison_plan(record["question"], record["evidence_pack"])
    assert result.strategy == BACKEND_NAME
    assert result.answer == record["answer"]
    assert result.checkpoint_sha256 == DEFAULT_WEIGHTS_SHA256
    assert result.neural_used is True


def test_registered_checkpoint_runs_end_to_end_on_balanced_validation_sample():
    pytest.importorskip("torch")
    records = [
        json.loads(line)
        for line in (
            ROOT / "data/distillation/realizer_abstractive_v1/validation.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = []
    for relation in ("the same", "different"):
        selected.extend(
            record for record in records
            if record["composition"]["relation"] == relation
        )
        selected = selected[:5] if relation == "the same" else selected[:10]
    assert len(selected) == 10

    config = ProductionNEXUSConfig.grounded()
    for record in selected:
        graph = InMemoryGraphStore()
        entry_nodes = []
        for index, fact in enumerate(record["evidence_pack"]["node_facts"], 1):
            node_id = f"RuntimeEvidence{index}"
            entry_nodes.append(node_id)
            graph.add_node(Node(
                node_id,
                "Document",
                properties={"description": fact["text"]},
                sources=[fact["source"]],
            ))
        result = answer_question(
            record["question"],
            graph,
            config=config,
            entry_nodes_override=entry_nodes,
        )
        assert result["answer"] == record["answer"], record["id"]
        assert result["realization"]["strategy"] == BACKEND_NAME
        assert result["realization"]["checkpoint_sha256"] == DEFAULT_WEIGHTS_SHA256
