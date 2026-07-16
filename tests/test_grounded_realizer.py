"""Quality and safety contracts for Grounded Realizer v2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.evaluate_grounded_realizer import evaluate_records
from benchmarks.train_nexus_realizer import (
    load_training_inputs,
    serialization_coverage_for_config,
    serialize_grounded_source,
)
from nexus.realizer.grounded import evidence_candidates, realize_grounded


def _record() -> dict:
    return {
        "id": "q1",
        "question": "What accuracy did NEXUS achieve?",
        "answer": "NEXUS achieved 92% accuracy.",
        "evidence_pack": {
            "node_facts": [{
                "text": "TrainClaim_abcdef: NEXUS achieved 92% accuracy.",
                "source": "report.json#accuracy",
                "confidence": 1.0,
            }],
            "snippets": [],
            "paths": [],
            "facts": [],
        },
    }


def test_evidence_copy_removes_internal_id_and_matches_reference():
    result = realize_grounded(_record())
    assert result.answer == "NEXUS achieved 92% accuracy."
    assert result.strategy == "evidence_copy"
    assert result.grounding_score == 1.0
    assert result.evidence_source == "report.json#accuracy"


def test_garbled_or_unsupported_neural_answer_falls_back():
    garbled = realize_grounded(_record(), "x x x x x \ufffd")
    invented = realize_grounded(_record(), "NEXUS achieved 99% accuracy.")
    appended_claim = realize_grounded(
        _record(), "NEXUS achieved 92% accuracy with Orion Ultra."
    )
    assert garbled.fallback_used is True
    assert invented.fallback_used is True
    assert invented.neural_grounding_score == 0.0
    assert appended_claim.fallback_used is True
    assert {garbled.answer, invented.answer, appended_claim.answer} == {
        "NEXUS achieved 92% accuracy."
    }


def test_supported_neural_answer_is_allowed():
    result = realize_grounded(_record(), "NEXUS achieved 92% accuracy.")
    assert result.strategy == "neural_grounded"
    assert result.fallback_used is False


def test_missing_evidence_fails_closed():
    result = realize_grounded({"question": "Unknown?", "evidence_pack": {}})
    assert result.answer == "Insufficient evidence to answer."
    assert result.strategy == "insufficient_evidence"


def test_compact_serialization_keeps_answer_bearing_evidence_first():
    text = serialize_grounded_source(_record(), 80)
    assert text.startswith("[EVIDENCE] NEXUS achieved 92% accuracy.")
    assert len(text.encode("utf-8")) <= 80


def test_compact_serialization_has_full_real_dataset_coverage():
    _, config, splits = load_training_inputs(
        Path("data/distillation/realizer_v1/manifest.json"),
        Path("training/nexus_realizer_v2.json"),
    )
    coverage = [
        serialization_coverage_for_config(record, config)
        for records in splits.values()
        for record in records
    ]
    assert min(coverage) == 1.0


def test_grounded_validation_protocol_uses_labels_only_for_scoring():
    artifact = evaluate_records([_record()])
    assert artifact["status"] == "GROUNDED_REALIZER_PASS"
    assert artifact["metrics"]["exact_match_rate"] == 1.0
    changed_label = _record()
    changed_label["answer"] = "A deliberately wrong evaluation label."
    assert realize_grounded(changed_label).answer == "NEXUS achieved 92% accuracy."


def test_real_validation_split_is_extractively_solved():
    path = Path("data/distillation/realizer_v1/validation.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    artifact = evaluate_records(records)
    assert artifact["status"] == "GROUNDED_REALIZER_PASS"
    assert artifact["metrics"]["exact_match_rate"] >= 0.99
    assert artifact["metrics"]["hallucination_rate"] == 0.0


def test_stable_model_starts_near_uniform_loss():
    torch = pytest.importorskip("torch")
    from benchmarks.train_nexus_realizer import _batch, _encode, _loss
    from nexus.realizer.model import build_model, parameter_count

    _, config, splits = load_training_inputs(
        Path("data/distillation/realizer_v1/manifest.json"),
        Path("training/nexus_realizer_v2.json"),
    )
    torch.manual_seed(config["seed"])
    model = build_model(config["model"])
    examples = _encode(splits["train"][:4], config)
    source, target = _batch(examples, torch)
    loss = float(_loss(model, source, target, torch).detach())
    uniform_loss = math.log(config["model"]["vocab_size"])
    assert loss < config["training"]["initial_loss_max"]
    assert abs(loss - uniform_loss) < 1.5
    assert parameter_count(model) < 1_100_000
    assert source.shape[0] == 4
    assert source.shape[1] <= config["model"]["max_input_tokens"]
