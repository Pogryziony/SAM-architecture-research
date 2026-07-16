"""Integrity and inference smoke tests for the accepted comparison-plan pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.realizer_contracts import sha256_file
from benchmarks.train_nexus_realizer import load_training_inputs


ROOT = Path(__file__).parent.parent
MODEL_ROOT = ROOT / "models/realizer/abstractive_v1_plan_v3"
CONFIG = ROOT / "training/nexus_realizer_abstractive_v1.json"
DATASET = ROOT / "data/distillation/realizer_abstractive_v1/manifest.json"


def test_checkpoint_manifest_and_hash_are_exact():
    manifest = json.loads((MODEL_ROOT / "manifest.json").read_text(encoding="utf-8"))
    weights = MODEL_ROOT / manifest["weights"]["path"]
    assert manifest["status"] == "PILOT_CHECKPOINT_ACCEPTED"
    assert weights.stat().st_size == manifest["weights"]["size_bytes"]
    assert sha256_file(weights) == manifest["weights"]["sha256"]
    assert sha256_file(CONFIG) == manifest["config_sha256"]
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    assert dataset["dataset_sha256"] == manifest["dataset_sha256"]


def test_checkpoint_follows_both_verified_relation_plans():
    torch = pytest.importorskip("torch")
    from benchmarks.train_nexus_realizer_v2 import _generate_and_score
    from nexus.realizer.decoder import DecoderConfig
    from nexus.realizer.model import build_model, parameter_count

    manifest, config, splits = load_training_inputs(DATASET, CONFIG)
    model = build_model(config["model"])
    model.load_state_dict(torch.load(
        MODEL_ROOT / "model.pt", map_location="cpu", weights_only=True,
    ))
    assert parameter_count(model) == 959747
    records = [
        next(row for row in splits["validation"] if row["composition"]["relation"] == relation)
        for relation in ("the same", "different")
    ]
    metrics = _generate_and_score(
        model, records, config,
        DecoderConfig(
            strategy="constrained_relation_v2", repetition_penalty=1.0,
            no_repeat_ngram_size=0, max_length=16,
        ),
        max_samples=2,
    )
    assert metrics["answer_exact_match_rate"] == 1.0
    assert metrics["relation_accuracy_by_class"] == {
        "different": 1.0, "the same": 1.0,
    }
    assert metrics["slot_placeholder_exact_rate"] == 1.0
