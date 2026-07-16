"""Contracts and run-preparation tests for the abstractive Realizer pilot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.abstractive_realizer_contracts import (
    load_abstractive_splits, materialize_slot_template,
    validate_abstractive_manifest, validate_abstractive_record,
)
from benchmarks.acquire_realizer_train_data import load_verified_acquisition
from benchmarks.build_abstractive_realizer_dataset import build_abstractive_dataset
from benchmarks.check_abstractive_full_training_readiness import build_readiness
from benchmarks.prepare_abstractive_realizer_run import evaluate_preparation
from benchmarks.realizer_contracts import sha256_file, validate_dataset_manifest
from benchmarks.train_nexus_realizer import (
    load_training_inputs, serialize_source_for_config,
    serialization_coverage_for_config, training_target_for_config,
    validate_readiness_for_training,
)
from benchmarks.train_nexus_realizer_v2 import _select_generation_subset
from nexus.realizer.decoder import score_candidate_texts
from nexus.realizer.tokenizer import ByteTokenizer


@pytest.fixture(scope="module")
def prepared_dataset(tmp_path_factory: pytest.TempPathFactory):
    root = Path.cwd()
    acquisition_manifest = root / "data/realizer_train/source_claims_v1/manifest.json"
    consumed_manifest_path = root / "data/distillation/realizer_v1/manifest.json"
    acquisition, _ = load_verified_acquisition(
        acquisition_manifest, root, verify_current_sources=False,
    )
    consumed_manifest = json.loads(consumed_manifest_path.read_text(encoding="utf-8"))
    assert validate_dataset_manifest(consumed_manifest, consumed_manifest_path.parent) == []
    consumed_by_split = {
        split: [
            json.loads(line)
            for line in (
                consumed_manifest_path.parent / consumed_manifest["splits"][split]["path"]
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        for split in ("train", "validation")
    }
    output = tmp_path_factory.mktemp("abstractive_dataset")
    manifest = build_abstractive_dataset(
        acquisition,
        consumed_by_split["train"] + consumed_by_split["validation"],
        consumed_by_split["validation"],
        output,
        source_sha="a" * 40,
        acquisition_manifest_sha256=sha256_file(acquisition_manifest),
        consumed_manifest_sha256=sha256_file(consumed_manifest_path),
    )
    return output, manifest


def test_dataset_is_unique_multi_evidence_and_leakage_safe(prepared_dataset):
    root, manifest = prepared_dataset
    assert manifest["pairs_accepted"] >= 1000
    assert manifest["atomic_claims_used"] == 2 * manifest["pairs_accepted"]
    assert manifest["atomic_claim_reuse_count"] == 0
    assert manifest["single_candidate_target_count"] == 0
    assert manifest["quarantined_source_families"] == 44
    assert manifest["old_question_overlap"] == manifest["old_answer_overlap"] == 0
    assert 0.15 <= manifest["validation_fraction_actual"] <= 0.25
    assert set(manifest["counts_by_relation"]) == {"different", "the same"}
    assert set(manifest["counts_by_task"]) == {
        "config_value_comparison", "table_value_comparison",
    }
    assert validate_abstractive_manifest(manifest, root) == []

    _, splits = load_abstractive_splits(root / "manifest.json")
    train_families = {family for row in splits["train"] for family in row["source_families"]}
    val_families = {family for row in splits["validation"] for family in row["source_families"]}
    assert train_families.isdisjoint(val_families)


def test_slots_preserve_bindings_and_target_is_not_one_fact(prepared_dataset):
    root, _ = prepared_dataset
    _, splits = load_abstractive_splits(root / "manifest.json")
    record = splits["train"][0]
    facts = {item["text"] for item in record["evidence_pack"]["node_facts"]}
    assert record["answer"] not in facts
    assert materialize_slot_template(record["training_target"], record["slots"]) == record["answer"]
    assert validate_abstractive_record(record) == []

    tampered = json.loads(json.dumps(record))
    tampered["composition"]["relation"] = (
        "the same" if record["composition"]["relation"] == "different" else "different"
    )
    assert "incorrect_relation" in validate_abstractive_record(tampered)


def test_trainer_loads_slot_contract_without_target_leakage(prepared_dataset):
    root, _ = prepared_dataset
    config_path = Path("training/nexus_realizer_abstractive_v1.json")
    _, config, splits = load_training_inputs(root / "manifest.json", config_path)
    record = splits["train"][0]
    serialized = serialize_source_for_config(record, config)
    assert record["training_target"] not in serialized
    assert record["answer"] not in serialized
    assert str(record["slots"]["SOURCE_1"]) not in serialized
    assert str(record["slots"]["SOURCE_2"]) not in serialized
    assert serialization_coverage_for_config(record, config) == 1.0
    expected = "SAME" if record["composition"]["relation"] == "the same" else "DIFFERENT"
    assert f"[VERIFIED_RELATION] {expected}" in serialized
    assert training_target_for_config(record, config) == expected

    tampered = json.loads(json.dumps(record))
    tampered["composition"]["relation"] = (
        "different" if record["composition"]["relation"] == "the same" else "the same"
    )
    with pytest.raises(ValueError, match="contradicts immutable evidence"):
        serialize_source_for_config(tampered, config)


def test_generation_subset_is_relation_balanced():
    records = [
        {"id": f"same-{index}", "composition": {"relation": "the same"}}
        for index in range(8)
    ] + [
        {"id": f"different-{index}", "composition": {"relation": "different"}}
        for index in range(4)
    ]
    selected = _select_generation_subset(records, 6, 123)
    relations = [record["composition"]["relation"] for record in selected]
    assert relations.count("the same") == 3
    assert relations.count("different") == 3
    assert _select_generation_subset(records, 6, 123) == selected


def test_constrained_candidate_scoring_returns_complete_allowed_label():
    torch = pytest.importorskip("torch")
    tokenizer = ByteTokenizer()
    expected = tokenizer.encode("SAME", 32)

    class PreferSame(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, source, target):
            logits = torch.full(
                (target.shape[0], target.shape[1], tokenizer.vocab_size), -8.0,
                device=target.device,
            )
            for position in range(target.shape[1]):
                token = expected[min(position + 1, len(expected) - 1)]
                logits[:, position, token] = 8.0 + self.anchor
            return logits

    selected, diagnostics = score_candidate_texts(
        PreferSame(), [tokenizer.BOS, tokenizer.EOS],
        ["SAME", "DIFFERENT"], tokenizer, max_length=32,
    )
    assert selected == "SAME"
    assert diagnostics["strategy"] == "constrained_candidates"
    assert diagnostics["score_margin"] > 0


def test_no_write_preflight_passes(prepared_dataset):
    pytest.importorskip("torch")
    root, _ = prepared_dataset
    result = evaluate_preparation(
        root / "manifest.json",
        Path("training/nexus_realizer_abstractive_v1.json"),
        run_overfit_smoke=False,
    )
    assert result["status"] == "READY_FOR_BOUNDED_PILOT"
    assert result["preflight"]["status"] == "PREFLIGHT_PASS"
    assert result["preflight"]["weights_written"] is False


def test_full_training_readiness_binds_pilot_inputs(prepared_dataset, tmp_path):
    root, manifest = prepared_dataset
    config_path = Path("training/nexus_realizer_abstractive_v1.json")
    preparation = evaluate_preparation(
        root / "manifest.json", config_path, run_overfit_smoke=False,
    )
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
    weights_path = tmp_path / "model.pt"
    weights_path.write_bytes(b"test-weights")
    identity = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], text=True,
        ).strip(),
    }
    evaluation = {
        "status": "PILOT_CHECKPOINT_ACCEPTED",
        "blocking_checks": [],
        "checks": [{"name": "quality", "passed": True}],
        "dataset_sha256": manifest["dataset_sha256"],
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "source": identity,
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    readiness = build_readiness(
        preparation_path, evaluation_path, root / "manifest.json",
        config_path, weights_path,
    )
    assert readiness["status"] == "READY_FOR_FULL_TRAINING"
    assert readiness["full_training_launched"] is False
    assert validate_readiness_for_training(
        readiness, root / "manifest.json", config_path,
    ) == []
