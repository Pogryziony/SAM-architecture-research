"""Contracts and run-preparation tests for the abstractive Realizer pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.abstractive_realizer_contracts import (
    load_abstractive_splits, materialize_slot_template,
    validate_abstractive_manifest, validate_abstractive_record,
)
from benchmarks.acquire_realizer_train_data import load_verified_acquisition
from benchmarks.build_abstractive_realizer_dataset import build_abstractive_dataset
from benchmarks.prepare_abstractive_realizer_run import evaluate_preparation
from benchmarks.realizer_contracts import sha256_file, validate_dataset_manifest
from benchmarks.train_nexus_realizer import (
    load_training_inputs, serialize_source_for_config,
    serialization_coverage_for_config, training_target_for_config,
)


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
    assert serialization_coverage_for_config(record, config) == 1.0
    assert training_target_for_config(record, config) == record["training_target"]


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
