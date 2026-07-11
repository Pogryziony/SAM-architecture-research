from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stack.encoder.model_bundle import BundleVerificationError, verify_model_bundle


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bundle(root: Path) -> tuple[Path, Path]:
    model_dir = root / "model"
    model_dir.mkdir()
    config = {
        "run_id": "run-1", "source_sha": "a" * 40, "winner": "entity_ranker_v3"
    }
    validation = {
        "run_id": "run-1",
        "source_sha": "a" * 40,
        "selection": {
            "winner": "entity_ranker_v3",
            "winner_recall@10": 0.75,
            "baseline_recall@10": 0.40,
            "val_gate_70pct": True,
            "baseline_gate_15pp": True,
            "proceed_to_frozen": True,
        },
        "dataset_stats": {"validation_groups": 150},
        "evaluations": [
            {"metrics": {"total_gold_entities": 182.0}},
        ],
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    vocab_bytes = b'{"word_vocab":{}}'
    weights_bytes = b"external-model-weights"
    validation_bytes = json.dumps(validation, sort_keys=True).encode()
    (model_dir / "config.json").write_bytes(config_bytes)
    (model_dir / "vocab.json").write_bytes(vocab_bytes)
    (model_dir / "weights.pt").write_bytes(weights_bytes)
    val_path = root / "validation.json"
    val_path.write_bytes(validation_bytes)
    manifest = {
        "manifest_schema_version": "2.0",
        "run_id": "run-1",
        "training_source_sha": "a" * 40,
        "validation_artifact": {
            "path": "validation.json",
            "sha256": _sha(validation_bytes),
            "size_bytes": len(validation_bytes),
        },
        "files": {
            "config.json": {"sha256": _sha(config_bytes), "size_bytes": len(config_bytes)},
            "vocab.json": {"sha256": _sha(vocab_bytes), "size_bytes": len(vocab_bytes)},
        },
        "external_weights": {
            "storage": "external",
            "published": False,
            "local_filename": "weights.pt",
            "sha256": _sha(weights_bytes),
            "size_bytes": len(weights_bytes),
        },
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return model_dir, val_path


def test_valid_external_bundle_is_verified(tmp_path: Path):
    model_dir, val_path = _write_bundle(tmp_path)
    verified = verify_model_bundle(tmp_path, model_dir, val_path)
    assert verified.hashes["weights.pt"] == _sha(b"external-model-weights")


def test_missing_external_weights_fail_closed(tmp_path: Path):
    model_dir, val_path = _write_bundle(tmp_path)
    (model_dir / "weights.pt").unlink()
    with pytest.raises(BundleVerificationError, match="external weights is missing"):
        verify_model_bundle(tmp_path, model_dir, val_path)


def test_tampered_external_weights_are_rejected(tmp_path: Path):
    model_dir, val_path = _write_bundle(tmp_path)
    (model_dir / "weights.pt").write_bytes(b"tampered-model-weights")
    with pytest.raises(BundleVerificationError, match="weights.*mismatch"):
        verify_model_bundle(tmp_path, model_dir, val_path)


def test_tampered_validation_artifact_is_rejected(tmp_path: Path):
    model_dir, val_path = _write_bundle(tmp_path)
    val_path.write_bytes(val_path.read_bytes() + b"\n")
    with pytest.raises(BundleVerificationError, match="validation artifact.*mismatch"):
        verify_model_bundle(tmp_path, model_dir, val_path)


def test_frozen_evaluator_uses_verified_single_read_contract():
    source = (
        Path(__file__).parents[1] / "benchmarks" / "entity_ranker_v3_final.py"
    ).read_text(encoding="utf-8")

    assert "verify_model_bundle(" in source
    assert "load_and_validate_new_holdout(" in source
    assert "evaluate_contract(" in source
    assert "split_path_p.read_bytes" not in source
    assert "from stack.encoder.frozen_split_guard import validate_new_holdout" not in source
