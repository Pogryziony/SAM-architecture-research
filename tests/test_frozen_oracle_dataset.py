"""Frozen oracle dataset + paired oracle/predicted reporting guards."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.build_frozen_oracle_dataset import build_frozen_dataset, write_frozen_dataset
from benchmarks.run_oracle_vs_predicted import (
    pair_rows,
    summarize_rows,
    validate_paired_artifact,
)
from benchmarks.run_nexus_oracle import validate_oracle_records


_ROOT = Path(__file__).resolve().parents[1]
_QUESTIONS = _ROOT / "stack" / "encoder" / "data" / "val.jsonl"
_RELATIONS = _ROOT / "benchmarks" / "qa-dataset" / "relation_gold.jsonl"
_ORACLE_V1 = _ROOT / "benchmarks" / "qa-dataset" / "oracle_v1.jsonl"
_ORACLE_MANIFEST = _ROOT / "benchmarks" / "qa-dataset" / "oracle_v1.manifest.json"


def test_build_frozen_oracle_dataset_is_valid(tmp_path: Path):
    records, manifest = build_frozen_dataset(_QUESTIONS, _RELATIONS)
    assert validate_oracle_records(records) == []
    assert manifest["record_count"] == len(records)
    assert manifest["sha256"]
    assert "relation" in manifest["category_counts"]
    out = tmp_path / "oracle_v1.jsonl"
    man = tmp_path / "oracle_v1.manifest.json"
    write_frozen_dataset(records, manifest, out, man, force=True)
    assert out.exists() and man.exists()
    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(loaded) == len(records)


def test_committed_oracle_v1_matches_rebuild():
    assert _ORACLE_V1.exists(), "oracle_v1.jsonl must be committed"
    assert _ORACLE_MANIFEST.exists(), "oracle_v1.manifest.json must be committed"
    records, manifest = build_frozen_dataset(_QUESTIONS, _RELATIONS)
    committed = [
        json.loads(line)
        for line in _ORACLE_V1.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert committed == records
    committed_manifest = json.loads(_ORACLE_MANIFEST.read_text(encoding="utf-8"))
    assert committed_manifest["sha256"] == manifest["sha256"]
    assert committed_manifest["record_count"] == manifest["record_count"]


def test_pair_rows_and_validate_paired_artifact():
    oracle_rows = [{
        "question_id": "q1",
        "category": "answer",
        "answer": "A",
        "fact_accuracy": 0.9,
        "token_f1": 0.8,
        "gold_path_recall": None,
        "gold_entity_coverage": 1.0,
        "should_abstain": False,
        "predicted_abstain": False,
        "reasoning_action": "answer",
        "proof_valid": True,
        "provenance_coverage": 1.0,
        "latency_ms": 10.0,
    }]
    predicted_rows = [{
        "question_id": "q1",
        "category": "answer",
        "answer": "B",
        "fact_accuracy": 0.4,
        "token_f1": 0.5,
        "gold_path_recall": None,
        "gold_entity_coverage": 0.5,
        "should_abstain": False,
        "predicted_abstain": False,
        "reasoning_action": "answer",
        "proof_valid": False,
        "provenance_coverage": 0.5,
        "latency_ms": 12.0,
    }]
    paired = pair_rows(oracle_rows, predicted_rows)
    assert paired[0]["delta"]["fact_accuracy"] == 0.5
    artifact = {
        "schema_version": "nexus-oracle-vs-predicted-v1",
        "oracle": {"evaluation_mode": "oracle", "metrics": summarize_rows(oracle_rows)},
        "predicted": {"evaluation_mode": "predicted", "metrics": summarize_rows(predicted_rows)},
        "paired": paired,
        "dataset": {"file_sha256": "abc"},
    }
    assert validate_paired_artifact(artifact) == []
    artifact["predicted"]["evaluation_mode"] = "oracle"
    assert "predicted evaluation_mode mismatch" in validate_paired_artifact(artifact)
