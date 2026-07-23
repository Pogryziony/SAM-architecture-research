"""Phase 4 local Qwen identity, corpus leakage, and adapter unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nexus.baselines.local_qwen import (
    FROZEN_DECODING,
    LocalQwenAdapter,
    LocalQwenIdentity,
    LocalQwenUnavailableError,
    discover_local_qwen,
)
from nexus.baselines.rag_corpus import (
    assert_no_leakage,
    build_canonical_corpus,
    reciprocal_rank_fusion,
)
from nexus.evaluation.adjudication_io import export_dual_packets, merge_dual_responses
from nexus.evaluation.compare import compare_paired_artifacts
from nexus.evaluation.validate import ValidationError


def test_local_qwen_identity_hash_changes_with_digest():
    a = LocalQwenIdentity(
        runtime="ollama",
        runtime_version="0.32.1",
        model_name="qwen3.6:latest",
        digest="07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522",
        architecture="qwen35moe",
        parameter_size="36.0B",
        quantization="Q4_K_M",
        context_length=262144,
        embedding_length=2048,
        host="http://127.0.0.1:11434",
        think_disabled=True,
        decoding=dict(FROZEN_DECODING),
    )
    b = LocalQwenIdentity(
        runtime="ollama",
        runtime_version="0.32.1",
        model_name="qwen3.6:latest",
        digest="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        architecture="qwen35moe",
        parameter_size="36.0B",
        quantization="Q4_K_M",
        context_length=262144,
        embedding_length=2048,
        host="http://127.0.0.1:11434",
        think_disabled=True,
        decoding=dict(FROZEN_DECODING),
    )
    assert a.identity_hash != b.identity_hash


def test_discover_refuses_wrong_model_name():
    fake = {
        "models": [
            {
                "name": "qwen2.5:latest",
                "digest": "abc",
                "details": {"family": "qwen2", "parameter_size": "7B"},
            }
        ]
    }
    with patch("nexus.baselines.local_qwen._http_json", return_value=fake):
        with pytest.raises(LocalQwenUnavailableError):
            discover_local_qwen()


def test_adapter_rejects_unfrozen_decoding_keys():
    identity = LocalQwenIdentity(
        runtime="ollama",
        runtime_version="0.32.1",
        model_name="qwen3.6:latest",
        digest="07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522",
        architecture="qwen35moe",
        parameter_size="36.0B",
        quantization="Q4_K_M",
        context_length=262144,
        embedding_length=2048,
        host="http://127.0.0.1:11434",
        think_disabled=True,
    )
    adapter = LocalQwenAdapter(identity)
    with pytest.raises(ValueError, match="unfrozen"):
        adapter.generate("hi", decoding={**FROZEN_DECODING, "temperature": 0.9, "foo": 1})


def test_corpus_rejects_oracle_path(tmp_path: Path):
    bad = tmp_path / "benchmarks" / "qa-dataset" / "oracle_v1.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text('{"gold_answer":"x","oracle":true}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        assert_no_leakage(bad, bad.read_text(encoding="utf-8"))


def test_build_corpus_smoke():
    root = Path(__file__).resolve().parents[1]
    corpus = build_canonical_corpus(
        root,
        globs=("README.md", "docs/CURRENT_STATE.md"),
        chunk_size=400,
        overlap=40,
    )
    assert corpus["file_count"] >= 1
    assert corpus["chunk_count"] >= 1
    assert len(corpus["corpus_sha256"]) == 64


def test_rrf_fusion_stable():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], top_k=2)
    assert fused[0][0] in {"a", "b"}
    assert len(fused) == 2


def test_dual_packet_export(tmp_path: Path):
    questions = [
        {
            "id": "q1",
            "question": "Compare two long narrative systems in detail please?",
            "category": "comparative",
            "gold_answer": " ".join(["word"] * 20),
        }
    ]
    manifest = export_dual_packets(
        questions,
        {
            "nexus": {
                "q1": {
                    "final_answer": "n",
                    "citations": ["doc:1"],
                    "structured_evidence": {"k": "v"},
                }
            },
            "qwen": {"q1": {"final_answer": "q", "retrieved_documents": ["doc:2"]}},
        },
        tmp_path,
        require_evidence=True,
    )
    assert Path(manifest["annotator_A_packet"]).exists()
    assert Path(manifest["annotator_B_packet"]).exists()
    packet = Path(manifest["annotator_A_packet"]).read_text(encoding="utf-8")
    assert "citation:" in packet or "structured_evidence=" in packet


def test_pending_adjudication_still_blocks_compare():
    left = {
        "schema_version": "nexus-eval-result-v1",
        "created_utc": "2026-07-22T00:00:00+00:00",
        "source_commit": "s",
        "dataset_id": "d",
        "dataset_sha256": "a" * 64,
        "system_id": "sys_a",
        "profile": "p",
        "config_hash": "h",
        "questions_total": 1,
        "adjudication_status": "PENDING_ADJUDICATION",
        "per_question": [
            {
                "question_id": "q1",
                "domain": "sam",
                "question_type": "factual",
                "dataset_id": "d",
                "dataset_sha256": "a" * 64,
                "system_id": "sys_a",
                "profile": "p",
                "config_hash": "h",
                "config_identity_schema": "nexus-config-identity-v2",
                "model_id": "m",
                "checkpoint_id": "c",
                "source_commit": "s",
                "executed_at_utc": "2026-07-22T00:00:00+00:00",
                "terminal_outcome": "answered",
                "metrics": {
                    "grounded_correct": {
                        "applicable": True,
                        "value": 1.0,
                        "numerator": 1.0,
                        "denominator": 1.0,
                        "reason": "t",
                    }
                },
            }
        ],
        "aggregates": {},
        "status": "OK",
        "comparison_mode": "controlled",
    }
    right = json.loads(json.dumps(left))
    right["system_id"] = "sys_b"
    right["per_question"][0]["system_id"] = "sys_b"
    with pytest.raises(ValidationError, match="PENDING_ADJUDICATION"):
        compare_paired_artifacts(left, right)


def test_mock_generate_not_real_benchmark_label():
    """Unit fixtures must not look like real Qwen benchmark artifacts."""
    identity = LocalQwenIdentity(
        runtime="mock",
        runtime_version="0",
        model_name="qwen3.6:latest",
        digest="07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522",
        architecture="qwen35moe",
        parameter_size="36.0B",
        quantization="Q4_K_M",
        context_length=262144,
        embedding_length=2048,
        host="http://127.0.0.1:11434",
        think_disabled=True,
    )
    adapter = LocalQwenAdapter(identity)

    def fake_http(url, payload=None, timeout=30.0):
        return {
            "response": "OK",
            "eval_count": 1,
            "eval_duration": 1_000_000,
            "prompt_eval_count": 1,
            "prompt_eval_duration": 1_000_000,
            "load_duration": 0,
            "total_duration": 2_000_000,
        }

    with patch("nexus.baselines.local_qwen._http_json", side_effect=fake_http):
        gen = adapter.generate("Reply OK")
    assert gen.parsed_answer == "OK"
    # Explicit: this test path is a mock protocol fixture
    assert identity.runtime == "mock"
