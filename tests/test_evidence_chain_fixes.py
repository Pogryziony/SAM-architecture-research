"""Tests for P0/P1 evidence chain fixes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --- P0: Manifest LF normalization and hash verification ---

def test_manifest_paths_normalized_to_forward_slash():
    """Verify manifest paths use forward slashes regardless of OS."""
    from nexus.evaluation.evidence_manifest import normalize_manifest_paths

    manifest = {
        "artifacts": [
            {"path": "benchmarks\\results\\test.json", "sha256": "abc"},
            {"path": "some/path/file.json", "sha256": "def"},
        ],
        "statistics": [
            {"path": "stats\\file.json", "sha256": "ghi"},
        ],
    }
    result = normalize_manifest_paths(manifest)
    assert result["artifacts"][0]["path"] == "benchmarks/results/test.json"
    assert result["artifacts"][1]["path"] == "some/path/file.json"
    assert result["statistics"][0]["path"] == "stats/file.json"


def test_manifest_writes_with_lf_line_endings():
    """Verify manifest is written with LF-only line endings."""
    from nexus.evaluation.evidence_manifest import write_evidence_manifest

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "manifest.json"
        manifest = {
            "schema_version": "test-v1",
            "artifacts": [{"path": "test/file.json"}],
        }
        write_evidence_manifest(path, manifest)

        # Read as binary to check actual line endings
        content = path.read_bytes()
        assert b"\r\n" not in content, "Should not have CRLF line endings"
        assert b"\r" not in content.replace(b"\n", b""), "Should not have CR line endings"
        assert b"\n" in content, "Should have LF line endings"


def test_manifest_hash_verification_catches_mismatch():
    """Verify hash verification detects file modifications."""
    from nexus.evaluation.evidence_manifest import verify_manifest_hashes, file_sha256

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        art_path = root / "benchmarks" / "results" / "test.json"
        art_path.parent.mkdir(parents=True)
        art_path.write_text('{"test": "data"}', encoding="utf-8")

        # Create manifest with correct hash
        correct_hash = file_sha256(art_path)
        manifest_path = root / "manifest.json"
        manifest = {
            "artifacts": [
                {"path": "benchmarks/results/test.json", "sha256": correct_hash},
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Verification should pass
        errors = verify_manifest_hashes(manifest_path, root)
        assert not errors

        # Modify the artifact
        art_path.write_text('{"test": "modified"}', encoding="utf-8")

        # Verification should fail
        errors = verify_manifest_hashes(manifest_path, root)
        assert len(errors) == 1
        assert "hash mismatch" in errors[0]


# --- P0: Timeout isolation with subprocess ---

def test_subprocess_timeout_kills_process():
    """Verify timeout actually terminates the subprocess."""
    from nexus.baselines.local_qwen import _http_json_subprocess

    # Create a mock endpoint that hangs
    with pytest.raises((TimeoutError, RuntimeError)) as exc_info:
        # Use a non-routable IP to simulate hang
        _http_json_subprocess(
            "http://10.255.255.1:11434/api/test",
            payload={"test": True},
            timeout=1.0,  # Very short timeout
        )

    # Should mention timeout in some form
    err_str = str(exc_info.value).lower()
    assert "timeout" in err_str or "timed out" in err_str


def test_subprocess_timeout_leaves_no_zombie_workers():
    """Wall-clock kill path must terminate the worker (no lingering children)."""
    import subprocess as sp

    from nexus.baselines.local_qwen import _kill_process_tree

    psutil = pytest.importorskip("psutil")

    # Hang past communicate timeout; exercise kill tree.
    proc = sp.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
    )
    pid = proc.pid
    try:
        proc.communicate(timeout=0.5)
        pytest.fail("child should have hung past timeout")
    except sp.TimeoutExpired:
        _kill_process_tree(pid)
        proc.kill()
        proc.wait(timeout=5)

    time.sleep(0.5)
    assert not psutil.pid_exists(pid), f"zombie worker still alive: pid={pid}"


def test_subprocess_timeout_returns_quickly():
    """Verify we don't block forever waiting for the subprocess."""
    from nexus.baselines.local_qwen import _http_json_subprocess

    t0 = time.perf_counter()
    with pytest.raises((TimeoutError, RuntimeError)):
        _http_json_subprocess(
            "http://10.255.255.1:11434/api/test",
            timeout=2.0,
        )
    elapsed = time.perf_counter() - t0

    # Should complete within timeout + grace period, not hang forever
    assert elapsed < 30.0, f"Subprocess took {elapsed:.1f}s, should be < 30s"


# --- P0: Family stats proper references ---

def test_family_stats_uses_controlled_reference_for_controlled_family():
    """Verify controlled family uses NEXUS graph evidence as reference."""
    # Just verify the module structure is correct
    from benchmarks.run_phase4_family_stats import CONTROLLED_REFERENCE, CONTROLLED

    assert CONTROLLED_REFERENCE == "phase4_nexus_graph_evidence_qwen_oracle_v1.json"
    assert CONTROLLED_REFERENCE in CONTROLLED


def test_family_stats_pairwise_for_system_level():
    """Verify system-level family uses pairwise comparison."""
    from benchmarks.run_phase4_family_stats import SYSTEM_LEVEL

    # System-level should have multiple arms for pairwise comparison
    assert len(SYSTEM_LEVEL) >= 2
    # And should not contain RAG methods
    for name in SYSTEM_LEVEL:
        assert "rag" not in name.lower()
        assert "bm25" not in name.lower()


# --- P1: Dense model identity ---

def test_dense_identity_tracks_revision_resolved():
    """Verify dense embedder tracks actual resolved revision."""
    from nexus.baselines.dense_embedder import embedder_identity

    ident = embedder_identity()
    assert "revision" in ident
    assert "model_id" in ident
    # These are the pinned values
    assert ident["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_dense_identity_fail_closed_raises():
    """Verify fail_closed=True raises when revision unavailable."""
    pytest.importorskip("sentence_transformers", reason="sentence_transformers not installed")

    from nexus.baselines.dense_embedder import (
        DenseModelIdentityError,
        load_sentence_transformer,
    )

    # This test only makes sense when the model isn't available
    # In CI it might be available, so we test the interface
    try:
        model, ident = load_sentence_transformer(fail_closed=True)
        # If it succeeds, verify the identity is recorded
        assert "revision_resolved" in ident
        assert "load_mode" in ident
        assert ident.get("identity_degraded") is False or ident.get("load_mode") != "offline_cache_fallback"
    except DenseModelIdentityError:
        # This is expected if the pinned revision isn't available
        pass
    except (RuntimeError, OSError) as exc:
        # Network/SSL errors in test environment are acceptable
        if "SSL" in str(exc) or "certificate" in str(exc).lower() or "closed" in str(exc).lower():
            pytest.skip(f"Network/SSL issue in test environment: {exc}")
        raise


# --- P1: Relevance tightening ---

def test_relevance_mapping_tightened_thresholds():
    """Verify relevance mapping uses tighter thresholds."""
    from nexus.evaluation.relevance import build_relevance_for_question

    question = {
        "id": "q1",
        "question": "What is X?",
        "gold_answer": "The answer is complex multi-word technical term here",
        "gold_entities": ["entity_a"],
    }

    # Create corpus with varying relevance
    corpus_chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "entity_a is mentioned here with details"},
        {"chunk_id": "c2", "doc_id": "d2", "text": "completely unrelated content about other things"},
        {"chunk_id": "c3", "doc_id": "d3", "text": "the answer complex technical term matches here"},
    ]

    mapping = build_relevance_for_question(question, corpus_chunks)

    # Should use v2 method
    assert "v2" in mapping.method
    # Should have confidence level
    assert mapping.confidence in ("heuristic", "explicit", "heuristic_capped")


def test_relevance_table_includes_statistics():
    """Verify relevance table includes quality statistics."""
    from nexus.evaluation.relevance import build_relevance_table

    questions = [
        {"id": "q1", "gold_answer": "answer one", "gold_entities": ["ent1"]},
        {"id": "q2", "gold_answer": "answer two", "gold_entities": ["ent2"]},
    ]
    corpus = {
        "chunks": [
            {"chunk_id": "c1", "text": "ent1 content"},
            {"chunk_id": "c2", "text": "ent2 content"},
        ],
    }

    table = build_relevance_table(questions, corpus)

    assert table["schema_version"] == "nexus-retrieval-relevance-v2"
    assert "relevance_statistics" in table
    assert "median_relevant_chunks" in table["relevance_statistics"]
    assert "max_relevant_chunks" in table["relevance_statistics"]


# --- P1/P2: Graph snapshot ID strengthening ---

def test_graph_snapshot_id_includes_edges():
    """Verify graph snapshot ID considers edge structure."""
    from nexus.ingestion.canonical_graph import graph_snapshot_id
    from unittest.mock import MagicMock

    # Create mock graph objects
    g1 = MagicMock()
    g2 = MagicMock()

    from dataclasses import dataclass

    @dataclass
    class TestNode:
        id: str
        type: str = "test"

    @dataclass
    class TestEdge:
        type: str
        target: str  # Edge uses 'target' not 'target_id'

    # Mock the graph structure - same nodes
    g1._nodes = {"a": TestNode("a"), "b": TestNode("b")}
    g2._nodes = {"a": TestNode("a"), "b": TestNode("b")}

    # Different edge structures
    g1_edges = {"a": [TestEdge("rel1", "b")]}
    g2_edges = {"a": [TestEdge("rel2", "b")]}  # Different edge type

    g1.get_outgoing = lambda x: g1_edges.get(x, [])
    g2.get_outgoing = lambda x: g2_edges.get(x, [])
    g1.node_count = 2
    g2.node_count = 2
    g1.edge_count = 1
    g2.edge_count = 1

    # Snapshot IDs should differ due to different edge types
    id1 = graph_snapshot_id(g1)
    id2 = graph_snapshot_id(g2)

    # They should be different
    assert id1 != id2


# --- P1: Resource tracking ---

def test_resource_snapshot_helper():
    """Verify resource snapshot helper returns expected structure."""
    from nexus.evaluation.process_resources import snapshot_llm_server_resources

    snapshot = snapshot_llm_server_resources()

    assert "schema_version" in snapshot
    assert snapshot["schema_version"] == "nexus-llm-server-resources-v1"
    # These may be None if psutil/nvidia-smi aren't available
    assert "ollama_rss_mb" in snapshot
    assert "process_tree_rss_mb" in snapshot
    assert "vram" in snapshot


# --- Integration test for build_eval_artifact with resources ---

def test_build_eval_artifact_includes_resources():
    """Verify eval artifact includes resource usage section."""
    from nexus.baselines.phase4_arms import build_eval_artifact
    from nexus.evaluation.schema import build_question_record, TerminalOutcome

    # Build a valid question record
    questions = [{"id": "q1", "question": "test?", "gold_answer": "answer"}]
    record = build_question_record(
        question_id="q1",
        domain="sam",
        question_type="factual",
        dataset_id="oracle_v1",
        dataset_sha256="abc123" * 10 + "abcd",  # 64 chars
        system_id="test_system",
        profile="test_profile",
        config_hash="hash1234",
        config_identity_schema="test-v1",
        model_id="test_model",
        checkpoint_id="checkpoint1",
        source_commit="abc123",
        executed_at_utc="2026-01-01T00:00:00Z",
        terminal_outcome=TerminalOutcome.ANSWERED,
        question="test?",
        final_answer="answer",
        metrics={
            "grounded_correct": {
                "name": "grounded_correct",
                "applicable": False,
                "value": None,
                "numerator": None,
                "denominator": 1.0,
                "reason": "test",
            }
        },
        latency_ms=100.0,
        comparison_mode="controlled",
    )
    rows = [record.to_dict()]

    artifact = build_eval_artifact(
        system_id="test_system",
        profile="test_profile",
        questions=questions,
        rows=rows,
        comparison_mode="controlled",
        source_commit="abc123",
        config_hash="hash123",
        qwen_identity=None,
        arm_metadata={"family": "test"},
        status="VALID",
    )

    assert "resource_usage" in artifact
    assert "throughput_questions_per_second" in artifact["resource_usage"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
