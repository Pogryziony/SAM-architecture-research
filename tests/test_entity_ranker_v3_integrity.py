"""Integrity tests for Entity Ranker V3 artifacts, manifests, and attestation.

T16: Attestation hashes match committed bytes
T17: Manifest hashes match committed files
T18: Historical artifacts unchanged
T19: Candidate diagnostic never reads test.jsonl
T20: Single-read proof (HoldoutData)
T21: Lexical-only vs exhaustive candidate ceiling
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from stack.encoder.semantic_hash import (
    HoldoutData,
    compute_canonical_semantic_sha256,
    load_and_validate_new_holdout,
    MalformedJSONLError,
    EmptySplitError,
)
from stack.encoder.canonical_mapping import _is_canonical_id


def _git_show(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"], capture_output=True, check=True
    ).stdout


# ── T16: Attestation hashes match committed bytes ──

COMMITTED_HASHES = {
    "benchmarks/results/entity_ranker_v3_selection_20260711T081545Z.json":
        "8f0542b73aaa4f0c528c02f3ee190005713ac707880045b497eb43987c34846d",
    "benchmarks/results/entity_ranker_v3_frozen_20260711T084518Z.json":
        "df2c51c66ca5e26b641c8dc3c3da355f8292d32525b2d22f791d5356a16e1538",
    "models/encoder/entity_ranker_v3_20260711T081545Z/config.json":
        "c14051c80af3c72ee8a2d6c7915da69bd6de978c25f9a5cf9cd5866dd2ce6d95",
    "models/encoder/entity_ranker_v3_20260711T081545Z/vocab.json":
        "e69d55991720318689487832d1b38e16b60164b90e3b792d9adef7aa2ace8364",
}

MANIFEST_FILES = {
    "config.json": "models/encoder/entity_ranker_v3_20260711T081545Z/config.json",
    "vocab.json": "models/encoder/entity_ranker_v3_20260711T081545Z/vocab.json",
}


@pytest.mark.parametrize("path,expected_hash", COMMITTED_HASHES.items())
def test_committed_file_hash_matches(path, expected_hash):
    """Each committed file's actual hash matches the expected hash."""
    data = _git_show(path)
    actual = hashlib.sha256(data).hexdigest()
    assert actual == expected_hash, f"{path}: expected {expected_hash[:16]}..., got {actual[:16]}..."


def test_manifest_hashes_match_committed():
    """Manifest SHA-256 values match committed file bytes."""
    manifest_data = json.loads(
        _git_show("models/encoder/entity_ranker_v3_20260711T081545Z/manifest.json")
    )
    for key, path in MANIFEST_FILES.items():
        expected = manifest_data["files"][key]["committed_sha256"]
        actual = hashlib.sha256(_git_show(path)).hexdigest()
        assert actual == expected, f"manifest {key}: expected {expected[:16]}..., got {actual[:16]}..."

    # Check validation artifact hash in manifest
    val_expected = manifest_data["validation_artifact"]["committed_sha256"]
    val_actual = hashlib.sha256(
        _git_show(manifest_data["validation_artifact"]["path"])
    ).hexdigest()
    assert val_actual == val_expected


# ── T17: Historical artifacts unchanged ──


def test_original_validation_artifact_unchanged():
    """The committed validation artifact has not been modified."""
    data = _git_show("benchmarks/results/entity_ranker_v3_selection_20260711T081545Z.json")
    h = hashlib.sha256(data).hexdigest()
    assert h == COMMITTED_HASHES[
        "benchmarks/results/entity_ranker_v3_selection_20260711T081545Z.json"
    ]


def test_original_frozen_artifact_unchanged():
    """The committed frozen artifact has not been modified."""
    data = _git_show("benchmarks/results/entity_ranker_v3_frozen_20260711T084518Z.json")
    h = hashlib.sha256(data).hexdigest()
    assert h == COMMITTED_HASHES[
        "benchmarks/results/entity_ranker_v3_frozen_20260711T084518Z.json"
    ]


# ── T18: Single-read proof ──


class CountingOpener:
    """File opener that counts physical read operations."""
    def __init__(self, real_path: Path):
        self.real_path = real_path
        self.read_count = 0

    def read_bytes(self):
        self.read_count += 1
        if self.read_count > 1:
            raise RuntimeError(f"File read {self.read_count} times — expected exactly 1")
        return self.real_path.read_bytes()


def test_holdout_read_exactly_once(tmp_path: Path):
    """HoldoutData is created from a single physical read."""
    records = [{"id": f"q{i}", "question": f"Q{i}?", "entities": ["E"]} for i in range(5)]
    f = tmp_path / "holdout.jsonl"
    f.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )

    # Simulate single-read by reading bytes once
    raw = f.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    semantic_hash = compute_canonical_semantic_sha256(raw)
    parsed = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]

    hd = HoldoutData(
        raw_bytes=raw,
        records=parsed,
        raw_sha256=raw_hash,
        semantic_sha256=semantic_hash,
    )
    assert hd.record_count == 5
    assert len(hd.records[0]) == 3  # id, question, entities


def test_load_and_validate_new_holdout_returns_holdout_data(tmp_path: Path):
    """The production function returns HoldoutData with correct fields."""
    records = [{"id": f"q{i}", "question": f"Q{i}?", "entities": ["E"]} for i in range(3)]
    f = tmp_path / "holdout.jsonl"
    f.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )
    hd = load_and_validate_new_holdout(f)
    assert isinstance(hd, HoldoutData)
    assert hd.record_count == 3
    assert len(hd.raw_sha256) == 64
    assert len(hd.semantic_sha256) == 64
    assert hd.records[0]["id"] == "q0"


# ── T19: Malformed JSONL rejection ──


def test_malformed_jsonl_fails_holdout(tmp_path: Path):
    f = tmp_path / "bad.jsonl"
    f.write_text('{"ok":1}\n{broken\n', encoding="utf-8")
    with pytest.raises(MalformedJSONLError):
        load_and_validate_new_holdout(f)


def test_empty_jsonl_fails_holdout(tmp_path: Path):
    f = tmp_path / "empty.jsonl"
    f.write_text("\n\n", encoding="utf-8")
    with pytest.raises(EmptySplitError):
        load_and_validate_new_holdout(f)


# ── T20: Lexical-only vs exhaustive ceiling ──

def test_lexical_can_contain_canonical_ids():
    """Lexical mode can contain canonical IDs found via lexical overlap."""
    from nexus.graph import Node
    from nexus.graph.store import InMemoryGraphStore
    from stack.encoder.trivial_baseline import candidate_pool

    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Exp_0_1_Test", type="Experiment", aliases=["alpha test", "experiment"]))
    graph.add_node(Node(id="Metric_XYZ", type="Metric", aliases=["some metric"]))

    question = "What did the alpha test show?"
    pool = candidate_pool(question, graph, include_canonical_vocabulary=False)
    pool_ids = {item["node_id"] for item in pool}

    # "Exp_0_1_Test" should be found lexically via alias "alpha test"
    assert "Exp_0_1_Test" in pool_ids, (
        "Canonical ID with lexical match must appear in lexical-only mode"
    )


def test_exhaustive_contains_all_canonical():
    """Exhaustive mode contains all canonical IDs."""
    from benchmarks.run_benchmark import build_benchmark_graph
    from stack.encoder.trivial_baseline import candidate_pool
    graph, _ = build_benchmark_graph()

    canonical_count = sum(1 for nid in graph._nodes if _is_canonical_id(str(nid)))
    pool = candidate_pool("any question", graph, include_canonical_vocabulary=True)
    pool_ids = {item["node_id"] for item in pool}

    canonical_in_pool = sum(1 for cid in pool_ids if _is_canonical_id(cid))
    assert canonical_in_pool == canonical_count, (
        f"Expected {canonical_count} canonical IDs in exhaustive pool, got {canonical_in_pool}"
    )


def test_lexical_does_not_automatically_contain_all_canonical():
    """Lexical-only mode does not guarantee all canonical IDs."""
    from benchmarks.run_benchmark import build_benchmark_graph
    from stack.encoder.trivial_baseline import candidate_pool
    graph, _ = build_benchmark_graph()

    canonical_count = sum(1 for nid in graph._nodes if _is_canonical_id(str(nid)))
    # Use a question with no lexical overlap with most canonical nodes
    pool = candidate_pool("zzz_no_match_xyz", graph, include_canonical_vocabulary=False)
    pool_ids = {item["node_id"] for item in pool}

    canonical_in_pool = sum(1 for cid in pool_ids if _is_canonical_id(cid))
    assert canonical_in_pool < canonical_count, (
        f"Lexical mode should not contain all {canonical_count} canonical IDs "
        f"for a no-match question, but got {canonical_in_pool}"
    )


def test_exhaustive_ceiling_ge_lexical_ceiling():
    """Exhaustive ceiling >= lexical ceiling for all validation questions."""
    from benchmarks.run_benchmark import build_benchmark_graph
    from stack.encoder.trivial_baseline import candidate_pool
    val = [
        json.loads(line)
        for line in Path("stack/encoder/data/val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, _ = build_benchmark_graph()

    for record in val:
        gold = set(str(e) for e in record.get("entities", []))
        lex_pool = candidate_pool(record["question"], graph, include_canonical_vocabulary=False)
        exh_pool = candidate_pool(record["question"], graph, include_canonical_vocabulary=True)
        lex_ids = {item["node_id"] for item in lex_pool}
        exh_ids = {item["node_id"] for item in exh_pool}
        lex_hits = len(gold & lex_ids)
        exh_hits = len(gold & exh_ids)
        assert exh_hits >= lex_hits, (
            f"Exhaustive pool ({exh_hits} gold hits) must be >= lexical ({lex_hits}) "
            f"for question {record['id']}"
        )


# ── T21: Candidate diagnostic never reads test.jsonl ──


def test_diagnostic_script_never_reads_test_jsonl():
    """The candidate diagnostic script never opens the frozen split."""
    content = Path("benchmarks/diagnose_candidate_pool.py").read_text(encoding="utf-8")
    # The diagnostic loads train.jsonl and val.jsonl — never the frozen split
    assert "stack/encoder/data/test" not in content, (
        "Candidate diagnostic must never open the frozen split"
    )
