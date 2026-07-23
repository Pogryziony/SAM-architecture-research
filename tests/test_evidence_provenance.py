"""CI gates for evidence provenance (source_commit ↔ dataset identity)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus.evaluation.evidence_provenance import (
    CANONICAL_DATASET_SHA256,
    FORBIDDEN_SOURCE_COMMITS_FOR_CA96877,
    assert_source_commit_owns_dataset,
    dense_embedding_identity_ok,
    oracle_dataset_sha256_at_commit,
)
from nexus.evaluation.validate import ValidationError

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"

PHASE4_ARMS = [
    "phase4_qwen_closed_book_oracle_v1.json",
    "phase4_qwen_long_context_oracle_v1.json",
    "phase4_bm25_rag_qwen_oracle_v1.json",
    "phase4_dense_rag_qwen_oracle_v1.json",
    "phase4_hybrid_rag_qwen_oracle_v1.json",
    "phase4_hybrid_rerank_rag_qwen_oracle_v1.json",
    "phase4_nexus_graph_evidence_qwen_oracle_v1.json",
    "eval_oracle_v1_grounded_evidence_repair.json",
]

DENSE_ARMS = [
    "phase4_dense_rag_qwen_oracle_v1.json",
    "phase4_hybrid_rag_qwen_oracle_v1.json",
    "phase4_hybrid_rerank_rag_qwen_oracle_v1.json",
]


def test_forbidden_commits_do_not_own_ca96877():
    for commit in FORBIDDEN_SOURCE_COMMITS_FOR_CA96877:
        sha = oracle_dataset_sha256_at_commit(ROOT, commit)
        assert sha != CANONICAL_DATASET_SHA256
        with pytest.raises(ValidationError, match="predate|predates|claims source_commit"):
            assert_source_commit_owns_dataset(
                {
                    "source_commit": commit,
                    "dataset_sha256": CANONICAL_DATASET_SHA256,
                },
                root=ROOT,
                name="probe",
            )


def test_head_owns_canonical_dataset():
    import subprocess

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    sha = oracle_dataset_sha256_at_commit(ROOT, head)
    assert sha == CANONICAL_DATASET_SHA256


@pytest.mark.parametrize("name", PHASE4_ARMS)
def test_phase4_arm_source_commit_owns_dataset(name: str):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    art = json.loads(path.read_text(encoding="utf-8"))
    assert art.get("dataset_sha256") == CANONICAL_DATASET_SHA256
    assert_source_commit_owns_dataset(art, root=ROOT, name=name)
    source = art["source_commit"]
    assert source not in FORBIDDEN_SOURCE_COMMITS_FOR_CA96877


@pytest.mark.parametrize("name", DENSE_ARMS)
def test_dense_arm_metadata_has_revision_resolved(name: str):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    art = json.loads(path.read_text(encoding="utf-8"))
    errors = dense_embedding_identity_ok(art.get("arm_metadata") or {})
    assert not errors, f"{name}: " + "; ".join(errors)


def test_dense_identity_helper_flags_legacy_revision_only():
    errors = dense_embedding_identity_ok(
        {
            "retrieval_method": "dense",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "embedding_identity_sha256": "a" * 64,
        }
    )
    assert errors
