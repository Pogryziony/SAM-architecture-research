"""Provenance gates: source_commit must own the claimed dataset identity."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from nexus.evaluation.dataset_identity import hash_dataset
from nexus.evaluation.validate import ValidationError

# Commits known to predate the ca96877 oracle identity; claiming them with
# ca96877 is always a provenance failure (dirty-tree / metadata rebind).
FORBIDDEN_SOURCE_COMMITS_FOR_CA96877 = frozenset(
    {
        "b4d7e9e0314bd9363fdc392e6db17ed1f890c277",
        "6a13f8e260b192e58ca6a3a6815ec3ff1029d89e",
    }
)

CANONICAL_DATASET_SHA256 = (
    "ca96877de86990e7757c18efe3576ec660b454d6984866cbebc5939ead1a63d5"
)

PRE_CA96877_DATASET_SHA256 = (
    "568f9ce4544426e092cb5e1dfc2abbb6f0ad2bf2ec146a7f7b309b3b438e5dfc"
)

# Shallow CI checkouts often lack historical blobs; keep audited identities here.
# Values are oracle_v1.manifest.json sha256 (canonical dataset identity).
ORACLE_SHA_AT_COMMIT: dict[str, str] = {
    "b4d7e9e0314bd9363fdc392e6db17ed1f890c277": PRE_CA96877_DATASET_SHA256,
    "6a13f8e260b192e58ca6a3a6815ec3ff1029d89e": PRE_CA96877_DATASET_SHA256,
    "75748b74e7ec0d1bd48ceec0646edab5f6a208c3": CANONICAL_DATASET_SHA256,
    "93cd009a8d6ccb91f48a436395f2d62006f31470": CANONICAL_DATASET_SHA256,
}

ORACLE_MANIFEST_REL = "benchmarks/qa-dataset/oracle_v1.manifest.json"
ORACLE_JSONL_REL = "benchmarks/qa-dataset/oracle_v1.jsonl"


def _git_show_text(root: Path, commit: str, rel_path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{rel_path}"],
        cwd=root,
        text=True,
        stderr=subprocess.DEVNULL,
    )


def _git_rev_parse(root: Path, rev: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", rev],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _lookup_known_oracle_sha(commit: str) -> str | None:
    if commit in ORACLE_SHA_AT_COMMIT:
        return ORACLE_SHA_AT_COMMIT[commit]
    for known, sha in ORACLE_SHA_AT_COMMIT.items():
        if len(commit) >= 12 and (
            known.startswith(commit) or commit.startswith(known[:12])
        ):
            return sha
    return None


def _working_tree_oracle_sha(root: Path) -> str | None:
    jsonl_path = root / ORACLE_JSONL_REL
    if not jsonl_path.exists():
        return None
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return hash_dataset(rows)


def oracle_dataset_sha256_at_commit(root: Path, commit: str) -> str:
    """Return the oracle dataset identity at ``commit``.

    Order: audited commit map (shallow-CI safe) → git manifest → git jsonl →
    working-tree hash when ``commit`` resolves to HEAD.
    """
    if not commit or commit == "UNKNOWN":
        raise ValidationError("artifact source_commit is missing/UNKNOWN")

    known = _lookup_known_oracle_sha(commit)
    if known:
        return known

    try:
        manifest_text = _git_show_text(root, commit, ORACLE_MANIFEST_REL)
        manifest = json.loads(manifest_text)
        sha = str(manifest.get("sha256") or "")
        if len(sha) == 64:
            return sha
    except Exception:
        pass

    try:
        jsonl = _git_show_text(root, commit, ORACLE_JSONL_REL)
        rows = [json.loads(line) for line in jsonl.splitlines() if line.strip()]
        return hash_dataset(rows)
    except Exception:
        pass

    head = _git_rev_parse(root, "HEAD")
    resolved = _git_rev_parse(root, commit) or commit
    if head and (
        resolved == head
        or head.startswith(commit)
        or commit.startswith(head[:12])
    ):
        wt = _working_tree_oracle_sha(root)
        if wt:
            return wt

    raise ValidationError(
        f"source_commit {commit[:12]} oracle identity unavailable "
        "(shallow clone missing blob; add to ORACLE_SHA_AT_COMMIT)"
    )


def assert_source_commit_owns_dataset(
    artifact: Mapping[str, Any],
    *,
    root: Path,
    name: str = "artifact",
) -> None:
    """Fail closed when source_commit's oracle identity ≠ artifact dataset_sha256."""
    source = str(artifact.get("source_commit") or "")
    dataset_sha = str(artifact.get("dataset_sha256") or "")
    if not source or not dataset_sha:
        raise ValidationError(f"{name} missing source_commit or dataset_sha256")

    if (
        dataset_sha == CANONICAL_DATASET_SHA256
        and source in FORBIDDEN_SOURCE_COMMITS_FOR_CA96877
    ):
        raise ValidationError(
            f"{name} claims source_commit {source[:12]} with dataset "
            f"{dataset_sha[:12]}… — that commit predates the canonical oracle "
            "identity; regenerate arms from a clean checkout"
        )

    commit_sha = oracle_dataset_sha256_at_commit(root, source)
    if commit_sha != dataset_sha:
        raise ValidationError(
            f"{name} source_commit {source[:12]} has oracle dataset "
            f"{commit_sha[:12]}… but artifact claims {dataset_sha[:12]}…"
        )


def dense_embedding_identity_ok(arm_metadata: Mapping[str, Any]) -> list[str]:
    """Return errors if dense embedding fields omit truthful load identity."""
    errors: list[str] = []
    candidates: list[Mapping[str, Any]] = [arm_metadata]
    nested = arm_metadata.get("dense")
    if isinstance(nested, dict):
        candidates.append(nested)

    saw_dense = False
    for meta in candidates:
        method = str(meta.get("retrieval_method") or "")
        has_embed = "embedding_model" in meta or "embedding_revision" in meta
        if not (
            method == "dense"
            or has_embed
            and "embedding_identity_sha256" in meta
        ):
            continue
        saw_dense = True
        if "embedding_revision" in meta and "embedding_revision_resolved" not in meta:
            errors.append(
                "dense metadata uses legacy embedding_revision without "
                "embedding_revision_resolved / embedding_load_mode"
            )
        if "embedding_revision_resolved" not in meta:
            errors.append("missing embedding_revision_resolved")
        if "embedding_load_mode" not in meta:
            errors.append("missing embedding_load_mode")
    if not saw_dense:
        return []
    return errors
