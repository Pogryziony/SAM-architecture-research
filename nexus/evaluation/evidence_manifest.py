"""End-to-end evidence manifest binding commit → rows → stats → claim."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _git(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def current_checkout_identity(root: Path) -> dict[str, Any]:
    commit = _git(["git", "rev-parse", "HEAD"], root)
    tree = _git(["git", "rev-parse", "HEAD^{tree}"], root)
    dirty = _git(["git", "status", "--porcelain"], root)
    return {
        "source_commit": commit or "UNKNOWN",
        "source_tree": tree or "UNKNOWN",
        "working_tree_dirty": bool(dirty),
    }


def lf_normalize_bytes(data: bytes) -> bytes:
    """Normalize CRLF/CR to LF so Windows checkouts match git blob hashes."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def file_sha256(path: Path, *, lf_normalize: bool = True) -> str:
    """SHA-256 of file bytes, LF-normalized by default for cross-platform stability."""
    data = path.read_bytes()
    if lf_normalize:
        data = lf_normalize_bytes(data)
    return hashlib.sha256(data).hexdigest()


def git_blob_sha256(root: Path, rel_path: str, *, lf_normalize: bool = True) -> str | None:
    """SHA-256 of the as-in-git blob for ``rel_path``, or None if unavailable."""
    rel = rel_path.replace("\\", "/")
    try:
        raw = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    if lf_normalize:
        raw = lf_normalize_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _normalize_path_for_manifest(p: Path, root: Path) -> str:
    """Convert path to forward-slash relative path for manifest."""
    if p.is_relative_to(root):
        rel = str(p.relative_to(root))
    else:
        rel = str(p)
    # Always use forward slashes for cross-platform consistency
    return rel.replace("\\", "/")


def build_evidence_manifest(
    *,
    root: Path,
    dataset_id: str,
    dataset_sha256: str,
    graph_snapshot_id: str,
    model_identities: Mapping[str, Any],
    config_hashes: Mapping[str, str],
    prompt_sha256: Mapping[str, str],
    artifact_paths: Sequence[Path],
    adjudication_status: str,
    statistics_paths: Sequence[Path] | None = None,
    claim_eligibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkout = current_checkout_identity(root)
    arts = []
    for p in artifact_paths:
        if not p.exists():
            arts.append({"path": _normalize_path_for_manifest(p, root), "missing": True})
            continue
        arts.append(
            {
                "path": _normalize_path_for_manifest(p, root),
                "sha256": file_sha256(p),
                "bytes": p.stat().st_size,
            }
        )
    stats = []
    for p in statistics_paths or []:
        if p.exists():
            stats.append(
                {
                    "path": _normalize_path_for_manifest(p, root),
                    "sha256": file_sha256(p),
                }
            )
    manifest = {
        "schema_version": "nexus-evidence-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkout": checkout,
        "dataset": {"dataset_id": dataset_id, "dataset_sha256": dataset_sha256},
        "graph_snapshot_id": graph_snapshot_id,
        "model_identities": dict(model_identities),
        "config_hashes": dict(config_hashes),
        "prompt_sha256": dict(prompt_sha256),
        "artifacts": arts,
        "adjudication_status": adjudication_status,
        "statistics": stats,
        "claim_eligibility": dict(
            claim_eligibility
            or {
                "full_primary_superiority": False,
                "reason": "explicit gate; require complete adjudication + sealed external",
            }
        ),
    }
    # Normalize paths before hashing for cross-platform consistency
    normalized = normalize_manifest_paths(manifest)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["manifest_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return manifest


def normalize_manifest_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    """Convert Windows backslash paths to forward slashes for cross-platform hash stability."""
    result = dict(manifest)
    if "artifacts" in result:
        result["artifacts"] = [
            {**a, "path": a.get("path", "").replace("\\", "/")}
            for a in result["artifacts"]
        ]
    if "statistics" in result:
        result["statistics"] = [
            {**s, "path": s.get("path", "").replace("\\", "/")}
            for s in result["statistics"]
        ]
    return result


def write_evidence_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write manifest with explicit LF line endings for hash stability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize paths and ensure consistent LF line endings
    normalized = normalize_manifest_paths(dict(manifest))
    content = json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    # Ensure LF-only line endings for cross-platform hash consistency
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    path.write_bytes(content.encode("utf-8"))


def verify_manifest_hashes(
    manifest_path: Path,
    root: Path,
    *,
    prefer_git_blob: bool = True,
) -> list[str]:
    """Verify all artifact hashes match LF-normalized file bytes (or as-in-git blobs).

    Returns a list of error messages (empty if all hashes match).
    """
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"manifest not found: {manifest_path}"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for art in manifest.get("artifacts", []):
        art_path_str = art.get("path", "")
        if not art_path_str or art.get("missing"):
            continue
        # Normalize path separators
        art_path_str = art_path_str.replace("\\", "/")
        art_path = root / art_path_str
        if not art_path.exists():
            errors.append(f"artifact missing: {art_path_str}")
            continue
        expected_hash = art.get("sha256")
        if not expected_hash:
            continue

        disk_hash = file_sha256(art_path, lf_normalize=True)
        if disk_hash == expected_hash:
            continue
        git_hash = (
            git_blob_sha256(root, art_path_str, lf_normalize=True)
            if prefer_git_blob
            else None
        )
        # Accept as-in-git when CRLF checkout mutates working-tree bytes.
        if git_hash and git_hash == expected_hash:
            continue
        detail = (
            f"expected {expected_hash[:16]}..., disk_lf={disk_hash[:16]}..."
        )
        if git_hash:
            detail += f", git_lf={git_hash[:16]}..."
        errors.append(f"hash mismatch for {art_path_str}: {detail}")

    for stat in manifest.get("statistics", []):
        stat_path_str = (stat.get("path") or "").replace("\\", "/")
        if not stat_path_str:
            continue
        stat_path = root / stat_path_str
        expected_hash = stat.get("sha256")
        if not expected_hash:
            continue
        if not stat_path.exists():
            errors.append(f"statistics missing: {stat_path_str}")
            continue
        actual_hash = file_sha256(stat_path, lf_normalize=True)
        if actual_hash != expected_hash:
            errors.append(
                f"hash mismatch for {stat_path_str}: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )

    return errors
