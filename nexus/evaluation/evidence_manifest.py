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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
            arts.append({"path": str(p), "missing": True})
            continue
        arts.append(
            {
                "path": str(p.relative_to(root)) if p.is_relative_to(root) else str(p),
                "sha256": file_sha256(p),
                "bytes": p.stat().st_size,
            }
        )
    stats = []
    for p in statistics_paths or []:
        if p.exists():
            stats.append(
                {
                    "path": str(p.relative_to(root)) if p.is_relative_to(root) else str(p),
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
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["manifest_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return manifest


def write_evidence_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
