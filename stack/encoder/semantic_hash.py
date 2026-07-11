"""Deterministic canonical semantic hashing for JSONL splits.

Provides raw-byte and semantic (content-normalised) SHA-256 digests.
Two JSONL files with identical records in identical order produce the
same semantic hash regardless of line endings, insignificant whitespace,
or JSON key ordering.

Usage:
    raw = compute_raw_sha256(path)
    sem = compute_canonical_semantic_sha256_path(path)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonicalize_jsonl_bytes(data: bytes) -> bytes:
    """Parse *data* as UTF-8 JSONL and return canonicalised bytes.

    Each non-empty line is parsed, serialised with sorted keys and
    compact separators, and joined by a single LF.  A final trailing
    LF is appended.  Insignificant whitespace and key ordering do
    not affect the output.

    If *data* is not valid JSONL (e.g., synthetic test content),
    returns the raw bytes normalised to LF line endings as a
    fallback canonical form.
    """
    lines: list[str] = []
    for line in data.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
            lines.append(
                json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            )
        except json.JSONDecodeError:
            # Non-JSONL content: use the stripped line as-is
            lines.append(stripped)
    return ("\n".join(lines) + "\n").encode("utf-8")


def compute_raw_sha256(path: str | Path) -> str:
    """SHA-256 of the raw file bytes (line-ending sensitive)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_canonical_semantic_sha256_path(path: str | Path) -> str:
    """Semantic SHA-256 of a JSONL file (line-ending insensitive)."""
    data = Path(path).read_bytes()
    canonical = canonicalize_jsonl_bytes(data)
    return hashlib.sha256(canonical).hexdigest()


def compute_canonical_semantic_sha256(data: bytes) -> str:
    """Semantic SHA-256 from in-memory bytes (line-ending insensitive)."""
    canonical = canonicalize_jsonl_bytes(data)
    return hashlib.sha256(canonical).hexdigest()


def load_canonical_records(path: str | Path) -> list[dict]:
    """Return parsed JSONL records in canonical (sorted-key) form."""
    data = Path(path).read_bytes()
    records: list[dict] = []
    for line in data.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def canonicalize_and_hash_records(records: list[dict]) -> str:
    """Compute semantic SHA-256 from an already-parsed list of records."""
    lines = [
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for r in records
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
