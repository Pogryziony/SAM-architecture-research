"""Deterministic canonical semantic hashing for JSONL splits.

Provides raw-byte and semantic (content-normalised) SHA-256 digests.
Two JSONL files with identical records in identical order produce the
same semantic hash regardless of line endings, insignificant whitespace,
or JSON key ordering.

Malformed JSONL lines raise MalformedJSONLError — no silent fallback.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class MalformedJSONLError(ValueError):
    """Raised when a JSONL line cannot be parsed as valid JSON."""

    def __init__(self, line_number: int, raw_line: str, original_error: str):
        self.line_number = line_number
        self.raw_line = raw_line
        self.original_error = original_error
        super().__init__(
            f"Line {line_number}: invalid JSON — {original_error}"
        )


class EmptySplitError(ValueError):
    """Raised when a JSONL split contains zero records."""


@dataclass(frozen=True)
class HoldoutData:
    """Single-read holdout split with pre-computed hashes and parsed records."""
    raw_bytes: bytes
    records: list[dict]
    raw_sha256: str
    semantic_sha256: str

    @property
    def record_count(self) -> int:
        return len(self.records)


def canonicalize_jsonl_bytes(data: bytes) -> bytes:
    """Parse *data* as UTF-8 JSONL and return canonicalised bytes.

    Each non-empty line is parsed, serialised with sorted keys and
    compact separators, and joined by a single LF.  A final trailing
    LF is appended.

    Raises MalformedJSONLError for any line that is not valid JSON.
    Raises EmptySplitError if the split contains zero records.
    """
    lines: list[str] = []
    for idx, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise MalformedJSONLError(
                line_number=idx,
                raw_line=stripped[:200],
                original_error=str(exc),
            )
        lines.append(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
    if not lines:
        raise EmptySplitError("JSONL split contains zero non-empty records")
    return ("\n".join(lines) + "\n").encode("utf-8")


def compute_raw_sha256(path: str | Path) -> str:
    """SHA-256 of the raw file bytes (line-ending sensitive)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_canonical_semantic_sha256_path(path: str | Path) -> str:
    """Semantic SHA-256 of a JSONL file (line-ending insensitive)."""
    data = Path(path).read_bytes()
    return compute_canonical_semantic_sha256(data)


def compute_canonical_semantic_sha256(data: bytes) -> str:
    """Semantic SHA-256 from in-memory bytes (line-ending insensitive)."""
    canonical = canonicalize_jsonl_bytes(data)
    return hashlib.sha256(canonical).hexdigest()


def load_and_validate_new_holdout(
    split_path: str | Path,
    consumed_hashes: set[str] | None = None,
) -> HoldoutData:
    """Read a holdout split exactly once and validate it is unconsumed.

    Opens *split_path* once, reads all bytes, computes raw and semantic
    hashes, parses records, and rejects the split if any hash matches
    *consumed_hashes*.

    Returns HoldoutData with raw bytes, parsed records, and both hashes.
    The caller must use the returned records — do not read the file again.
    """
    from stack.encoder.frozen_split_guard import _consumed_hash_set, ConsumedSplitError

    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    # ── Single physical read ──
    raw_bytes = path.read_bytes()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Parse and canonicalize from the same bytes
    canonical = canonicalize_jsonl_bytes(raw_bytes)
    semantic_hash = hashlib.sha256(canonical).hexdigest()

    # Parse records
    records = load_canonical_records_from_bytes(raw_bytes)

    # Check consumed
    if consumed_hashes is None:
        consumed_hashes = _consumed_hash_set()

    for label, h in [("raw", raw_hash), ("semantic", semantic_hash)]:
        if h in consumed_hashes:
            raise ConsumedSplitError(
                f"Split {path} has {label} SHA-256 {h} which matches "
                f"a consumed split."
            )

    return HoldoutData(
        raw_bytes=raw_bytes,
        records=records,
        raw_sha256=raw_hash,
        semantic_sha256=semantic_hash,
    )


def load_canonical_records(path: str | Path) -> list[dict]:
    """Return parsed JSONL records in canonical (sorted-key) form."""
    return load_canonical_records_from_bytes(Path(path).read_bytes())


def load_canonical_records_from_bytes(data: bytes) -> list[dict]:
    """Return parsed JSONL records from in-memory bytes."""
    records: list[dict] = []
    for idx, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise MalformedJSONLError(
                line_number=idx,
                raw_line=stripped[:200],
                original_error=str(exc),
            )
    return records


def canonicalize_and_hash_records(records: list[dict]) -> str:
    """Compute semantic SHA-256 from an already-parsed list of records."""
    lines = [
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for r in records
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
