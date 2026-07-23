"""Canonical dataset identity over every behavior/scoring field.

Question-only hashes are forbidden for primary evaluation artifacts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

# Fields that affect scoring, adjudication, or protocol identity.
CANONICAL_RECORD_FIELDS: tuple[str, ...] = (
    "id",
    "question",
    "gold_answer",
    "gold_entities",
    "gold_path",
    "should_abstain",
    "category",
    "question_type",
    "path_required",
    "source_split",
    "domain",
    "as_known_at",
    "as_valid_at",
    "structured_gold",
    "gold_structured",
    "relevant_doc_ids",
    "relevant_chunk_ids",
    "rubric",
)


def canonical_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a stable, sorted view of scoring-relevant fields."""
    out: dict[str, Any] = {}
    for key in CANONICAL_RECORD_FIELDS:
        if key not in record:
            continue
        value = record[key]
        if value is None:
            continue
        out[key] = value
    # Require an id
    if "id" not in out and "question_id" in record:
        out["id"] = record["question_id"]
    return out


def hash_canonical_record(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        canonical_record(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_dataset(records: Sequence[Mapping[str, Any]]) -> str:
    """Full-dataset SHA-256 over canonical records in id-sorted order."""
    ordered = sorted(
        (canonical_record(r) for r in records),
        key=lambda r: str(r.get("id") or ""),
    )
    if not ordered:
        raise ValueError("cannot hash empty dataset")
    # Fail closed: gold mutation must change identity
    blob = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def question_only_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Legacy Phase-4 question-only hash — forbidden for primary artifacts."""
    payload = json.dumps(
        [{"id": r.get("id"), "question": r.get("question")} for r in records],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_primary_dataset_hash(artifact: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    """Raise if artifact uses a question-only hash or mismatches canonical hash."""
    expected = hash_dataset(records)
    got = str(artifact.get("dataset_sha256") or "")
    if got != expected:
        legacy = question_only_hash(records)
        if got == legacy:
            raise ValueError(
                "artifact uses forbidden question-only dataset_sha256; "
                "regenerate with hash_dataset()"
            )
        raise ValueError(
            f"dataset_sha256 mismatch: got {got[:16]}… expected {expected[:16]}…"
        )
