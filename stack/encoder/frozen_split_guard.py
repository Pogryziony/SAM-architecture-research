"""Frozen split guard — prevents reuse of consumed evaluation splits.

The consumed frozen test split is permanently rejected by raw LF hash,
raw CRLF hash, and canonical semantic hash.  Future evaluations must
use a new, unlabeled holdout split.

All consumed-hash constants are derived from the committed repository
bytes.  This module must NOT read test.jsonl at import time to
determine hashes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from stack.encoder.semantic_hash import compute_canonical_semantic_sha256


# ── Consumed split registry ──────────────────────────────────────────────

@dataclass(frozen=True)
class ConsumedSplit:
    """Immutable record of a consumed evaluation split."""
    name: str
    raw_lf_sha256: str       # committed LF bytes
    raw_crlf_sha256: str     # local CRLF bytes (from original experiment)
    semantic_sha256: str     # canonical JSONL digest
    consumed_utc: str
    experiment_run_id: str
    status: str = "consumed"

    @property
    def all_hashes(self) -> set[str]:
        return {self.raw_lf_sha256, self.raw_crlf_sha256, self.semantic_sha256}


# Raw LF hash (committed bytes):
#   git show HEAD:stack/encoder/data/test.jsonl | sha256sum
CONSUMED_FROZEN_RAW_LF = (
    "b413a792d96b54b3913faea5ea999ee1f21821e00db795f7810113c6fc1bab71"
)

# Raw CRLF hash (local Windows workspace during original experiment):
CONSUMED_FROZEN_RAW_CRLF = (
    "ac7877084f2384d2e80ef3ce43d48c842eb4d404936d3139a1c7b06d41616c6a"
)

# Canonical semantic hash (JSONL content, line-ending insensitive):
CONSUMED_FROZEN_SEMANTIC = (
    "37c6fe6e83972de703efdc09fde1c0c19363ac0fab1f5e07c67e06ad98071556"
)

CONSUMED_FROZEN = ConsumedSplit(
    name="stack/encoder/data/test.jsonl",
    raw_lf_sha256=CONSUMED_FROZEN_RAW_LF,
    raw_crlf_sha256=CONSUMED_FROZEN_RAW_CRLF,
    semantic_sha256=CONSUMED_FROZEN_SEMANTIC,
    consumed_utc="2026-07-11T08:45:18Z",
    experiment_run_id="entity_ranker_v3_frozen_20260711T084518Z",
)

# ── Validation split (must never be used as frozen holdout) ──────────────

VALIDATION_RAW_LF = (
    "030005a1306d6eb2e57219967ff84e09df9927d018854dea3af948917ae0fdd5"
)
VALIDATION_RAW_CRLF = (
    "f95e212502c7c5ad5a615a3e1921e62ef7e1e961a229f44be63e3f829fdacd09"
)
VALIDATION_SEMANTIC = (
    "82f859e5c9c9d4aba0e2bdc9f382d13c57c792125a7c4f00798d0b4eff6697c2"
)

# ── Exported constants for backward compatibility ────────────────────────

CONSUMED_FROZEN_SHA256 = CONSUMED_FROZEN_RAW_LF
VALIDATION_SPLIT_SHA256 = VALIDATION_RAW_LF


# ── All consumed hashes (raw + semantic) ─────────────────────────────────

def _consumed_hash_set() -> set[str]:
    """All hashes that must be rejected."""
    return CONSUMED_FROZEN.all_hashes | {
        VALIDATION_RAW_LF, VALIDATION_RAW_CRLF, VALIDATION_SEMANTIC,
    }


# ── Public API ───────────────────────────────────────────────────────────

class ConsumedSplitError(ValueError):
    """Raised when an attempt is made to reuse a consumed split."""


def check_split_not_consumed(
    split_path: Path,
    consumed_hashes: Optional[set[str]] = None,
) -> tuple[str, str]:
    """Verify that a split has not been consumed.

    Reads *split_path* once, computes both raw and semantic SHA-256,
    and rejects it if either matches a consumed hash.

    Returns:
        (raw_sha256, semantic_sha256)
    """
    if consumed_hashes is None:
        consumed_hashes = _consumed_hash_set()

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    data = split_path.read_bytes()
    raw_hash = hashlib.sha256(data).hexdigest()
    semantic_hash = compute_canonical_semantic_sha256(data)

    for label, h in [("raw", raw_hash), ("semantic", semantic_hash)]:
        if h in consumed_hashes:
            raise ConsumedSplitError(
                f"Split {split_path} has {label} SHA-256 {h} which matches "
                f"a consumed split. This split must never be reused for "
                f"evaluation. Use a new, unlabeled holdout split instead."
            )

    return raw_hash, semantic_hash


def validate_new_holdout(
    split_path: Path,
    consumed_hashes: Optional[set[str]] = None,
) -> tuple[str, str]:
    """Validate that *split_path* is a new, unconsumed holdout.

    Rejects splits whose raw or semantic SHA-256 matches any consumed
    split, including the validation split.

    Returns:
        (raw_sha256, semantic_sha256)
    """
    if consumed_hashes is None:
        consumed_hashes = _consumed_hash_set()

    raw_hash, semantic_hash = check_split_not_consumed(split_path, consumed_hashes)
    return raw_hash, semantic_hash
