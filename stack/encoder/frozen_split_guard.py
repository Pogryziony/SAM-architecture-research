"""Frozen split guard — prevents reuse of consumed evaluation splits.

The consumed frozen test split is permanently rejected.  Future
evaluations must use a new, unlabeled holdout split.

This module must NOT read test.jsonl to determine the consumed hash.
The hash is embedded as a constant derived from the original split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

# SHA-256 of stack/encoder/data/test.jsonl as consumed by the
# Entity Ranker V3 frozen evaluation on 2026-07-11.
# This split must never be reused for development, tuning,
# model selection, or reporting.
CONSUMED_FROZEN_SHA256 = (
    "ac7877084f2384d2e80ef3ce43d48c842eb4d404936d3139a1c7b06d41616c6a"
)

# SHA-256 of stack/encoder/data/val.jsonl used for validation.
# This split may be used for validation only.  It must never be
# used as a frozen holdout.
VALIDATION_SPLIT_SHA256 = (
    "f95e212502c7c5ad5a615a3e1921e62ef7e1e961a229f44be63e3f829fdacd09"
)


class ConsumedSplitError(ValueError):
    """Raised when an attempt is made to reuse a consumed split."""


def check_split_not_consumed(
    split_path: Path,
    consumed_hashes: Optional[set[str]] = None,
) -> str:
    """Verify that a split has not been consumed.

    Reads *split_path* once, computes its SHA-256, and rejects it
    if the hash matches any entry in *consumed_hashes*.

    Args:
        split_path: Path to the split file.
        consumed_hashes: SHA-256 hashes of consumed splits.
            Defaults to {CONSUMED_FROZEN_SHA256, VALIDATION_SPLIT_SHA256}
            when evaluating a test-like split.

    Returns:
        The SHA-256 hex digest of *split_path*.

    Raises:
        ConsumedSplitError: If the split hash matches a consumed hash.
        FileNotFoundError: If *split_path* does not exist.
    """
    if consumed_hashes is None:
        consumed_hashes = {CONSUMED_FROZEN_SHA256, VALIDATION_SPLIT_SHA256}

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    data = split_path.read_bytes()
    file_hash = hashlib.sha256(data).hexdigest()

    if file_hash in consumed_hashes:
        raise ConsumedSplitError(
            f"Split {split_path} has SHA-256 {file_hash} which matches "
            f"a consumed split. This split must never be reused for "
            f"evaluation. Use a new, unlabeled holdout split instead."
        )

    return file_hash


def validate_new_holdout(
    split_path: Path,
    consumed_hashes: Optional[set[str]] = None,
) -> str:
    """Validate that *split_path* is a new, unconsumed holdout.

    Only permits splits whose SHA-256 does NOT match any consumed hash.
    This is the function to call before a new frozen evaluation.

    Args:
        split_path: Path to the new holdout split.
        consumed_hashes: Set of consumed hashes. Defaults to
            {CONSUMED_FROZEN_SHA256}.

    Returns:
        The SHA-256 hex digest of *split_path*.

    Raises:
        ConsumedSplitError: If the hash matches a consumed split.
        FileNotFoundError: If *split_path* does not exist.
    """
    if consumed_hashes is None:
        consumed_hashes = {CONSUMED_FROZEN_SHA256}

    file_hash = check_split_not_consumed(split_path, consumed_hashes)

    # Also reject if it matches the validation split
    if file_hash == VALIDATION_SPLIT_SHA256:
        raise ConsumedSplitError(
            f"Split {split_path} matches the validation split. "
            f"The validation split must not be used as a frozen holdout."
        )

    return file_hash
