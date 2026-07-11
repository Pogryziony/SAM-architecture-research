"""Tests for the frozen split guard module.

T13: Consumed-split rejection
T14: New-holdout validation
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stack.encoder.frozen_split_guard import (
    CONSUMED_FROZEN_SHA256,
    VALIDATION_SPLIT_SHA256,
    ConsumedSplitError,
    check_split_not_consumed,
    validate_new_holdout,
)


# ── T13: Consumed split rejection ──


def test_consumed_frozen_hash_rejected(tmp_path: Path):
    """The consumed frozen split hash is permanently rejected."""
    # Create a synthetic file whose hash matches the consumed value
    # We can't create a file with a specific hash, so we test by adding
    # a known hash to the consumed set
    f = tmp_path / "test.jsonl"
    f.write_text("synthetic test split content\n")
    file_hash = hashlib.sha256(f.read_bytes()).hexdigest()

    # Add this hash to consumed set + verify rejection
    consumed = {CONSUMED_FROZEN_SHA256, file_hash}
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        check_split_not_consumed(f, consumed)


def test_consumed_frozen_hash_constant_matches_actual():
    """Verify the embedded CONSUMED_FROZEN_SHA256 constant is the actual
    test.jsonl hash.  This test reads the file to confirm the constant."""
    from pathlib import Path
    test_path = Path("stack/encoder/data/test.jsonl")
    actual = hashlib.sha256(test_path.read_bytes()).hexdigest()
    assert actual == CONSUMED_FROZEN_SHA256, (
        f"CONSUMED_FROZEN_SHA256 ({CONSUMED_FROZEN_SHA256[:16]}...)"
        f" does not match actual test.jsonl hash ({actual[:16]}...)."
        f" Update the constant if the split was legitimately replaced."
    )


def test_validation_split_constant_matches_actual():
    """Verify the embedded VALIDATION_SPLIT_SHA256 constant matches."""
    from pathlib import Path
    val_path = Path("stack/encoder/data/val.jsonl")
    actual = hashlib.sha256(val_path.read_bytes()).hexdigest()
    assert actual == VALIDATION_SPLIT_SHA256, (
        f"VALIDATION_SPLIT_SHA256 ({VALIDATION_SPLIT_SHA256[:16]}...)"
        f" does not match actual val.jsonl hash ({actual[:16]}...)."
    )


def test_new_synthetic_split_accepted(tmp_path: Path):
    """A synthetic temporary split with an unknown hash is accepted."""
    f = tmp_path / "new_holdout.jsonl"
    f.write_text("new holdout question\n" * 10)
    file_hash = validate_new_holdout(f)
    assert len(file_hash) == 64
    assert file_hash == hashlib.sha256(f.read_bytes()).hexdigest()


def test_reject_split_matching_consumed_hash(tmp_path: Path):
    """A split whose hash matches the consumed value is rejected."""
    # Read the actual test.jsonl content and copy it to tmp_path
    test_content = Path("stack/encoder/data/test.jsonl").read_bytes()
    f = tmp_path / "copy_of_test.jsonl"
    f.write_bytes(test_content)
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        validate_new_holdout(f)


def test_reject_validation_split_as_holdout(tmp_path: Path):
    """The validation split must not be used as a frozen holdout."""
    val_content = Path("stack/encoder/data/val.jsonl").read_bytes()
    f = tmp_path / "copy_of_val.jsonl"
    f.write_bytes(val_content)
    with pytest.raises(ConsumedSplitError, match="validation split"):
        validate_new_holdout(f)


def test_missing_file_raises():
    """A nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        validate_new_holdout(Path("/nonexistent/path/12345.jsonl"))


def test_check_split_with_custom_consumed_set(tmp_path: Path):
    """Custom consumed hashes work with check_split_not_consumed."""
    f = tmp_path / "custom.jsonl"
    f.write_text("custom content")
    fh = hashlib.sha256(f.read_bytes()).hexdigest()

    # Accept with default consumed set
    result = check_split_not_consumed(f)
    assert result == fh

    # Reject with custom set containing this hash
    with pytest.raises(ConsumedSplitError):
        check_split_not_consumed(f, {fh})


# ── T14: Additional guard tests ──


def test_no_silent_override_for_consumed_hash():
    """There must be no flag that silently permits the consumed hash."""
    # The function has no 'force' or 'override' parameter
    import inspect
    sig = inspect.signature(validate_new_holdout)
    params = list(sig.parameters.keys())
    assert "force" not in params
    assert "override" not in params
    assert "allow_consumed" not in params
