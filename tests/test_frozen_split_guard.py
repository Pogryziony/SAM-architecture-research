"""Tests for the frozen split guard module.

T13: Consumed-split rejection (raw LF, raw CRLF, semantic)
T14: New-holdout validation
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from stack.encoder.frozen_split_guard import (
    CONSUMED_FROZEN_SHA256,
    CONSUMED_FROZEN_RAW_LF,
    CONSUMED_FROZEN_RAW_CRLF,
    CONSUMED_FROZEN_SEMANTIC,
    VALIDATION_SPLIT_SHA256,
    VALIDATION_RAW_LF,
    VALIDATION_RAW_CRLF,
    VALIDATION_SEMANTIC,
    ConsumedSplitError,
    check_split_not_consumed,
    validate_new_holdout,
)


def _git_show(path: str) -> bytes:
    """Return committed bytes for *path* at HEAD."""
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"], capture_output=True, check=True
    ).stdout


# ── CRLF helper ──

def _crlf(data: bytes) -> bytes:
    return data.replace(b"\n", b"\r\n")


# ── T13: Consumed split rejection ──


def test_consumed_frozen_hash_rejected(tmp_path: Path):
    """Any file with a consumed raw hash is rejected."""
    f = tmp_path / "test.jsonl"
    # Use valid JSONL content
    f.write_text('{"id":"q1","question":"Q?","entities":["A"]}\n', encoding="utf-8")
    fh = hashlib.sha256(f.read_bytes()).hexdigest()
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        check_split_not_consumed(f, {fh})


def test_consumed_frozen_lf_rejected(tmp_path: Path):
    """LF copy of the consumed frozen split is rejected."""
    data = _git_show("stack/encoder/data/test.jsonl")
    f = tmp_path / "lf_test.jsonl"
    f.write_bytes(data)
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        validate_new_holdout(f)


def test_consumed_frozen_crlf_rejected(tmp_path: Path):
    """CRLF copy of the consumed frozen split is rejected."""
    data = _git_show("stack/encoder/data/test.jsonl")
    f = tmp_path / "crlf_test.jsonl"
    f.write_bytes(_crlf(data))
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        validate_new_holdout(f)


def test_validation_split_rejected_as_holdout(tmp_path: Path):
    """The validation split is rejected as a frozen holdout."""
    data = _git_show("stack/encoder/data/val.jsonl")
    f = tmp_path / "val.jsonl"
    f.write_bytes(data)
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        validate_new_holdout(f)


def test_validation_crlf_rejected_as_holdout(tmp_path: Path):
    """CRLF copy of validation is also rejected."""
    data = _git_show("stack/encoder/data/val.jsonl")
    f = tmp_path / "val_crlf.jsonl"
    f.write_bytes(_crlf(data))
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        validate_new_holdout(f)


def test_renamed_consumed_split_rejected(tmp_path: Path):
    """Renaming doesn't bypass the guard."""
    data = _git_show("stack/encoder/data/test.jsonl")
    f = tmp_path / "renamed_holdout.jsonl"
    f.write_bytes(data)
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        validate_new_holdout(f)


def test_copied_consumed_split_rejected(tmp_path: Path):
    """Copying the file doesn't bypass the guard."""
    data = _git_show("stack/encoder/data/test.jsonl")
    f = tmp_path / "copy.jsonl"
    f.write_bytes(data)
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        validate_new_holdout(f)


def test_whitespace_reformatted_rejected(tmp_path: Path):
    """Whitespace-reformatted consumed split is rejected via semantic hash."""
    data = _git_show("stack/encoder/data/test.jsonl")
    # Add extra whitespace around JSON — should still have same semantic hash
    modified = data.decode("utf-8").replace('{"id"', '  {"id"').encode("utf-8")
    f = tmp_path / "reformatted.jsonl"
    f.write_bytes(modified)
    with pytest.raises(ConsumedSplitError, match="consumed split"):
        validate_new_holdout(f)


def test_genuinely_new_synthetic_split_accepted(tmp_path: Path):
    """A genuinely new, synthetic holdout passes validation."""
    records = [
        {"id": f"new_q_{i}", "question": f"Question {i}?", "entities": ["E1", "E2"]}
        for i in range(10)
    ]
    f = tmp_path / "new_holdout.jsonl"
    f.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )
    raw_hash, semantic_hash = validate_new_holdout(f)
    assert len(raw_hash) == 64
    assert len(semantic_hash) == 64


def test_changed_question_not_treated_as_identical(tmp_path: Path):
    """A modified question must NOT be treated as the consumed split."""
    records = [
        {"id": "new_q_0", "question": "Completely different question?", "entities": ["X"]}
        for _ in range(10)
    ]
    f = tmp_path / "different.jsonl"
    f.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )
    raw_hash, semantic_hash = validate_new_holdout(f)
    # Should not raise — genuinely different content
    assert raw_hash not in {
        CONSUMED_FROZEN_RAW_LF, CONSUMED_FROZEN_RAW_CRLF, VALIDATION_RAW_LF, VALIDATION_RAW_CRLF
    }


def test_missing_file_raises():
    """A nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        validate_new_holdout(Path("/nonexistent/path/12345.jsonl"))


def test_no_silent_override():
    """There must be no force/override flag."""
    import inspect
    sig = inspect.signature(validate_new_holdout)
    params = list(sig.parameters.keys())
    assert "force" not in params
    assert "override" not in params
    assert "allow_consumed" not in params


def test_return_type_is_tuple():
    """validate_new_holdout returns (raw_hash, semantic_hash)."""
    import json
    records = [{"id": "q1", "question": "Q?", "entities": ["E"]}]
    f = Path("_tmp_new_holdout_test.jsonl")
    f.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )
    try:
        result = validate_new_holdout(f)
        assert isinstance(result, tuple)
        assert len(result) == 2
        raw_hash, semantic_hash = result
        assert len(raw_hash) == 64
        assert len(semantic_hash) == 64
    finally:
        f.unlink(missing_ok=True)


# ── Hash constant verification against committed bytes ──

def test_guard_constants_match_committed_bytes():
    """Verify frozen split guard constants match committed repository bytes."""
    committed_test = _git_show("stack/encoder/data/test.jsonl")
    committed_val = _git_show("stack/encoder/data/val.jsonl")

    from stack.encoder.semantic_hash import compute_canonical_semantic_sha256

    # Raw LF
    assert hashlib.sha256(committed_test).hexdigest() == CONSUMED_FROZEN_RAW_LF
    assert hashlib.sha256(committed_val).hexdigest() == VALIDATION_RAW_LF

    # Raw CRLF
    assert hashlib.sha256(_crlf(committed_test)).hexdigest() == CONSUMED_FROZEN_RAW_CRLF
    assert hashlib.sha256(_crlf(committed_val)).hexdigest() == VALIDATION_RAW_CRLF

    # Semantic
    assert compute_canonical_semantic_sha256(committed_test) == CONSUMED_FROZEN_SEMANTIC
    assert compute_canonical_semantic_sha256(committed_val) == VALIDATION_SEMANTIC
