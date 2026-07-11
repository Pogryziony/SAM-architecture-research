"""Tests for canonical semantic JSONL hashing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stack.encoder.semantic_hash import (
    canonicalize_jsonl_bytes,
    compute_raw_sha256,
    compute_canonical_semantic_sha256_path,
    compute_canonical_semantic_sha256,
    canonicalize_and_hash_records,
)


def _crlf(data: bytes) -> bytes:
    return data.replace(b"\n", b"\r\n")


def test_lf_and_crlf_different_raw_same_semantic(tmp_path: Path):
    """LF and CRLF have different raw hashes but identical semantic hashes."""
    records = [{"id": "q1", "question": "What?", "entities": ["A"]}]
    lf = ("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n").encode("utf-8")
    crlf = _crlf(lf)

    f_lf = tmp_path / "lf.jsonl"
    f_crlf = tmp_path / "crlf.jsonl"
    f_lf.write_bytes(lf)
    f_crlf.write_bytes(crlf)

    assert compute_raw_sha256(f_lf) != compute_raw_sha256(f_crlf)
    assert compute_canonical_semantic_sha256_path(f_lf) == compute_canonical_semantic_sha256_path(f_crlf)


def test_insignificant_json_whitespace_same_semantic(tmp_path: Path):
    """Insignificant JSON whitespace within a line does not change the semantic hash."""
    # Both are valid JSONL (one JSON record per line); extra spaces inside JSON
    compact = b'{"id":"q1","question":"Q?"}\n'
    spaced = b'{  "id"  :  "q1"  ,  "question"  :  "Q?"  }\n'

    f1 = tmp_path / "compact.jsonl"
    f2 = tmp_path / "spaced.jsonl"
    f1.write_bytes(compact)
    f2.write_bytes(spaced)

    assert compute_canonical_semantic_sha256_path(f1) == compute_canonical_semantic_sha256_path(f2)


def test_changed_question_changes_semantic(tmp_path: Path):
    """Different questions produce different semantic hashes."""
    r1 = [{"id": "q1", "question": "Q1?", "entities": ["A"]}]
    r2 = [{"id": "q1", "question": "Q2?", "entities": ["A"]}]

    f1 = tmp_path / "q1.jsonl"
    f2 = tmp_path / "q2.jsonl"
    f1.write_text("\n".join(json.dumps(r, sort_keys=True) for r in r1) + "\n", encoding="utf-8")
    f2.write_text("\n".join(json.dumps(r, sort_keys=True) for r in r2) + "\n", encoding="utf-8")

    assert compute_canonical_semantic_sha256_path(f1) != compute_canonical_semantic_sha256_path(f2)


def test_changed_gold_changes_semantic(tmp_path: Path):
    """Different gold labels produce different semantic hashes."""
    r1 = [{"id": "q1", "question": "Q?", "entities": ["A"]}]
    r2 = [{"id": "q1", "question": "Q?", "entities": ["B"]}]

    f1 = tmp_path / "g1.jsonl"
    f2 = tmp_path / "g2.jsonl"
    f1.write_text("\n".join(json.dumps(r, sort_keys=True) for r in r1) + "\n", encoding="utf-8")
    f2.write_text("\n".join(json.dumps(r, sort_keys=True) for r in r2) + "\n", encoding="utf-8")

    assert compute_canonical_semantic_sha256_path(f1) != compute_canonical_semantic_sha256_path(f2)


def test_reordered_json_keys_same_semantic(tmp_path: Path):
    """Reordered JSON keys do not change the semantic hash."""
    # Use Python's insertion-order dicts to create different key orders
    r1_str = '{"question":"Q?","id":"q1","entities":["A"]}\n'
    # Different key order in JSON string
    r2_str = '{"entities":["A"],"id":"q1","question":"Q?"}\n'

    f1 = tmp_path / "order1.jsonl"
    f2 = tmp_path / "order2.jsonl"
    f1.write_text(r1_str, encoding="utf-8")
    f2.write_text(r2_str, encoding="utf-8")

    assert compute_canonical_semantic_sha256_path(f1) == compute_canonical_semantic_sha256_path(f2)


def test_reordered_records_change_semantic(tmp_path: Path):
    """Reordered JSONL records DO change the semantic hash."""
    records = [
        {"id": "q1", "question": "Q1?", "entities": ["A"]},
        {"id": "q2", "question": "Q2?", "entities": ["B"]},
    ]
    records_rev = list(reversed(records))

    f1 = tmp_path / "order.jsonl"
    f2 = tmp_path / "order_rev.jsonl"
    f1.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8"
    )
    f2.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records_rev) + "\n", encoding="utf-8"
    )

    assert compute_canonical_semantic_sha256_path(f1) != compute_canonical_semantic_sha256_path(f2)


def test_canonicalize_and_hash_records_uses_sorted_keys():
    """Records with different key orders produce the same hash."""
    r1 = [{"b": 2, "a": 1}]
    r2 = [{"a": 1, "b": 2}]
    assert canonicalize_and_hash_records(r1) == canonicalize_and_hash_records(r2)


def test_non_jsonl_content_raises_error(tmp_path: Path):
    """Non-JSONL content raises MalformedJSONLError, not silently hashed."""
    from stack.encoder.semantic_hash import MalformedJSONLError

    f = tmp_path / "not_jsonl.txt"
    f.write_text("this is not json\nnor is this\n", encoding="utf-8")
    with pytest.raises(MalformedJSONLError):
        compute_canonical_semantic_sha256_path(f)


def test_malformed_json_raises_error(tmp_path: Path):
    """Malformed JSON raises MalformedJSONLError."""
    from stack.encoder.semantic_hash import MalformedJSONLError

    f = tmp_path / "bad.jsonl"
    f.write_text('{"valid": 1}\n{broken json\n', encoding="utf-8")
    with pytest.raises(MalformedJSONLError):
        compute_canonical_semantic_sha256_path(f)


def test_truncated_line_raises_error(tmp_path: Path):
    """Truncated JSON line raises MalformedJSONLError."""
    from stack.encoder.semantic_hash import MalformedJSONLError

    f = tmp_path / "truncated.jsonl"
    f.write_text('{"valid": 1}\n{"incomplete":\n', encoding="utf-8")
    with pytest.raises(MalformedJSONLError):
        compute_canonical_semantic_sha256_path(f)


def test_empty_split_raises_error(tmp_path: Path):
    """Empty split raises EmptySplitError."""
    from stack.encoder.semantic_hash import EmptySplitError

    f = tmp_path / "empty.jsonl"
    f.write_text("\n\n", encoding="utf-8")
    with pytest.raises(EmptySplitError):
        compute_canonical_semantic_sha256_path(f)


def test_invalid_utf8_raises_error(tmp_path: Path):
    """Invalid UTF-8 raises UnicodeDecodeError."""
    f = tmp_path / "bad_utf8.jsonl"
    f.write_bytes(b'\x80\x81invalid utf-8\n')
    with pytest.raises(UnicodeDecodeError):
        compute_canonical_semantic_sha256_path(f)


def test_malformed_error_has_line_number(tmp_path: Path):
    """MalformedJSONLError includes the line number."""
    from stack.encoder.semantic_hash import MalformedJSONLError

    f = tmp_path / "numbered.jsonl"
    f.write_text('{"valid": 1}\n\n{"valid": 2}\n{broken\n', encoding="utf-8")
    try:
        compute_canonical_semantic_sha256_path(f)
    except MalformedJSONLError as exc:
        assert exc.line_number == 4  # line 3 is blank, line 4 is broken
        assert "broken" in exc.raw_line
