"""Tests for the non-autoregressive copy/edit transducer and edit scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus.realizer.edit_script import (
    DELETE,
    KEEP,
    REPLACE,
    apply_edit_target,
    compute_edit_target,
    edit_accuracy,
    tokenize,
)
from nexus.realizer.subword_tokenizer import TrainOnlySubwordTokenizer


# ── Edit script computation ──────────────────────────────────────────


def test_identical_strings_produce_all_keep():
    labels = compute_edit_target("Warsaw", "Warsaw")
    assert labels == ["Warsaw"]
    assert apply_edit_target("Warsaw", labels) == "Warsaw"


def test_replacement_substitutes_correct_token():
    labels = compute_edit_target("Warsaw", "Warszawa")
    assert labels == ["Warszawa"]
    result = apply_edit_target("Warsaw", labels)
    assert "Warszawa" in result


def test_deletion_removes_token():
    labels = compute_edit_target("the Warsaw", "Warsaw")
    # "the " is two tokens (word + space); both should be deleted.
    assert labels[0] == "[DELETE]"
    assert "Warsaw" in labels
    result = apply_edit_target("the Warsaw", labels)
    assert "the" not in result or result.strip() == "Warsaw"


def test_polish_inflection():
    labels = compute_edit_target("Warszawa", "Warszawy")
    result = apply_edit_target("Warszawa", labels)
    assert result == "Warszawy"


def test_numeric_preservation():
    labels = compute_edit_target("42", "42")
    assert labels == ["42"]
    assert apply_edit_target("42", labels) == "42"


def test_single_token_canonical_with_multi_token_target():
    labels = compute_edit_target("42", "the answer is 42")
    # Inserted tokens merge into the single canonical position.
    assert len(labels) == len(tokenize("42"))
    result = apply_edit_target("42", labels)
    # At minimum the canonical fact must be preserved.
    assert "42" in result


def test_edit_target_length_matches_canonical():
    for canonical, _ in [
        ("Warsaw", "Warszawa"),
        ("the quick brown fox", "the slow brown fox"),
        ("42", "the answer is 42"),
        ("", "anything"),
    ]:
        canonical_tokens = tokenize(canonical)
        if not canonical_tokens:
            assert compute_edit_target(canonical, _) == []
        else:
            labels = compute_edit_target(canonical, _)
            assert len(labels) == len(canonical_tokens), (
                f"length mismatch for {canonical!r}: "
                f"{len(labels)} != {len(canonical_tokens)}"
            )


def test_edit_accuracy_exact_match():
    labels = ["A", "B", "C"]
    assert edit_accuracy(labels, labels) == {
        "exact_match": 1.0, "position_accuracy": 1.0, "total": 3, "correct": 3,
    }


def test_edit_accuracy_partial():
    labels = ["A", "B", "C"]
    predicted = ["A", "X", "C"]
    result = edit_accuracy(labels, predicted)
    assert result["exact_match"] == 0.0
    assert result["position_accuracy"] == 2 / 3
    assert result["correct"] == 2


def test_edit_accuracy_length_mismatch():
    assert edit_accuracy(["A"], ["A", "B"])["exact_match"] == 0.0


# ── Tokenizer decode_token ───────────────────────────────────────────


def test_decode_token_roundtrips_known_piece():
    tokenizer = TrainOnlySubwordTokenizer.train(
        ["Warsaw Warsaw Warsaw is the answer. Warszawa to odpowiedź."],
        max_pieces=16,
    )
    encoded = tokenizer.encode("Warsaw", add_special_tokens=False)
    assert encoded
    decoded = tokenizer.decode_token(encoded[0])
    assert decoded == "Warsaw"


def test_decode_token_handles_byte_fallback():
    tokenizer = TrainOnlySubwordTokenizer([])  # no pieces — all bytes
    encoded = tokenizer.encode("Łódź", add_special_tokens=False)
    assert encoded
    for token_id in encoded:
        decoded = tokenizer.decode_token(token_id)
        # Each byte decodes to a UTF-8 fragment.
        assert isinstance(decoded, str)


def test_decode_token_rejects_unknown_id():
    tokenizer = TrainOnlySubwordTokenizer([])
    with pytest.raises(ValueError, match="unknown token id"):
        tokenizer.decode_token(99999)


# ── Transducer model (no PyTorch required) ───────────────────────────


def test_transducer_config_schema_constant():
    from nexus.realizer.copy_edit_transducer import CONFIG_SCHEMA

    assert CONFIG_SCHEMA == "nexus-copy-edit-transducer-config-v1"


def test_find_fact_positions():
    from nexus.realizer.copy_edit_transducer import find_fact_positions

    tokenizer = TrainOnlySubwordTokenizer.train(
        ["Warsaw is the capital."], max_pieces=16,
    )
    source_ids = tokenizer.encode("[FACT] Warsaw [/FACT] is the answer.")
    positions = find_fact_positions(source_ids, tokenizer, "Warsaw")
    assert len(positions) == len(tokenizer.encode("Warsaw", add_special_tokens=False))


def test_find_fact_positions_raises_when_missing():
    from nexus.realizer.copy_edit_transducer import find_fact_positions

    tokenizer = TrainOnlySubwordTokenizer([])
    source_ids = tokenizer.encode("No fact here.")
    with pytest.raises(ValueError, match="fact span not found"):
        find_fact_positions(source_ids, tokenizer, "Warsaw")


def test_build_label_ids():
    from nexus.realizer.copy_edit_transducer import build_label_ids, DELETE_ID

    tokenizer = TrainOnlySubwordTokenizer.train(
        ["Warsaw Warszawa Warsaw Warszawa"], max_pieces=16,
    )
    # build_label_ids maps label strings to token IDs.
    labels = build_label_ids(["Warsaw"], tokenizer)
    assert len(labels) == 1
    # The label should map to a valid token ID (not DELETE).
    assert labels[0] != DELETE_ID
    # Roundtrip: decode should recover the token.
    decoded = tokenizer.decode_token(labels[0])
    assert decoded == "Warsaw"


def test_build_label_ids_delete():
    from nexus.realizer.copy_edit_transducer import build_label_ids, DELETE_ID

    tokenizer = TrainOnlySubwordTokenizer([])
    labels = build_label_ids(["[DELETE]"], tokenizer)
    assert labels == [DELETE_ID]


def test_tokenize_for_transducer_uses_consistent_regex():
    from nexus.realizer.copy_edit_transducer import tokenize_for_transducer

    pieces = tokenize_for_transducer("Zażółć gęślą jaźń.")
    assert tokenize("Zażółć gęślą jaźń.") == pieces


# ── Transducer forward pass (requires PyTorch) ───────────────────────


@pytest.mark.torch
def test_transducer_forward_pass():
    torch = pytest.importorskip("torch")
    from nexus.realizer.copy_edit_transducer import build_copy_edit_transducer

    config = {
        "vocab_size": 512,
        "hidden_size": 32,
        "output_vocab_size": 512,
        "dropout": 0.0,
    }
    model = build_copy_edit_transducer(config)
    source = torch.tensor([[1, 7, 8, 2]], dtype=torch.long)
    logits = model(source)
    assert logits.shape == (1, 4, 512)  # (batch, seq, output_vocab)
    assert torch.isfinite(logits).all()


@pytest.mark.torch
def test_transducer_forward_with_fact_positions():
    torch = pytest.importorskip("torch")
    from nexus.realizer.copy_edit_transducer import build_copy_edit_transducer

    config = {
        "vocab_size": 512,
        "hidden_size": 32,
        "output_vocab_size": 512,
        "dropout": 0.0,
    }
    model = build_copy_edit_transducer(config)
    source = torch.tensor([[1, 7, 8, 2]], dtype=torch.long)
    fact_pos = torch.tensor([[1, 2]], dtype=torch.long)
    logits = model(source, fact_pos)
    assert logits.shape == (1, 2, 512)  # only fact positions


@pytest.mark.torch
def test_transducer_overfit_smoke():
    """Verify the transducer can overfit 4 identical records."""
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F
    from nexus.realizer.copy_edit_transducer import (
        build_copy_edit_transducer, find_fact_positions,
    )
    from nexus.realizer.edit_script import compute_edit_target, tokenize as edit_tokenize
    from nexus.realizer.plan_serializer import serialize_answer_plan_for_model
    from nexus.realizer.answer_plan import compile_answer_plan
    from benchmarks.build_realizer_corpus_v2 import _make_record
    from benchmarks.realizer_corpus_v2_contracts import text_fingerprint

    # Build a minimal fixture record.
    source_def = {
        "language": "en", "revision": "a" * 40,
        "license": "CC BY 4.0", "url": "https://example.test/data",
    }
    text = "The capital of Poland is Warsaw."
    record = _make_record(
        dataset="fixture", source=source_def, artifact_sha256="b" * 64,
        source_split="train", source_id="overfit-1",
        question="What is the capital of Poland?",
        answer="Warsaw", aliases=[], answerable=True,
        operator="extract", hops=1,
        evidence=[{
            "id": "fixture:overfit-1:e0", "title": "Fixture",
            "text": text, "text_sha256": text_fingerprint(text),
            "source_locator": "overfit-1", "supporting": True,
        }],
        groups=["fixture:overfit-1"], metadata={"task": "qa"},
    )
    record["dataset_split"] = "train"
    plan = compile_answer_plan(record)
    canonical = plan["resolved_answer"]["canonical_text"]

    # Train a tokenizer that sees both "Warsaw" and "Warszawa" so they
    # become single known pieces — matching the edit-script granularity.
    tokenizer = TrainOnlySubwordTokenizer.train(
        ["Warsaw Warsaw Warszawa Warszawa " * 20], max_pieces=32,
    )
    serialized = serialize_answer_plan_for_model(plan)
    source_ids = tokenizer.encode(serialized)

    # Collect 4 identical copies.
    rows = []
    for i in range(4):
        target = "Warszawa"  # Polish inflection
        labels = compute_edit_target(canonical, target)
        fact_pos = find_fact_positions(source_ids, tokenizer, canonical)
        rows.append((source_ids, fact_pos, labels, target))

    config = {
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": 16,
        "output_vocab_size": tokenizer.vocab_size + 5,
        "dropout": 0.0,
    }
    model = build_copy_edit_transducer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    initial_loss = None
    for epoch in range(60):
        model.train()
        total_loss = 0.0
        for src_ids, fpos, labels, _ in rows:
            src_t = torch.tensor([src_ids], dtype=torch.long)
            fpos_t = torch.tensor([fpos], dtype=torch.long)
            logits = model(src_t, fpos_t)
            # Build target IDs: use source IDs as "KEEP" targets since
            # the canonical tokens are the same for all records.
            from nexus.realizer.copy_edit_transducer import build_label_ids

            target_ids = build_label_ids(labels, tokenizer)
            label_t = torch.tensor([target_ids], dtype=torch.long)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                label_t.reshape(-1),
                ignore_index=0,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if initial_loss is None:
            initial_loss = total_loss
        if total_loss < 0.01:
            break

    assert initial_loss is not None
    # After 60 epochs of overfitting 4 records with lr=0.05, loss
    # should drop substantially from its initial value.
    assert total_loss < initial_loss * 0.3, (
        f"overfit loss did not converge: {initial_loss:.4f} → {total_loss:.4f}"
    )


# ── Integration: edit script → transducer prediction roundtrip ──────


@pytest.mark.torch
def test_transducer_prediction_shape():
    torch = pytest.importorskip("torch")
    from nexus.realizer.copy_edit_transducer import build_copy_edit_transducer

    config = {
        "vocab_size": 512, "hidden_size": 16,
        "output_vocab_size": 512, "dropout": 0.0,
    }
    model = build_copy_edit_transducer(config)
    source = torch.tensor([[1, 7, 8, 2]], dtype=torch.long)
    fact_pos = torch.tensor([[1]], dtype=torch.long)

    tokenizer = TrainOnlySubwordTokenizer([])
    predictions = model.predict(source, fact_pos, tokenizer)
    assert len(predictions) == 1
    assert len(predictions[0]) == 1
    assert isinstance(predictions[0][0], str)
