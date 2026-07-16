from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.build_realizer_corpus_v2 import _make_record
from benchmarks.prepare_realizer_answer_plan_v1 import prepare
from benchmarks.check_answer_plan_full_training_readiness import check as check_full_readiness
from benchmarks.realizer_corpus_v2_contracts import (
    MANIFEST_SCHEMA_VERSION,
    canonical_json,
    sha256_file,
    sha256_json,
    text_fingerprint,
)
from nexus.realizer.answer_plan import compile_answer_plan, validate_answer_plan
from nexus.realizer.plan_serializer import serialize_answer_plan
from nexus.realizer.subword_tokenizer import TrainOnlySubwordTokenizer


SOURCE = {
    "language": "en",
    "revision": "a" * 40,
    "license": "CC BY 4.0",
    "url": "https://example.test/data",
}


def _record(source_id: str, split: str, language: str = "en") -> dict:
    source = dict(SOURCE, language=language)
    answer = "Warszawa" if language == "pl" else "Warsaw"
    text = f"The supported answer is {answer}."
    row = _make_record(
        dataset="fixture", source=source, artifact_sha256="b" * 64,
        source_split=split, source_id=source_id,
        question="Jaka jest odpowiedź?" if language == "pl" else "What is the answer?",
        answer=answer, aliases=[], answerable=True, operator="extract", hops=1,
        evidence=[{
            "id": f"fixture:{source_id}:e0", "title": "Fixture", "text": text,
            "text_sha256": text_fingerprint(text), "source_locator": source_id,
            "supporting": True,
        }],
        groups=[f"fixture:{source_id}"], metadata={"task": "qa"},
    )
    row["dataset_split"] = split
    return row


def test_answer_plan_resolves_answer_and_binds_exact_provenance():
    record = _record("one", "train")
    plan = compile_answer_plan(record)
    assert validate_answer_plan(plan, record) == []
    assert plan["resolved_answer"]["canonical_text"] == "Warsaw"
    assert plan["reasoning_owner"] == "nexus_upstream"
    assert plan["realizer_may_change_facts"] is False


def test_answer_plan_uses_fact_alias_without_leaking_surface_target():
    record = _record("one", "train")
    record["answer"] = "in the city of Warsaw"
    record["answer_aliases"] = ["Warsaw"]
    plan = compile_answer_plan(record)
    assert validate_answer_plan(plan, record) == []
    assert plan["resolved_answer"]["canonical_text"] == "Warsaw"
    assert "in the city of Warsaw" not in serialize_answer_plan(plan)


def test_answer_plan_rejects_fact_and_provenance_tampering():
    record = _record("one", "train")
    plan = compile_answer_plan(record)
    plan["resolved_answer"]["canonical_text"] = "Krakow"
    errors = validate_answer_plan(plan, record)
    assert "record_answer_mismatch" in errors
    assert "invalid_answer_hash" in errors
    assert "unstable_plan_id" in errors


def test_answer_plan_rejects_incomplete_provenance_without_source_record():
    plan = compile_answer_plan(_record("one", "train"))
    del plan["provenance"][0]["source_locator"]
    assert "invalid_provenance" in validate_answer_plan(plan)


def test_serializer_contains_resolved_fact_and_provenance_not_raw_evidence():
    record = _record("one", "train")
    serialized = serialize_answer_plan(compile_answer_plan(record))
    assert "[RESOLVED_ANSWER] Warsaw" in serialized
    assert "fixture:one:e0@one" in serialized
    assert record["evidence"][0]["text"] not in serialized


def test_train_only_subword_tokenizer_is_lossless_and_deterministic():
    texts = ["Zażółć gęślą jaźń.", "Warsaw is an answer.", "Warsaw again."]
    first = TrainOnlySubwordTokenizer.train(texts, max_pieces=16)
    second = TrainOnlySubwordTokenizer.train(reversed(texts), max_pieces=16)
    assert first.to_dict() == second.to_dict()
    for text in texts + ["Nieznane słowo: Łódź 🚆"]:
        assert first.decode(first.encode(text)) == text
    assert first.to_dict()["unknown_token"] is None


def test_preparation_keeps_test_sealed_and_builds_ready_artifacts(tmp_path: Path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "prepared"
    corpus.mkdir()
    rows = {
        "train": [_record("train-en", "train"), _record("train-pl", "train", "pl")],
        "validation": [_record("validation", "validation")],
    }
    split_meta = {}
    for split, values in rows.items():
        path = corpus / f"{split}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in values), encoding="utf-8")
        split_meta[split] = {"path": path.name, "count": len(values), "sha256": sha256_file(path)}
    split_meta["test"] = {"path": "DO_NOT_OPEN.jsonl", "count": 99, "sha256": "c" * 64}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_sha256": sha256_json({key: value["sha256"] for key, value in split_meta.items()}),
        "splits": split_meta,
    }
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = prepare(
        corpus, output, max_pieces=32, abstention_holdout_minimum=0,
        surface_transform_minimum=0,
    )
    assert report["status"] == "READY_FOR_BOUNDED_PILOT"
    prepared_manifest = json.loads((output / "manifest.json").read_text())
    assert prepared_manifest["test_seal"]["accessed"] is False
    assert prepared_manifest["splits"]["train"]["count"] == 2
    assert json.loads((output / "length_audit.json").read_text())["roundtrip_failures"] == 0


def test_pointer_generator_returns_normalized_log_probabilities():
    torch = pytest.importorskip("torch")
    from nexus.realizer.pointer_generator import build_pointer_generator

    model = build_pointer_generator({
        "vocab_size": 32, "hidden_size": 16, "dropout": 0.0,
    })
    source = torch.tensor([[1, 7, 8, 2]])
    target = torch.tensor([[1, 7, 8]])
    copy_mask = torch.tensor([[False, True, True, False]])
    log_probabilities = model(source, target, copy_mask)
    assert log_probabilities.shape == (1, 3, 32)
    assert torch.allclose(log_probabilities.exp().sum(-1), torch.ones(1, 3), atol=1e-5)


def test_full_training_readiness_fails_closed_without_passing_pilots(tmp_path: Path):
    data = tmp_path / "readiness.json"
    data.write_text(json.dumps({
        "status": "READY_FOR_BOUNDED_PILOT", "canonical_sha256": "a" * 64,
        "baselines": {"test_split_accessed": False},
        "pilot_protocol": {"full_training_authorized": False},
    }), encoding="utf-8")
    report = check_full_readiness(data, None, None, None)
    assert report["status"] == "FULL_TRAINING_BLOCKED"
    assert report["blocking_checks"] == [
        "overfit_generation_gate", "representative_generation_gate",
        "small_generation_gate",
    ]
    assert report["full_training_launched"] is False
