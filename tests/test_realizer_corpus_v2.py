from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmarks.build_realizer_corpus_v2 import (
    _load_previous_realizer_text, _make_record, _stable_eval_split,
)
from benchmarks.convert_hotpot_parquet import compact_row
from benchmarks.realizer_corpus_v2_contracts import (
    MANIFEST_SCHEMA_VERSION,
    SPLITS,
    canonical_json,
    sha256_file,
    sha256_json,
    text_fingerprint,
    validate_manifest,
    validate_record,
)


SOURCE = {
    "language": "en",
    "revision": "a" * 40,
    "license": "CC BY 4.0",
    "url": "https://example.test/dataset",
}


def _record(source_id: str, source_split: str, group: str, text: str, language="en"):
    source = dict(SOURCE, language=language)
    record = _make_record(
        dataset="fixture", source=source, artifact_sha256="b" * 64,
        source_split=source_split, source_id=source_id,
        question=f"Question {source_id}?", answer=f"Answer {source_id}", aliases=[],
        answerable=True, operator="extract", hops=1,
        evidence=[{
            "id": f"fixture:{source_id}:e0", "title": "Fixture", "text": text,
            "text_sha256": text_fingerprint(text), "source_locator": source_id,
            "supporting": True,
        }],
        groups=[group], metadata={"domain": "fixture", "task": "qa"},
    )
    return record


def _write_manifest(root: Path, rows: dict[str, list[dict]]) -> dict:
    split_meta = {}
    languages = Counter()
    operators = Counter()
    sources = Counter()
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        path.write_text(
            "".join(canonical_json(row) + "\n" for row in rows[split]),
            encoding="utf-8", newline="\n",
        )
        split_meta[split] = {
            "path": path.name, "count": len(rows[split]), "sha256": sha256_file(path),
        }
        for row in rows[split]:
            languages[row["language"]] += 1
            operators[row["semantic_plan"]["operator"]] += 1
            sources[row["source"]["dataset"]] += 1
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "splits": split_meta,
        "statistics": {
            "records_by_split": {split: len(rows[split]) for split in SPLITS},
            "records_by_language": dict(sorted(languages.items())),
            "records_by_operator": dict(sorted(operators.items())),
            "records_by_source": dict(sorted(sources.items())),
        },
        "quality_gates": {"minimum_train_records": 2, "minimum_language_fraction": 0.0},
    }
    manifest["dataset_sha256"] = sha256_json({
        split: split_meta[split]["sha256"] for split in SPLITS
    })
    return manifest


def test_valid_record_binds_evidence_to_verified_plan():
    record = _record("one", "train", "doc-one", "Supported fact one.")
    assert validate_record(record) == []
    assert record["semantic_plan"]["reasoning_owner"] == "nexus_upstream"
    assert record["semantic_plan"]["realizer_may_change_facts"] is False


def test_record_rejects_plan_evidence_mismatch():
    record = _record("one", "train", "doc-one", "Supported fact one.")
    record["semantic_plan"]["supporting_evidence_ids"] = []
    assert "plan_evidence_mismatch" in validate_record(record)


def test_manifest_detects_cross_split_document_and_evidence_leakage(tmp_path: Path):
    train = _record("train-en", "train", "shared-doc", "Shared exact evidence.")
    train_pl = _record("train-pl", "train", "pl-doc", "Inny dowód.", language="pl")
    validation = _record(
        "validation", "validation", "shared-doc", "Shared exact evidence."
    )
    validation["dataset_split"] = "validation"
    test = _record("test", "test", "test-doc", "Test evidence.")
    manifest = _write_manifest(tmp_path, {
        "train": [train, train_pl], "validation": [validation], "test": [test],
    })
    errors = validate_manifest(manifest, tmp_path)
    assert "train_eval_document_leakage" in errors
    assert "train_eval_evidence_leakage" in errors


def test_manifest_accepts_unique_group_disjoint_records(tmp_path: Path):
    train_en = _record("train-en", "train", "en-doc", "English evidence.")
    train_pl = _record("train-pl", "train", "pl-doc", "Polski dowód.", language="pl")
    validation = _record("validation", "validation", "val-doc", "Validation evidence.")
    validation["dataset_split"] = "validation"
    test = _record("test", "test", "test-doc", "Test evidence.")
    manifest = _write_manifest(tmp_path, {
        "train": [train_en, train_pl], "validation": [validation], "test": [test],
    })
    assert validate_manifest(manifest, tmp_path) == []


def test_eval_partition_is_deterministic_and_never_moves_train_or_test():
    assert _stable_eval_split("dataset", "id", "train") == "train"
    assert _stable_eval_split("dataset", "id", "test") == "test"
    first = _stable_eval_split("dataset", "id", "validation")
    assert first in {"validation", "test"}
    assert _stable_eval_split("dataset", "id", "validation") == first


def test_hotpot_conversion_keeps_only_gold_supporting_sentences():
    row = {
        "id": "x", "question": "Q?", "answer": "A", "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": ["A", "B"], "sent_id": [1, 0]},
        "context": {
            "title": ["A", "B", "Distractor"],
            "sentences": [["noise", "fact a"], ["fact b"], ["noise"]],
        },
    }
    converted = compact_row(row)
    assert [item["text"] for item in converted["evidence"]] == ["fact a", "fact b"]


def test_committed_registry_forbids_artificial_expansion():
    registry = json.loads(Path("training/realizer_corpus_v2_sources.json").read_text())
    assert registry["policy"] == {
        "allow_generated_records": False,
        "allow_translation": False,
        "allow_paraphrase_expansion": False,
        "maximum_records_per_source_record": 1,
        "split_unit": "source_document",
        "train_uses_native_train_only": True,
    }
    assert set(registry["sources"]) == {
        "hotpot_qa", "multihop_rag", "musique", "polqa", "poquad",
    }


def test_previous_realizer_records_are_registered_as_exclusions():
    questions, text_keys, manifests = _load_previous_realizer_text()
    assert sum(item["records_scanned"] for item in manifests) == 8_769
    assert questions
    assert text_keys
    assert {item["path"] for item in manifests} == {
        "data/distillation/realizer_v1/manifest.json",
        "data/distillation/realizer_abstractive_v1/manifest.json",
    }


def test_poquad_keeps_extractive_fact_as_alias_for_generative_target(tmp_path: Path):
    from benchmarks.build_realizer_corpus_v2 import _iter_poquad

    payload = {"data": [{
        "id": "doc", "title": "Tytuł", "url": "https://example.test/doc",
        "paragraphs": [{"context": "Wydarzenie miało miejsce w 1953 roku.", "qas": [{
            "question": "W którym roku?", "is_impossible": False,
            "answers": [{"text": "1953", "generative_answer": "w 1953 roku"}],
        }]}],
    }]}
    path = tmp_path / "poquad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    source = dict(SOURCE, language="pl")
    row = next(_iter_poquad(
        path, source, {"source_split": "train", "sha256": "b" * 64}, "poquad"
    ))
    assert row["answer"] == "w 1953 roku"
    assert row["answer_aliases"] == ["1953"]
