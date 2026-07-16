"""Immutable contracts for the multi-source NEXUS Realizer corpus v2."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "nexus-realizer-corpus-v2"
MANIFEST_SCHEMA_VERSION = "nexus-realizer-corpus-v2-manifest"
LANGUAGES = {"en", "pl"}
SPLITS = ("train", "validation", "test")
OPERATORS = {"extract", "compose_path", "compare", "abstain"}
ABSTENTION_TARGETS = {
    "en": "The provided evidence is insufficient to answer the question.",
    "pl": "Podane dowody nie wystarczają do udzielenia odpowiedzi.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def record_id(
    source_dataset: str, source_revision: str, source_split: str,
    source_record_id: str,
) -> str:
    return "rc2_" + sha256_json({
        "dataset": source_dataset,
        "revision": source_revision,
        "source_split": source_split,
        "source_record_id": source_record_id,
    })[:32]


def text_fingerprint(value: Any) -> str:
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()


def record_text_key(record: dict[str, Any]) -> str:
    return sha256_json({
        "question": normalized_text(record.get("question", "")),
        "answer": normalized_text(record.get("answer", "")),
    })


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported_schema")
    split = record.get("dataset_split")
    if split not in SPLITS:
        errors.append("invalid_dataset_split")
    language = record.get("language")
    if language not in LANGUAGES:
        errors.append("invalid_language")
    source = record.get("source", {})
    required_source = {
        "dataset", "revision", "source_split", "record_id", "license", "url",
    }
    if not isinstance(source, dict) or not required_source <= set(source):
        errors.append("invalid_source")
        source = {}
    expected_id = record_id(
        str(source.get("dataset", "")), str(source.get("revision", "")),
        str(source.get("source_split", "")), str(source.get("record_id", "")),
    )
    if record.get("id") != expected_id:
        errors.append("unstable_id")
    if split == "train" and source.get("source_split") != "train":
        errors.append("train_not_from_native_train")
    if not normalized_text(record.get("question", "")):
        errors.append("empty_question")
    answerable = record.get("answerable")
    if not isinstance(answerable, bool):
        errors.append("invalid_answerability")
    answer = str(record.get("answer", "")).strip()
    if not answer:
        errors.append("empty_answer")
    plan = record.get("semantic_plan", {})
    if not isinstance(plan, dict) or plan.get("verified") is not True:
        errors.append("unverified_semantic_plan")
        plan = {}
    if plan.get("operator") not in OPERATORS:
        errors.append("invalid_operator")
    if not isinstance(plan.get("hops"), int) or int(plan.get("hops", 0)) < 0:
        errors.append("invalid_hops")
    if answerable is False:
        if plan.get("operator") != "abstain":
            errors.append("unanswerable_without_abstention")
        if language in LANGUAGES and answer != ABSTENTION_TARGETS[language]:
            errors.append("noncanonical_abstention")
    elif plan.get("operator") == "abstain":
        errors.append("answerable_abstention")
    evidence = record.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        errors.append("missing_evidence")
        evidence = []
    evidence_ids: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            errors.append("invalid_evidence")
            continue
        if not str(item.get("id", "")).strip() or not str(item.get("text", "")).strip():
            errors.append("empty_evidence")
        if item.get("supporting") is not True:
            errors.append("non_supporting_evidence")
        item_id = str(item.get("id", ""))
        if item_id in evidence_ids:
            errors.append("duplicate_evidence_id")
        evidence_ids.add(item_id)
        if item.get("text_sha256") != text_fingerprint(item.get("text", "")):
            errors.append("invalid_evidence_hash")
    planned = plan.get("supporting_evidence_ids", [])
    if set(planned) != evidence_ids:
        errors.append("plan_evidence_mismatch")
    groups = record.get("document_group_ids", [])
    if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)):
        errors.append("invalid_document_groups")
    return sorted(set(errors))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_manifest(manifest: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported_manifest_schema")
    ids: set[str] = set()
    text_keys: set[str] = set()
    question_keys: set[str] = set()
    split_groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    split_evidence: dict[str, set[str]] = {split: set() for split in SPLITS}
    counts: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    train_operators: Counter[str] = Counter()
    train_sources: Counter[str] = Counter()
    for split in SPLITS:
        meta = manifest.get("splits", {}).get(split, {})
        path = root / str(meta.get("path", ""))
        if not path.is_file():
            errors.append(f"missing_{split}_file")
            continue
        if meta.get("sha256") != sha256_file(path):
            errors.append(f"{split}_hash_mismatch")
        split_count = 0
        for index, record in enumerate(iter_jsonl(path)):
            split_count += 1
            for error in validate_record(record):
                errors.append(f"{split}[{index}]:{error}")
            if record.get("dataset_split") != split:
                errors.append(f"{split}[{index}]:split_mismatch")
            rid = str(record.get("id", ""))
            text_key = record_text_key(record)
            question_key = normalized_text(record.get("question", ""))
            if rid in ids:
                errors.append(f"{split}[{index}]:duplicate_id")
            if text_key in text_keys:
                errors.append(f"{split}[{index}]:duplicate_question_answer")
            if question_key in question_keys:
                errors.append(f"{split}[{index}]:duplicate_question")
            ids.add(rid)
            text_keys.add(text_key)
            question_keys.add(question_key)
            split_groups[split].update(map(str, record.get("document_group_ids", [])))
            split_evidence[split].update(
                str(item.get("text_sha256", "")) for item in record.get("evidence", [])
            )
            counts[split] += 1
            languages[str(record.get("language"))] += 1
            operators[str(record.get("semantic_plan", {}).get("operator"))] += 1
            sources[str(record.get("source", {}).get("dataset"))] += 1
            if split == "train":
                train_operators[str(record.get("semantic_plan", {}).get("operator"))] += 1
                train_sources[str(record.get("source", {}).get("dataset"))] += 1
        if meta.get("count") != split_count:
            errors.append(f"{split}_count_mismatch")
    eval_groups = split_groups["validation"] | split_groups["test"]
    eval_evidence = split_evidence["validation"] | split_evidence["test"]
    if split_groups["train"] & eval_groups:
        errors.append("train_eval_document_leakage")
    if split_evidence["train"] & eval_evidence:
        errors.append("train_eval_evidence_leakage")
    if split_groups["validation"] & split_groups["test"]:
        errors.append("validation_test_document_leakage")
    if split_evidence["validation"] & split_evidence["test"]:
        errors.append("validation_test_evidence_leakage")
    expected = manifest.get("statistics", {})
    if expected.get("records_by_split") != dict(sorted(counts.items())):
        errors.append("split_statistics_mismatch")
    if expected.get("records_by_language") != dict(sorted(languages.items())):
        errors.append("language_statistics_mismatch")
    if expected.get("records_by_operator") != dict(sorted(operators.items())):
        errors.append("operator_statistics_mismatch")
    if expected.get("records_by_source") != dict(sorted(sources.items())):
        errors.append("source_statistics_mismatch")
    gates = manifest.get("quality_gates", {})
    if counts["train"] < int(gates.get("minimum_train_records", 50_000)):
        errors.append("insufficient_train_records")
    if counts["validation"] < int(gates.get("minimum_validation_records", 0)):
        errors.append("insufficient_validation_records")
    if counts["test"] < int(gates.get("minimum_test_records", 0)):
        errors.append("insufficient_test_records")
    total_train = max(1, counts["train"])
    for language in LANGUAGES:
        train_language = sum(
            1 for record in iter_jsonl(root / manifest["splits"]["train"]["path"])
            if record.get("language") == language
        )
        if train_language / total_train < float(gates.get("minimum_language_fraction", 0.25)):
            errors.append(f"insufficient_{language}_coverage")
    if len(train_sources) < int(gates.get("minimum_train_sources", 0)):
        errors.append("insufficient_train_source_diversity")
    for operator, gate_name in (
        ("abstain", "minimum_abstention_fraction"),
        ("compare", "minimum_comparison_fraction"),
        ("compose_path", "minimum_composition_fraction"),
    ):
        if train_operators[operator] / total_train < float(gates.get(gate_name, 0.0)):
            errors.append(f"insufficient_{operator}_coverage")
    if gates.get("zero_synthetic_records") is True:
        for field in ("synthetic_records", "translated_records", "paraphrase_expansions"):
            if expected.get(field) != 0:
                errors.append(f"nonzero_{field}")
    canonical_hashes = {
        split: manifest.get("splits", {}).get(split, {}).get("sha256", "")
        for split in SPLITS
    }
    if manifest.get("dataset_sha256") != sha256_json(canonical_hashes):
        errors.append("dataset_hash_mismatch")
    return sorted(set(errors))


__all__ = [
    "ABSTENTION_TARGETS", "LANGUAGES", "MANIFEST_SCHEMA_VERSION", "OPERATORS",
    "SCHEMA_VERSION", "SPLITS", "canonical_json", "iter_jsonl", "normalized_text",
    "record_id", "record_text_key", "sha256_file", "sha256_json", "text_fingerprint",
    "validate_manifest", "validate_record",
]
