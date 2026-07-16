"""Contracts for leakage-safe multi-evidence Realizer training data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from benchmarks.realizer_contracts import (
    canonical_json, normalize_question, sha256_file, sha256_json,
)


SCHEMA_VERSION = "nexus-realizer-abstractive-v1"
TRAINING_CONFIG_SCHEMA = "nexus-realizer-abstractive-training-v1"
SLOT_NAMES = (
    "SOURCE_1", "VALUE_1", "SUBJECT_1",
    "SOURCE_2", "VALUE_2", "SUBJECT_2",
)


def normalize_answer(text: str) -> str:
    return " ".join(str(text).casefold().split())


def comparison_record_id(composition: dict[str, Any]) -> str:
    identity = {
        "contract": SCHEMA_VERSION,
        "task_type": composition.get("task_type"),
        "claim_ids": sorted(composition.get("claim_ids", [])),
    }
    return "abstractive_" + sha256_json(identity)[:24]


def materialize_slot_template(template: str, slots: dict[str, Any]) -> str:
    output = str(template)
    for name in SLOT_NAMES:
        output = output.replace(f"[{name}]", str(slots.get(name, "")))
    return output


def _evidence_texts(record: dict[str, Any]) -> list[str]:
    evidence = record.get("evidence_pack", {})
    if not isinstance(evidence, dict):
        return []
    return [
        str(item.get("text", "")).strip()
        for item in evidence.get("node_facts", [])
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ]


def validate_abstractive_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported_schema")
    for key in ("id", "question", "answer", "training_target", "source_sha", "config_hash"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            errors.append(f"missing_{key}")
    if record.get("source_split") != "train":
        errors.append("non_train_source")

    composition = record.get("composition", {})
    claim_ids = composition.get("claim_ids", []) if isinstance(composition, dict) else []
    source_families = record.get("source_families", [])
    if not isinstance(claim_ids, list) or len(set(claim_ids)) != 2:
        errors.append("requires_two_unique_claims")
    if not isinstance(source_families, list) or len(set(source_families)) != 2:
        errors.append("requires_two_unique_source_families")
    if record.get("id") != comparison_record_id(composition if isinstance(composition, dict) else {}):
        errors.append("unstable_id")

    question = str(record.get("question", ""))
    expected_question_hash = hashlib.sha256(
        normalize_question(question).encode("utf-8")
    ).hexdigest()
    if record.get("normalized_question_sha256") != expected_question_hash:
        errors.append("invalid_question_hash")

    evidence_texts = _evidence_texts(record)
    if len(evidence_texts) != 2 or len({normalize_answer(v) for v in evidence_texts}) != 2:
        errors.append("requires_two_distinct_evidence_units")
    answer = str(record.get("answer", "")).strip()
    if normalize_answer(answer) in {normalize_answer(value) for value in evidence_texts}:
        errors.append("answer_is_single_evidence_candidate")

    slots = record.get("slots", {})
    if not isinstance(slots, dict) or set(slots) != set(SLOT_NAMES):
        errors.append("invalid_slots")
        slots = {}
    else:
        if any(not str(slots[name]).strip() for name in SLOT_NAMES):
            errors.append("empty_slot")
        target = str(record.get("training_target", ""))
        if any(target.count(f"[{name}]") != 1 for name in SLOT_NAMES):
            errors.append("slot_placeholder_count")
        if materialize_slot_template(target, slots) != answer:
            errors.append("materialized_answer_mismatch")

    relation = composition.get("relation") if isinstance(composition, dict) else None
    if relation not in {"the same", "different"}:
        errors.append("invalid_relation")
    elif slots:
        values_equal = normalize_answer(slots["VALUE_1"]) == normalize_answer(slots["VALUE_2"])
        if values_equal is not (relation == "the same"):
            errors.append("incorrect_relation")
        if relation not in str(record.get("training_target", "")):
            errors.append("relation_missing_from_target")

    verification = record.get("target_verification", {})
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        errors.append("target_verifier_failed")
    audit = record.get("reasoning_audit", {})
    if not isinstance(audit, dict) or audit.get("proof_valid") is not True:
        errors.append("invalid_proof")
    elif audit.get("provenance_coverage") != 1.0:
        errors.append("incomplete_provenance")
    return sorted(set(errors))


def assert_no_source_family_leakage(
    splits: dict[str, Sequence[dict[str, Any]]]
) -> None:
    owners: dict[str, str] = {}
    ids: set[str] = set()
    questions: set[str] = set()
    answers: set[str] = set()
    claims: set[str] = set()
    for split, records in splits.items():
        for record in records:
            record_id = str(record.get("id", ""))
            if record_id in ids:
                raise ValueError(f"duplicate record id: {record_id}")
            ids.add(record_id)
            question = normalize_question(str(record.get("question", "")))
            answer = normalize_answer(str(record.get("answer", "")))
            if question in questions:
                raise ValueError(f"duplicate normalized question: {record_id}")
            if answer in answers:
                raise ValueError(f"duplicate normalized answer: {record_id}")
            questions.add(question)
            answers.add(answer)
            for claim in record.get("composition", {}).get("claim_ids", []):
                if claim in claims:
                    raise ValueError(f"atomic claim reused: {claim}")
                claims.add(claim)
            for family in record.get("source_families", []):
                previous = owners.setdefault(str(family), split)
                if previous != split:
                    raise ValueError(
                        f"source family leakage for {family}: {previous} vs {split}"
                    )


def validate_abstractive_manifest(
    manifest: dict[str, Any], root: Path
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    total = 0
    loaded: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split in loaded:
        meta = manifest.get("splits", {}).get(split, {})
        path_value = meta.get("path")
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"missing {split} path")
            continue
        path = root / path_value
        if not path.is_file():
            errors.append(f"missing {split} file")
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        loaded[split] = rows
        total += len(rows)
        if meta.get("count") != len(rows):
            errors.append(f"{split} count mismatch")
        if meta.get("sha256") != sha256_file(path):
            errors.append(f"{split} sha256 mismatch")
        for index, record in enumerate(rows):
            for error in validate_abstractive_record(record):
                errors.append(f"{split}[{index}]:{error}")
            if record.get("dataset_split") != split:
                errors.append(f"{split}[{index}]:split_mismatch")
    try:
        assert_no_source_family_leakage(loaded)
    except ValueError as exc:
        errors.append(str(exc))

    if manifest.get("pairs_accepted") != total:
        errors.append("accepted pair count mismatch")
    split_hashes = {
        name: manifest.get("splits", {}).get(name, {}).get("sha256", "")
        for name in loaded
    }
    if manifest.get("dataset_sha256") != sha256_json(split_hashes):
        errors.append("dataset sha256 mismatch")
    minimum = manifest.get("min_pairs_target")
    if not isinstance(minimum, int) or minimum < 1:
        errors.append("invalid minimum pair target")
    elif manifest.get("target_met") is not (total >= minimum):
        errors.append("target_met mismatch")
    actual_fraction = round(len(loaded["validation"]) / total, 6) if total else 0.0
    if manifest.get("validation_fraction_actual") != actual_fraction:
        errors.append("validation fraction mismatch")
    builder_config = manifest.get("builder_config")
    if not isinstance(builder_config, dict) or manifest.get("builder_config_sha256") != sha256_json(builder_config):
        errors.append("builder config sha256 mismatch")
    return errors


def load_abstractive_splits(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_abstractive_manifest(manifest, manifest_path.parent)
    if errors:
        raise ValueError("invalid abstractive dataset: " + "; ".join(errors[:20]))
    splits = {
        split: [
            json.loads(line)
            for line in (manifest_path.parent / manifest["splits"][split]["path"])
            .read_text(encoding="utf-8").splitlines()
            if line
        ]
        for split in ("train", "validation")
    }
    return manifest, splits


__all__ = [
    "SCHEMA_VERSION", "SLOT_NAMES", "TRAINING_CONFIG_SCHEMA",
    "assert_no_source_family_leakage", "comparison_record_id",
    "load_abstractive_splits", "materialize_slot_template", "normalize_answer",
    "validate_abstractive_manifest", "validate_abstractive_record",
]
