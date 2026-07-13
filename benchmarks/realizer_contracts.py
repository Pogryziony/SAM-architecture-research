"""Fail-closed contracts shared by NEXUS Realizer data and evaluation.

The functions in this module are deliberately model-free.  They validate
records, create leakage-safe entity-family splits, and hash every material
input so a training run can be reproduced without committing model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "nexus-realizer-v1"
REFUSAL_MARKERS = (
    "insufficient evidence",
    "cannot answer",
    "not enough evidence",
    "unable to determine",
    "i don't know",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(text: str) -> str:
    """Normalize only for duplicate detection, never for model input."""
    text = text.casefold().strip()
    text = re.sub(r"[^\w%]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def stable_example_id(question: str, entities: Sequence[str]) -> str:
    payload = {"question": normalize_question(question), "entities": sorted(set(entities))}
    return "realizer_" + sha256_json(payload)[:20]


def record_entities(record: dict[str, Any]) -> list[str]:
    values = record.get("canonical_entities", record.get("gold_entities", []))
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip() for value in values if str(value).strip()})


def validate_distillation_record(record: dict[str, Any]) -> list[str]:
    """Return all reasons a pair is unsafe to use for training."""
    errors: list[str] = []
    required_strings = ("id", "question", "answer", "source_sha", "config_hash")
    for key in required_strings:
        if not isinstance(record.get(key), str) or not record[key].strip():
            errors.append(f"missing_{key}")

    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    if len(question.strip()) < 8:
        errors.append("question_too_short")
    if len(answer.strip()) < 10:
        errors.append("answer_too_short")
    if any(marker in answer.casefold() for marker in REFUSAL_MARKERS):
        errors.append("refusal_detected")
    if record.get("source_split") != "train":
        errors.append("non_train_source")
    if not record_entities(record):
        errors.append("no_entities")

    evidence = record.get("evidence_pack")
    if not isinstance(evidence, dict) or not evidence:
        errors.append("no_evidence_pack")
    elif not any(evidence.get(key) for key in ("paths", "node_facts", "numbers")):
        errors.append("empty_evidence_pack")

    verification = record.get("target_verification")
    if not isinstance(verification, dict) or not verification.get("passed", False):
        errors.append("target_verifier_failed")

    audit = record.get("reasoning_audit")
    if not isinstance(audit, dict) or not audit.get("proof_valid", False):
        errors.append("invalid_proof")
    else:
        action = audit.get("recommended_action")
        coverage = audit.get("provenance_coverage")
        if action == "abstain":
            errors.append("audit_abstains")
        if action == "answer" and coverage != 1.0:
            errors.append("incomplete_provenance_for_answer")

    expected_id = stable_example_id(question, record_entities(record))
    if record.get("id") != expected_id:
        errors.append("unstable_id")
    expected_hash = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
    if record.get("normalized_question_sha256") != expected_hash:
        errors.append("invalid_question_hash")
    return sorted(set(errors))


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def entity_family_components(records: Sequence[dict[str, Any]]) -> list[list[int]]:
    """Group records transitively whenever they share a canonical entity."""
    uf = _UnionFind(len(records))
    owner: dict[str, int] = {}
    for index, record in enumerate(records):
        for entity in record_entities(record):
            if entity in owner:
                uf.union(index, owner[entity])
            else:
                owner[entity] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[uf.find(index)].append(index)
    return list(groups.values())


def split_by_entity_family(
    records: Sequence[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 20260711,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic train/validation splits with zero entity overlap."""
    if not records:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be within (0, 1)")
    components = entity_family_components(records)
    if len(components) < 2:
        raise ValueError("at least two disconnected entity families are required")

    ordered = sorted(
        components,
        key=lambda group: sha256_json({"seed": seed, "ids": sorted(records[i]["id"] for i in group)}),
    )
    target = max(1, math.ceil(len(records) * validation_fraction))
    validation_indices: set[int] = set()
    for group in ordered:
        if not validation_indices or len(validation_indices) < target:
            validation_indices.update(group)
    if len(validation_indices) == len(records):
        validation_indices.difference_update(ordered[-1])

    output = {"train": [], "validation": []}
    family_names: dict[int, str] = {}
    for group in components:
        entities = sorted({entity for i in group for entity in record_entities(records[i])})
        family = "family_" + sha256_json(entities)[:16]
        for i in group:
            family_names[i] = family

    for index, original in enumerate(records):
        record = dict(original)
        split = "validation" if index in validation_indices else "train"
        record["dataset_split"] = split
        record["entity_family"] = family_names[index]
        output[split].append(record)
    for values in output.values():
        values.sort(key=lambda item: item["id"])
    assert_no_split_leakage(output)
    return output


def assert_no_split_leakage(splits: dict[str, Sequence[dict[str, Any]]]) -> None:
    seen_questions: dict[str, str] = {}
    seen_entities: dict[str, str] = {}
    seen_ids: set[str] = set()
    for split_name, records in splits.items():
        for record in records:
            record_id = str(record.get("id", ""))
            if record_id in seen_ids:
                raise ValueError(f"duplicate record id: {record_id}")
            seen_ids.add(record_id)
            question = normalize_question(str(record.get("question", "")))
            previous = seen_questions.setdefault(question, split_name)
            if previous != split_name:
                raise ValueError(f"normalized question leakage between {previous} and {split_name}")
            for entity in record_entities(record):
                previous = seen_entities.setdefault(entity, split_name)
                if previous != split_name:
                    raise ValueError(f"entity leakage for {entity}: {previous} vs {split_name}")


def jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(record) + "\n").encode("utf-8") for record in records
    )


def validate_dataset_manifest(manifest: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not manifest.get("source_sha") or not manifest.get("config_hash"):
        errors.append("missing source/config identity")
    total = 0
    for split in ("train", "validation"):
        meta = manifest.get("splits", {}).get(split, {})
        path_value = meta.get("path")
        if not path_value:
            errors.append(f"missing {split} path")
            continue
        path = root / path_value
        if not path.is_file():
            errors.append(f"missing {split} file")
            continue
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        total += len(rows)
        if meta.get("count") != len(rows):
            errors.append(f"{split} count mismatch")
        if meta.get("sha256") != sha256_file(path):
            errors.append(f"{split} sha256 mismatch")
    if manifest.get("pairs_accepted") != total:
        errors.append("accepted pair count mismatch")
    split_hashes = {
        name: manifest.get("splits", {}).get(name, {}).get("sha256", "")
        for name in ("train", "validation")
    }
    if manifest.get("dataset_sha256") != sha256_json(split_hashes):
        errors.append("dataset sha256 mismatch")
    builder_config = manifest.get("builder_config")
    if not isinstance(builder_config, dict) or manifest.get("builder_config_sha256") != sha256_json(builder_config):
        errors.append("builder config sha256 mismatch")
    processed = manifest.get("questions_processed")
    rejected = manifest.get("pairs_rejected")
    if not isinstance(processed, int) or not isinstance(rejected, int):
        errors.append("missing processed/rejected counts")
    elif processed != total + rejected:
        errors.append("processed count mismatch")
    target = manifest.get("min_pairs_target")
    if not isinstance(target, int) or target < 1:
        errors.append("invalid minimum pair target")
    elif manifest.get("target_met") is not (total >= target):
        errors.append("target_met mismatch")
    actual_fraction = manifest.get("validation_fraction_actual")
    expected_fraction = round(
        manifest.get("splits", {}).get("validation", {}).get("count", 0) / total, 6
    ) if total else 0.0
    if actual_fraction != expected_fraction:
        errors.append("validation fraction mismatch")
    return errors


__all__ = [
    "SCHEMA_VERSION", "assert_no_split_leakage", "canonical_json",
    "jsonl_bytes", "normalize_question", "record_entities", "sha256_file",
    "sha256_json", "split_by_entity_family", "stable_example_id",
    "validate_dataset_manifest", "validate_distillation_record",
]
