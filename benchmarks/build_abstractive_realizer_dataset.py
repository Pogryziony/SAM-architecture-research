"""Build unique, multi-evidence train-only data for the next Realizer pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.abstractive_realizer_contracts import (
    SCHEMA_VERSION, assert_no_source_family_leakage, comparison_record_id,
    materialize_slot_template, normalize_answer, validate_abstractive_manifest,
    validate_abstractive_record,
)
from benchmarks.acquire_realizer_train_data import load_verified_acquisition
from benchmarks.realizer_contracts import (
    canonical_json, normalize_question, sha256_file, sha256_json,
    validate_dataset_manifest,
)


TARGET_TEMPLATE = (
    "[SOURCE_1] reports [VALUE_1] for [SUBJECT_1], while [SOURCE_2] reports "
    "[VALUE_2] for [SUBJECT_2]; the values are {relation}."
)


def _value(record: dict[str, Any]) -> str:
    answer = str(record.get("answer", "")).strip()
    subject = str(record.get("subject", "")).strip()
    predicate = str(record.get("predicate", "")).strip()
    if record.get("kind") == "config_value":
        marker = " is set to "
        return answer.rsplit(marker, 1)[-1].rstrip(".") if marker in answer else ""
    if record.get("kind") == "table_cell":
        prefix = f"For {subject}, {predicate} is "
        return answer[len(prefix):].rstrip(".") if answer.startswith(prefix) else ""
    return ""


def _usable(record: dict[str, Any]) -> bool:
    value = _value(record)
    return bool(
        value and len(value) <= 220
        and len(str(record.get("subject", ""))) <= 140
        and len(str(record.get("source_path", ""))) <= 180
    )


def _pair_source_families(
    records: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[str(record["source_family"])][key(record)].append(record)
    for family in grouped.values():
        for values in family.values():
            values.sort(key=lambda item: str(item["semantic_target_id"]))

    unused = set(grouped)
    output: list[tuple[dict[str, Any], dict[str, Any]]] = []
    while len(unused) >= 2:
        candidates: list[tuple[int, str, str, str]] = []
        ordered = sorted(unused)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                shared = set(grouped[left]) & set(grouped[right])
                count = sum(
                    min(len(grouped[left][item]), len(grouped[right][item]))
                    for item in shared
                )
                if count:
                    tie = hashlib.sha256(f"{left}|{right}".encode()).hexdigest()
                    candidates.append((count, tie, left, right))
        if not candidates:
            break
        _, _, left, right = max(candidates)
        for item in sorted(set(grouped[left]) & set(grouped[right])):
            output.extend(zip(grouped[left][item], grouped[right][item]))
        unused.remove(left)
        unused.remove(right)
    return output


def _build_record(
    left: dict[str, Any], right: dict[str, Any], *, source_sha: str,
) -> dict[str, Any]:
    left_value, right_value = _value(left), _value(right)
    relation = (
        "the same" if normalize_answer(left_value) == normalize_answer(right_value)
        else "different"
    )
    task_type = (
        "config_value_comparison"
        if left["kind"] == "config_value" else "table_value_comparison"
    )
    if task_type == "config_value_comparison":
        question = (
            f"Compare {left['subject']} in {left['source_path']} and "
            f"{right['source_path']}. What value does each source report, and "
            "are the values the same or different?"
        )
    else:
        question = (
            f"Compare {left['predicate']} for {left['subject']} in "
            f"{left['source_path']} with {right['subject']} in "
            f"{right['source_path']}. What does each source report, and are "
            "the values the same or different?"
        )
    slots = {
        "SOURCE_1": str(left["source_path"]),
        "VALUE_1": left_value,
        "SUBJECT_1": str(left["subject"]),
        "SOURCE_2": str(right["source_path"]),
        "VALUE_2": right_value,
        "SUBJECT_2": str(right["subject"]),
    }
    target = TARGET_TEMPLATE.format(relation=relation)
    composition = {
        "task_type": task_type,
        "operator": "compare_values",
        "relation": relation,
        "claim_ids": [left["semantic_target_id"], right["semantic_target_id"]],
    }
    source_families = [left["source_family"], right["source_family"]]
    component_id = "component_" + sha256_json(sorted(source_families))[:20]
    answer = materialize_slot_template(target, slots)
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": comparison_record_id(composition),
        "source_split": "train",
        "dataset_split": "",
        "question": question,
        "normalized_question_sha256": hashlib.sha256(
            normalize_question(question).encode("utf-8")
        ).hexdigest(),
        "question_type": "comparison",
        "intent": "multi_evidence_comparison",
        "canonical_entities": sorted(
            set(left["entities"] + right["entities"])
        ),
        "source_families": source_families,
        "source_family_component": component_id,
        "evidence_pack": {
            "node_facts": [
                {
                    "text": left["answer"], "source": left["source_path"],
                    "confidence": 1.0, "claim_id": left["semantic_target_id"],
                },
                {
                    "text": right["answer"], "source": right["source_path"],
                    "confidence": 1.0, "claim_id": right["semantic_target_id"],
                },
            ],
            "snippets": [], "paths": [], "facts": [], "numbers": [],
            "sources": [left["source_path"], right["source_path"]],
        },
        "slots": slots,
        "training_target": target,
        "answer": answer,
        "composition": composition,
        "target_verification": {
            "passed": True,
            "support_units": 2,
            "answer_is_single_candidate": False,
            "relation_verified": True,
        },
        "reasoning_audit": {
            "proof_valid": True,
            "provenance_coverage": 1.0,
            "recommended_action": "answer",
            "proof_steps": [
                {
                    "claim_id": item["semantic_target_id"],
                    "source_path": item["source_path"],
                    "source_locator": item["source_locator"],
                    "source_sha256": item["source_sha256"],
                }
                for item in (left, right)
            ],
        },
        "source_sha": source_sha,
        "config_hash": sha256_json({
            "schema": SCHEMA_VERSION, "target_template": TARGET_TEMPLATE,
        })[:16],
        "generator_identity": "nexus_multi_evidence_comparison_v1",
    }
    errors = validate_abstractive_record(record)
    if errors:
        raise ValueError(f"internal record error: {','.join(errors)}")
    return record


def _split_components(
    records: list[dict[str, Any]], validation_fraction: float, seed: int,
) -> dict[str, list[dict[str, Any]]]:
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        components[record["source_family_component"]].append(record)
    ordered = sorted(
        components,
        key=lambda component: sha256_json({"seed": seed, "component": component}),
    )
    target = max(1, round(len(records) * validation_fraction))
    validation_components: set[str] = set()
    count = 0
    for component in ordered:
        if count < target:
            validation_components.add(component)
            count += len(components[component])
    if len(validation_components) == len(components):
        validation_components.remove(ordered[-1])
    output = {"train": [], "validation": []}
    for original in records:
        record = dict(original)
        split = (
            "validation"
            if record["source_family_component"] in validation_components
            else "train"
        )
        record["dataset_split"] = split
        output[split].append(record)
    for values in output.values():
        values.sort(key=lambda item: item["id"])
    assert_no_source_family_leakage(output)
    return output


def build_abstractive_dataset(
    acquisition_records: list[dict[str, Any]],
    consumed_records: list[dict[str, Any]],
    consumed_validation_records: list[dict[str, Any]],
    output_dir: Path,
    *,
    source_sha: str,
    source_tree_sha: str = "",
    source_identity_mode: str = "local_checkout",
    acquisition_manifest_sha256: str,
    consumed_manifest_sha256: str,
    min_pairs: int = 1000,
    validation_fraction: float = 0.2,
    seed: int = 20260716,
) -> dict[str, Any]:
    if not source_sha:
        raise ValueError("source_sha is required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be within (0, 1)")
    quarantined = {
        entity
        for record in consumed_validation_records
        for entity in record.get("canonical_entities", [])
        if str(entity).startswith("TrainSource_")
    }
    old_questions = {normalize_question(item.get("question", "")) for item in consumed_records}
    old_answers = {normalize_answer(item.get("answer", "")) for item in consumed_records}
    eligible = [
        record for record in acquisition_records
        if record.get("source_family") not in quarantined and _usable(record)
    ]
    config_pairs = _pair_source_families(
        [item for item in eligible if item.get("kind") == "config_value"],
        lambda item: str(item["subject"]),
    )
    table_pairs = _pair_source_families(
        [item for item in eligible if item.get("kind") == "table_cell"],
        lambda item: str(item["predicate"]),
    )

    records: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    used_claims: set[str] = set()
    rejected: Counter[str] = Counter()
    for left, right in [*config_pairs, *table_pairs]:
        claim_ids = {left["semantic_target_id"], right["semantic_target_id"]}
        if claim_ids & used_claims:
            rejected["atomic_claim_reuse"] += 1
            continue
        record = _build_record(left, right, source_sha=source_sha)
        question = normalize_question(record["question"])
        answer = normalize_answer(record["answer"])
        if question in old_questions or answer in old_answers:
            rejected["consumed_text_overlap"] += 1
            continue
        if question in seen_questions or answer in seen_answers:
            rejected["duplicate_composition"] += 1
            continue
        records.append(record)
        used_claims.update(claim_ids)
        seen_questions.add(question)
        seen_answers.add(answer)

    if len(records) < 2:
        raise ValueError("fewer than two safe multi-evidence records")
    splits = _split_components(records, validation_fraction, seed)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_meta: dict[str, dict[str, Any]] = {}
    for split, rows in splits.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("xb") as handle:
            for row in rows:
                handle.write((canonical_json(row) + "\n").encode("utf-8"))
        split_meta[split] = {
            "path": path.name,
            "count": len(rows),
            "sha256": sha256_file(path),
            "source_families": len({f for row in rows for f in row["source_families"]}),
            "components": len({row["source_family_component"] for row in rows}),
        }

    operator_counts = Counter(
        row["composition"]["relation"] for rows in splits.values() for row in rows
    )
    task_counts = Counter(
        row["composition"]["task_type"] for rows in splits.values() for row in rows
    )
    builder_config = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "min_pairs": min_pairs,
        "source_split": "train",
        "consumed_validation_policy": "exclude_all_source_families",
        "atomic_claim_reuse": False,
        "pairing": "disjoint_source_family_greedy_max_shared_key_v1",
        "target_template": TARGET_TEMPLATE,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "source_tree_sha": source_tree_sha,
        "source_identity_mode": source_identity_mode,
        "config_hash": sha256_json(builder_config)[:16],
        "builder_config": builder_config,
        "builder_config_sha256": sha256_json(builder_config),
        "acquisition_manifest_sha256": acquisition_manifest_sha256,
        "consumed_manifest_sha256": consumed_manifest_sha256,
        "quarantined_source_families": len(quarantined),
        "eligible_atomic_claims": len(eligible),
        "atomic_claims_used": len(used_claims),
        "atomic_claim_reuse_count": 0,
        "old_question_overlap": 0,
        "old_answer_overlap": 0,
        "single_candidate_target_count": 0,
        "pairs_accepted": len(records),
        "pairs_rejected": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "min_pairs_target": min_pairs,
        "target_met": len(records) >= min_pairs,
        "counts_by_task": dict(sorted(task_counts.items())),
        "counts_by_relation": dict(sorted(operator_counts.items())),
        "splits": split_meta,
        "validation_fraction_actual": round(len(splits["validation"]) / len(records), 6),
        "dataset_sha256": sha256_json({
            name: meta["sha256"] for name, meta in sorted(split_meta.items())
        }),
    }
    errors = validate_abstractive_manifest(manifest, output_dir)
    if errors:
        raise RuntimeError("invalid generated dataset: " + "; ".join(errors[:20]))
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acquisition-manifest", type=Path,
        default=Path("data/realizer_train/source_claims_v1/manifest.json"),
    )
    parser.add_argument(
        "--consumed-manifest", type=Path,
        default=Path("data/distillation/realizer_v1/manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/distillation/realizer_abstractive_v1"),
    )
    parser.add_argument("--min-pairs", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args()

    acquisition, _ = load_verified_acquisition(
        args.acquisition_manifest, _project_root, verify_current_sources=False,
    )
    consumed_manifest = json.loads(args.consumed_manifest.read_text(encoding="utf-8"))
    errors = validate_dataset_manifest(consumed_manifest, args.consumed_manifest.parent)
    if errors:
        raise ValueError("invalid consumed manifest: " + "; ".join(errors))
    consumed_by_split = {
        split: [
            json.loads(line)
            for line in (
                args.consumed_manifest.parent / consumed_manifest["splits"][split]["path"]
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        for split in ("train", "validation")
    }
    consumed = consumed_by_split["train"] + consumed_by_split["validation"]
    if bool(args.source_commit) != bool(args.source_tree):
        raise ValueError("--source-commit and --source-tree must be provided together")
    source_sha = args.source_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    source_tree = args.source_tree or subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], text=True,
    ).strip()
    manifest = build_abstractive_dataset(
        acquisition, consumed, consumed_by_split["validation"], args.output_dir,
        source_sha=source_sha,
        source_tree_sha=source_tree,
        source_identity_mode=(
            "connector_published_equivalent_source"
            if args.source_commit else "local_checkout"
        ),
        acquisition_manifest_sha256=sha256_file(args.acquisition_manifest),
        consumed_manifest_sha256=sha256_file(args.consumed_manifest),
        min_pairs=args.min_pairs,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps({
        "status": "DATA_READY" if manifest["target_met"] else "DATA_BLOCKED",
        "pairs": manifest["pairs_accepted"],
        "train": manifest["splits"]["train"]["count"],
        "validation": manifest["splits"]["validation"]["count"],
        "dataset_sha256": manifest["dataset_sha256"],
    }, sort_keys=True))
    return 0 if manifest["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
