"""Prepare the sealed NEXUS AnswerPlan-v1 pilot artifacts without training.

The command reads only corpus-v2 train and validation.  The test split remains
sealed and is represented solely by its already-registered manifest metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_corpus_v2_contracts import (
    MANIFEST_SCHEMA_VERSION as CORPUS_MANIFEST_SCHEMA,
    canonical_json,
    iter_jsonl,
    normalized_text,
    sha256_file,
    sha256_json,
    validate_record,
)
from nexus.realizer.answer_plan import compile_answer_plan, validate_answer_plan
from nexus.realizer.plan_serializer import SERIALIZER_VERSION, serialize_answer_plan
from nexus.realizer.subword_tokenizer import TrainOnlySubwordTokenizer


RECORD_SCHEMA = "nexus-answer-plan-training-record-v1"
MANIFEST_SCHEMA = "nexus-answer-plan-dataset-manifest-v1"
REPORT_SCHEMA = "nexus-answer-plan-preparation-report-v1"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?)(?!\w)")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _output_record(record: dict[str, Any], output_split: str | None = None) -> dict[str, Any]:
    plan = compile_answer_plan(record)
    errors = validate_answer_plan(plan, record)
    if errors:
        raise ValueError(f"{record.get('id')}: {','.join(errors)}")
    return {
        "schema_version": RECORD_SCHEMA,
        "id": plan["id"],
        "source_record_id": record["id"],
        "dataset_split": output_split or record["dataset_split"],
        "language": record["language"],
        "operator": plan["operator"],
        "answer_plan": plan,
        "target": record["answer"],
        "document_group_ids": list(record["document_group_ids"]),
        "source": record["source"],
    }


def _compile_split(input_path: Path, output_path: Path, expected_count: int) -> dict[str, Any]:
    count = 0
    languages: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for source in iter_jsonl(input_path):
            source_errors = validate_record(source)
            if source_errors:
                raise ValueError(f"invalid source {source.get('id')}: {','.join(source_errors)}")
            row = _output_record(source)
            handle.write(canonical_json(row) + "\n")
            count += 1
            languages[row["language"]] += 1
            operators[row["operator"]] += 1
    if count != expected_count:
        raise ValueError(f"record count mismatch for {input_path}: {count} != {expected_count}")
    return {
        "path": output_path.name,
        "count": count,
        "sha256": sha256_file(output_path),
        "languages": dict(sorted(languages.items())),
        "operators": dict(sorted(operators.items())),
    }


def _select_abstention_holdout_groups(
    train_path: Path, minimum_abstentions: int,
) -> tuple[set[str], int]:
    """Choose whole PL document groups before training, never individual rows."""
    abstentions: Counter[str] = Counter()
    for record in iter_jsonl(train_path):
        groups = list(map(str, record.get("document_group_ids", [])))
        if record.get("language") == "pl" and record.get("semantic_plan", {}).get("operator") == "abstain" and len(groups) == 1:
            abstentions[groups[0]] += 1
    ordered = sorted(
        abstentions,
        key=lambda group: (sha256_json({"purpose": "answer_plan_v1_abstention_holdout", "group": group}), group),
    )
    selected: set[str] = set()
    count = 0
    for group in ordered:
        selected.add(group)
        count += abstentions[group]
        if count >= minimum_abstentions:
            break
    if count < minimum_abstentions:
        raise ValueError(f"only {count} document-disjoint abstentions available")
    return selected, count


def _compile_train_and_holdout(
    input_path: Path, train_path: Path, holdout_path: Path,
    selected_groups: set[str], expected_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    counters = {
        "train": {"count": 0, "languages": Counter(), "operators": Counter(), "groups": set()},
        "abstention_validation": {"count": 0, "languages": Counter(), "operators": Counter(), "groups": set()},
    }
    with train_path.open("w", encoding="utf-8", newline="\n") as train_handle, holdout_path.open("w", encoding="utf-8", newline="\n") as holdout_handle:
        for source in iter_jsonl(input_path):
            errors = validate_record(source)
            if errors:
                raise ValueError(f"invalid source {source.get('id')}: {','.join(errors)}")
            groups = set(map(str, source["document_group_ids"]))
            split = "abstention_validation" if groups & selected_groups else "train"
            row = _output_record(source, split)
            handle = holdout_handle if split == "abstention_validation" else train_handle
            handle.write(canonical_json(row) + "\n")
            counters[split]["count"] += 1
            counters[split]["languages"][row["language"]] += 1
            counters[split]["operators"][row["operator"]] += 1
            counters[split]["groups"].update(groups)
    if sum(item["count"] for item in counters.values()) != expected_count:
        raise ValueError("train/holdout record count mismatch")
    if counters["train"]["groups"] & counters["abstention_validation"]["groups"]:
        raise ValueError("document-group leakage between train and abstention holdout")
    result = []
    for split, path in (("train", train_path), ("abstention_validation", holdout_path)):
        item = counters[split]
        result.append({
            "path": path.name, "count": item["count"], "sha256": sha256_file(path),
            "languages": dict(sorted(item["languages"].items())),
            "operators": dict(sorted(item["operators"].items())),
        })
    return result[0], result[1]


def _training_texts(path: Path) -> Iterable[str]:
    for row in iter_jsonl(path):
        yield serialize_answer_plan(row["answer_plan"])
        yield row["target"]


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _summary(values: list[int]) -> dict[str, int]:
    return {
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "p999": _percentile(values, 0.999),
        "max": max(values, default=0),
    }


def _token_f1(prediction: str, target: str) -> float:
    predicted = Counter(_TOKEN_RE.findall(normalized_text(prediction)))
    gold = Counter(_TOKEN_RE.findall(normalized_text(target)))
    common = sum((predicted & gold).values())
    if not predicted or not gold:
        return float(predicted == gold)
    precision = common / sum(predicted.values())
    recall = common / sum(gold.values())
    return 2 * precision * recall / (precision + recall) if common else 0.0


def _baseline_predictions(row: dict[str, Any]) -> dict[str, str]:
    answer = row["answer_plan"]["resolved_answer"]["canonical_text"]
    language = row["language"]
    title = row["answer_plan"]["provenance"][0]["title"] or answer
    if row["operator"] == "abstain":
        answer = (
            "Podane dowody nie wystarczają do udzielenia odpowiedzi."
            if language == "pl" else
            "The provided evidence is insufficient to answer the question."
        )
    return {
        "plan_copy": answer,
        "registered_template": ("Odpowiedź: " if language == "pl" else "Answer: ") + answer,
        "evidence_title_pointer": title,
    }


def _evaluate_baselines(paths: Iterable[Path]) -> dict[str, Any]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    f1_sums: Counter[str] = Counter()
    slices: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    slice_f1: dict[str, float] = defaultdict(float)
    for row in (row for path in paths for row in iter_jsonl(path)):
        target = row["target"]
        plan = row["answer_plan"]
        allowed_numbers = set(_NUMBER_RE.findall(serialize_answer_plan(plan)))
        keys = ("overall", f"language:{row['language']}", f"operator:{row['operator']}")
        for name, prediction in _baseline_predictions(row).items():
            exact = normalized_text(prediction) == normalized_text(target)
            immutable = all(normalized_text(value) in normalized_text(prediction) for value in plan["resolved_answer"]["immutable_values"])
            unsupported = bool(set(_NUMBER_RE.findall(prediction)) - allowed_numbers)
            totals[name]["count"] += 1
            totals[name]["exact"] += int(exact)
            totals[name]["immutable"] += int(immutable)
            totals[name]["unsupported_number"] += int(unsupported)
            f1 = _token_f1(prediction, target)
            f1_sums[name] += f1
            for key in keys:
                slices[name][key]["count"] += 1
                slices[name][key]["exact"] += int(exact)
                slices[name][key]["immutable"] += int(immutable)
                slices[name][key]["unsupported_number"] += int(unsupported)
                slice_f1[f"{name}\0{key}"] += f1

    def metrics(counter: Counter[str], f1_sum: float) -> dict[str, Any]:
        count = counter["count"]
        return {
            "count": count,
            "exact_match": counter["exact"] / count,
            "token_f1": f1_sum / count,
            "immutable_preservation": counter["immutable"] / count,
            "unsupported_number_rate": counter["unsupported_number"] / count,
        }

    result = {}
    for name in sorted(totals):
        result[name] = {
            "overall": metrics(totals[name], f1_sums[name]),
            "slices": {
                key: metrics(counter, slice_f1[f"{name}\0{key}"])
                for key, counter in sorted(slices[name].items()) if key != "overall"
            },
        }
    return result


def prepare(
    corpus_root: Path, output_root: Path, max_pieces: int,
    abstention_holdout_minimum: int = 500,
    surface_transform_minimum: int = 20_000,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest_path = corpus_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA:
        raise ValueError("unsupported corpus-v2 manifest")

    split_meta: dict[str, Any] = {}
    train_source_meta = source_manifest["splits"]["train"]
    train_source_path = corpus_root / train_source_meta["path"]
    validation_source_meta = source_manifest["splits"]["validation"]
    validation_source_path = corpus_root / validation_source_meta["path"]
    for split, source_path, meta in (
        ("train", train_source_path, train_source_meta),
        ("validation", validation_source_path, validation_source_meta),
    ):
        if sha256_file(source_path) != meta["sha256"]:
            raise ValueError(f"{split} source hash mismatch")
    selected_groups, selected_abstentions = _select_abstention_holdout_groups(
        train_source_path, abstention_holdout_minimum
    )
    split_meta["train"], split_meta["abstention_validation"] = _compile_train_and_holdout(
        train_source_path, output_root / "train.jsonl",
        output_root / "abstention_validation.jsonl", selected_groups,
        train_source_meta["count"],
    )
    split_meta["validation"] = _compile_split(
        validation_source_path, output_root / "validation.jsonl",
        validation_source_meta["count"],
    )

    tokenizer = TrainOnlySubwordTokenizer.train(
        _training_texts(output_root / "train.jsonl"), max_pieces=max_pieces
    )
    tokenizer_value = tokenizer.to_dict()
    _write_json(output_root / "tokenizer.json", tokenizer_value)

    length_audit: dict[str, Any] = {"schema_version": "nexus-answer-plan-length-audit-v1", "splits": {}}
    all_inputs: list[int] = []
    all_targets: list[int] = []
    roundtrip_failures = 0
    target_visibility: dict[str, dict[str, int | float]] = {}
    for split in ("train", "validation", "abstention_validation"):
        input_lengths: list[int] = []
        target_lengths: list[int] = []
        target_visible = 0
        for row in iter_jsonl(output_root / f"{split}.jsonl"):
            source = serialize_answer_plan(row["answer_plan"])
            source_ids = tokenizer.encode(source)
            target_ids = tokenizer.encode(row["target"])
            target_visible += int(
                normalized_text(row["target"]) in normalized_text(source)
            )
            roundtrip_failures += int(tokenizer.decode(source_ids) != source)
            roundtrip_failures += int(tokenizer.decode(target_ids) != row["target"])
            input_lengths.append(len(source_ids))
            target_lengths.append(len(target_ids))
        all_inputs.extend(input_lengths)
        all_targets.extend(target_lengths)
        length_audit["splits"][split] = {
            "records": len(input_lengths),
            "input_tokens": _summary(input_lengths),
            "target_tokens": _summary(target_lengths),
        }
        target_visibility[split] = {
            "records": len(input_lengths),
            "target_visible_in_input": target_visible,
            "surface_transformation_records": len(input_lengths) - target_visible,
            "target_visibility_rate": target_visible / max(1, len(input_lengths)),
        }
    candidates = [256, 512, 1024, 2048, 4096]
    max_input = max(all_inputs, default=0)
    max_target = max(all_targets, default=0)
    chosen_budget = next((value for value in candidates if max_input <= value), None)
    chosen_target_budget = next((value for value in candidates if max_target <= value), None)
    length_audit.update({
        "tokenizer_sha256": tokenizer_value["canonical_sha256"],
        "roundtrip_failures": roundtrip_failures,
        "input_budget": chosen_budget,
        "input_budget_coverage": 1.0 if chosen_budget is not None else sum(v <= 4096 for v in all_inputs) / max(1, len(all_inputs)),
        "target_budget": chosen_target_budget,
        "target_budget_coverage": 1.0 if chosen_target_budget is not None else sum(v <= 4096 for v in all_targets) / max(1, len(all_targets)),
        "truncation_policy": "forbidden_for_required_plan_fields",
        "target_visibility": target_visibility,
    })
    _write_json(output_root / "length_audit.json", length_audit)

    baselines = {
        "schema_version": "nexus-answer-plan-deterministic-baselines-v1",
        "evaluation_split": "validation",
        "test_split_accessed": False,
        "metrics": _evaluate_baselines((
            output_root / "validation.jsonl",
            output_root / "abstention_validation.jsonl",
        )),
    }
    baselines["canonical_sha256"] = sha256_json(baselines)
    _write_json(output_root / "baselines.json", baselines)

    dataset_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "input_dataset_sha256": source_manifest["dataset_sha256"],
        "input_manifest_sha256": sha256_file(source_manifest_path),
        "serializer_version": SERIALIZER_VERSION,
        "splits": split_meta,
        "abstention_holdout": {
            "selection": "deterministic_whole_document_groups",
            "selected_group_count": len(selected_groups),
            "selected_abstention_count": selected_abstentions,
            "record_duplication": 0,
            "tokenizer_exposure": False,
        },
        "test_seal": {
            "status": "SEALED",
            "accessed": False,
            "source_path": source_manifest["splits"]["test"]["path"],
            "source_count": source_manifest["splits"]["test"]["count"],
            "registered_sha256": source_manifest["splits"]["test"]["sha256"],
            "allowed_use": "one final evaluation after checkpoint selection",
        },
        "expansion": {"input_records_per_output_record": 1, "synthetic": 0, "translated": 0, "paraphrased": 0},
        "tokenizer": {"path": "tokenizer.json", "file_sha256": sha256_file(output_root / "tokenizer.json"), "canonical_sha256": tokenizer_value["canonical_sha256"], "training_scope": "train_only"},
    }
    dataset_manifest["artifact_sha256"] = sha256_json(dataset_manifest)
    _write_json(output_root / "manifest.json", dataset_manifest)

    checks = {
        "compile_validation_complete": split_meta["validation"]["count"] == source_manifest["splits"]["validation"]["count"],
        "compile_train_complete": split_meta["train"]["count"] + split_meta["abstention_validation"]["count"] == source_manifest["splits"]["train"]["count"],
        "abstention_holdout_sufficient": split_meta["abstention_validation"]["operators"].get("abstain", 0) >= abstention_holdout_minimum,
        "one_to_one_no_expansion": True,
        "test_sealed": True,
        "tokenizer_train_only": tokenizer_value["training_scope"] == "train_only",
        "tokenizer_lossless": roundtrip_failures == 0,
        "no_required_field_truncation": chosen_budget is not None and chosen_target_budget is not None,
        "surface_transformation_train_minimum": target_visibility["train"]["surface_transformation_records"] >= surface_transform_minimum,
        "plan_copy_immutable": baselines["metrics"]["plan_copy"]["overall"]["immutable_preservation"] == 1.0,
        "plan_copy_unsupported_numbers": baselines["metrics"]["plan_copy"]["overall"]["unsupported_number_rate"] == 0.0,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "READY_FOR_BOUNDED_PILOT" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "blocking_checks": sorted(key for key, passed in checks.items() if not passed),
        "dataset_manifest_sha256": dataset_manifest["artifact_sha256"],
        "tokenizer_sha256": tokenizer_value["canonical_sha256"],
        "length_audit": length_audit,
        "baselines": baselines,
        "pilot_protocol": {
            "full_training_authorized": False,
            "stages": [
                {"name": "overfit_smoke", "records": 64, "epochs_max": 20, "writes_promoted_weights": False},
                {"name": "small_pilot", "records": 2048, "epochs_max": 1, "writes_promoted_weights": False},
                {"name": "representative_pilot", "records": 20000, "epochs_max": 1, "writes_promoted_weights": False},
                {"name": "full_train", "records": split_meta["train"]["count"], "epochs_max": 3, "authorized": False},
            ],
            "selection_split": "validation",
            "test_policy": "sealed until one checkpoint is selected",
            "stop_conditions": [
                "immutable_preservation < 0.99",
                "unsupported_number_rate > 0.01",
                "abstention regression in either language",
                "empty output or repetition regression",
                "no validation improvement after an epoch",
            ],
        },
    }
    report["canonical_sha256"] = sha256_json(report)
    _write_json(output_root / "readiness.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pieces", type=int, default=4096)
    parser.add_argument("--abstention-holdout-minimum", type=int, default=500)
    parser.add_argument("--surface-transform-minimum", type=int, default=20_000)
    args = parser.parse_args()
    report = prepare(
        args.corpus_root.resolve(), args.output.resolve(), args.max_pieces,
        args.abstention_holdout_minimum,
        args.surface_transform_minimum,
    )
    print(json.dumps({"status": report["status"], "canonical_sha256": report["canonical_sha256"], "blocking_checks": report["blocking_checks"]}, indent=2))
    return 0 if report["status"] == "READY_FOR_BOUNDED_PILOT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
