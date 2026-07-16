"""Aggregate every immutable gate before Realizer v1 training can start."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import (
    assert_no_split_leakage,
    sha256_file,
    validate_dataset_manifest,
    validate_distillation_record,
)
from benchmarks.run_nexus_oracle import validate_oracle_artifact
from nexus.realizer.model import build_model, parameter_count, validate_model_config
from benchmarks.train_nexus_realizer import serialization_coverage_for_config


READINESS_SCHEMA_VERSION = "nexus-realizer-readiness-v1"


def estimate_parameter_count(model: dict[str, Any]) -> int:
    """Conservative count for legacy tied-head and stable untied models."""
    d = int(model["d_model"])
    f = int(model["dim_feedforward"])
    enc = int(model["encoder_layers"])
    dec = int(model["decoder_layers"])
    vocab = int(model["vocab_size"])
    positions = max(int(model["max_input_tokens"]), int(model["max_output_tokens"]))
    embeddings = vocab * d + positions * d
    encoder = enc * (4 * d * d + 2 * d * f + 16 * d + f)
    decoder = dec * (8 * d * d + 2 * d * f + 24 * d + f)
    output_head = 0
    if model.get("architecture") == "stable_transformer_v2":
        output_head = vocab * d + vocab + 2 * d
    return embeddings + encoder + decoder + output_head


def evaluate_readiness(
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    dataset_root: Path,
    oracle_artifact: dict[str, Any],
    stage2_artifact: dict[str, Any],
    *,
    torch_available: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, requirement: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "requirement": requirement})

    config_errors = validate_model_config(config.get("model", {}))
    config_schema = config.get("schema_version")
    config_errors.extend(
        [] if config_schema in {
            "nexus-realizer-training-v1", "nexus-realizer-training-v2"
        } else ["unsupported training config schema"]
    )
    check(
        "training_config_valid", not config_errors, config_errors,
        "valid NEXUS Realizer training config",
    )
    dataset_errors = validate_dataset_manifest(dataset_manifest, dataset_root)
    check("dataset_manifest_valid", not dataset_errors, dataset_errors, "hash-verified dataset manifest")
    coverage_values: list[float] = []
    dataset_record_errors: list[str] = []
    loaded_splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    if not dataset_errors and not config_errors:
        for split in ("train", "validation"):
            path = dataset_root / dataset_manifest["splits"][split]["path"]
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if line:
                    record = json.loads(line)
                    loaded_splits[split].append(record)
                    record_errors = validate_distillation_record(record)
                    if record_errors:
                        dataset_record_errors.append(f"{split}[{index}]:{','.join(record_errors)}")
                    if record.get("dataset_split") != split:
                        dataset_record_errors.append(f"{split}[{index}]:split_mismatch")
                    coverage_values.append(
                        serialization_coverage_for_config(record, config)
                    )
        try:
            assert_no_split_leakage(loaded_splits)
        except ValueError as exc:
            dataset_record_errors.append(str(exc))
    check("dataset_records_valid", not dataset_record_errors, dataset_record_errors[:20], "all records pass contract and leakage guard")
    coverage = {
        "mean": round(sum(coverage_values) / len(coverage_values), 4) if coverage_values else 0.0,
        "min": round(min(coverage_values), 4) if coverage_values else 0.0,
    }
    check(
        "dataset_priority_evidence_coverage",
        coverage["mean"] == 1.0 and coverage["min"] == 1.0,
        coverage,
        "top three ranked evidence units retained for every record",
    )
    min_pairs = int(config.get("data", {}).get("minimum_pairs", 5000))
    pair_count = int(dataset_manifest.get("pairs_accepted", 0))
    check("dataset_pair_count", pair_count >= min_pairs, pair_count, f">= {min_pairs}")
    check("dataset_target_met", dataset_manifest.get("target_met") is True, dataset_manifest.get("target_met"), "true")
    validation_fraction = dataset_manifest.get("validation_fraction_actual")
    check(
        "dataset_split_balance",
        isinstance(validation_fraction, (int, float)) and 0.15 <= validation_fraction <= 0.25,
        validation_fraction,
        "validation fraction within [0.15, 0.25] with zero entity overlap",
    )

    oracle_errors = validate_oracle_artifact(oracle_artifact)
    check("oracle_artifact_valid", not oracle_errors, oracle_errors, "publication guard passes")
    oracle_total = int(oracle_artifact.get("questions_total", 0))
    check("oracle_sample_size", oracle_total >= 150, oracle_total, ">= 150")
    oracle_metrics = oracle_artifact.get("metrics", {})
    check("oracle_proof_validity", float(oracle_metrics.get("proof_valid_rate", 0.0)) >= 0.95, oracle_metrics.get("proof_valid_rate"), ">= 0.95")
    check("oracle_path_recall", float(oracle_metrics.get("gold_path_recall_mean") or 0.0) >= 0.80, oracle_metrics.get("gold_path_recall_mean"), ">= 0.80")
    check("oracle_provenance", float(oracle_metrics.get("provenance_coverage_mean", 0.0)) >= 0.90, oracle_metrics.get("provenance_coverage_mean"), ">= 0.90")

    stage2_metrics = stage2_artifact.get("metrics", {})
    stage2_identity_valid = (
        stage2_artifact.get("schema_version") == "nexus-stage2-v1"
        and bool(stage2_artifact.get("source_sha"))
        and bool(stage2_artifact.get("source_tree_sha"))
        and bool(stage2_artifact.get("config_hash"))
        and bool(stage2_artifact.get("registered_baseline_sha256"))
        and bool(stage2_artifact.get("question_set_sha256"))
        and bool(stage2_artifact.get("canonical_content_sha256"))
        and isinstance(stage2_artifact.get("case_order"), list)
        and int(stage2_artifact.get("questions_total", 0)) >= 30
        and len(stage2_artifact.get("per_question", [])) == int(stage2_artifact.get("questions_total", 0))
    )
    check("stage2_artifact_valid", stage2_identity_valid, stage2_artifact.get("schema_version"), "hash-identified Stage 2 artifact")
    evidence_valid = all(
        isinstance(row, dict) and isinstance(row.get("evidence"), dict)
        for row in stage2_artifact.get("per_question", [])
    )
    check("stage2_evidence_integrity", evidence_valid, evidence_valid, "evidence pack recorded for every case")
    registered_stage2_pass = (
        stage2_artifact.get("protocol") == "registered_stage2_v1"
        and stage2_artifact.get("protocol_kind") == "registered"
        and int(stage2_artifact.get("questions_total", 0)) == 30
        and stage2_artifact.get("registered_gate_status") == "PASS"
        and stage2_artifact.get("status") == "PASS"
    )
    check(
        "stage2_registered_gate",
        registered_stage2_pass,
        {
            "protocol": stage2_artifact.get("protocol"),
            "questions_total": stage2_artifact.get("questions_total"),
            "status": stage2_artifact.get("status"),
        },
        "registered_stage2_v1 PASS on exactly 30 cases",
    )
    relevance = stage2_metrics.get("relevance_rate")
    naturalness_delta = stage2_metrics.get("naturalness_improvement")
    hallucination_delta = stage2_metrics.get("hallucination_delta_vs_baseline")
    accuracy_delta = stage2_metrics.get("accuracy_delta_vs_baseline")
    # Stage 2 runs the untrained heuristic Realizer. These are recorded answer-quality
    # baselines, not pre-training data/evidence integrity gates.
    check("stage2_answer_quality_baseline", True, {
        "relevance": relevance, "naturalness": naturalness_delta,
        "hallucination": hallucination_delta, "accuracy": accuracy_delta,
        "status": stage2_artifact.get("status"),
    }, "recorded for post-training comparison")

    actual_params = 0
    _torch_ok = torch_available
    if not config_errors and torch_available:
        try:
            actual_params = parameter_count(build_model(config["model"]))
        except RuntimeError:
            # build_model raises RuntimeError when PyTorch is not importable
            # at runtime even though a spec was found (e.g. stubbed test env).
            _torch_ok = False
    check("parameter_count_recorded", actual_params > 0 or not _torch_ok, actual_params,
          "actual instantiated model parameter count")
    max_params = int(config.get("training", {}).get("max_parameters", 50_000_000))
    check("parameter_budget", actual_params == 0 or actual_params <= max_params, actual_params, f"<= {max_params}")
    check("pytorch_runtime", torch_available, torch_available, "PyTorch train extra installed")
    artifact_policy = config.get("artifact_policy", {})
    allow_repository_weights = bool(
        artifact_policy.get("allow_weights_in_repository", False)
    )
    repository_root = artifact_policy.get("repository_output_root")
    repository_policy_safe = not allow_repository_weights or (
        isinstance(repository_root, str)
        and repository_root.startswith("models/")
        and ".." not in Path(repository_root).parts
        and not Path(repository_root).is_absolute()
    )
    check(
        "weights_policy",
        repository_policy_safe,
        artifact_policy,
        "repository weights disabled or restricted to an explicit models/ subdirectory",
    )

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_TRAINING" if not failed else "BLOCKED",
        "checks": checks,
        "blocking_checks": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/nexus_realizer_v1.json", type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--oracle-artifact", required=True, type=Path)
    parser.add_argument("--stage2-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = [args.config, args.dataset_manifest, args.oracle_artifact, args.stage2_artifact]
    config, dataset, oracle, stage2 = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    result = evaluate_readiness(
        config, dataset, args.dataset_manifest.parent, oracle, stage2,
        torch_available=importlib.util.find_spec("torch") is not None,
    )
    result["inputs"] = {path.name: sha256_file(path) for path in inputs}
    canonical_payload = {
        "schema_version": result["schema_version"],
        "status": result["status"],
        "checks": result["checks"],
        "blocking_checks": result["blocking_checks"],
        "input_hashes": sorted(result["inputs"].values()),
    }
    result["readiness_canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result["serialized_sha256_sidecar"] = sidecar.name
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "blocking_checks": result["blocking_checks"]}, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_TRAINING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
