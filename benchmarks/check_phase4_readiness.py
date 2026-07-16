"""Aggregate the complete Phase 0-4 contract before Realizer training.

The result is deliberately stricter than the model-only readiness artifact.
It requires a valid paired Stage 0 baseline, deterministic registered Stage 2
evidence, passing Stage 3 dialogue behaviour, an authenticated ER3 checkpoint,
the unique train-only dataset, and successful no-write runtime smoke tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import sha256_file, validate_dataset_manifest
from benchmarks.run_nexus_oracle import validate_oracle_artifact
from benchmarks.train_nexus_realizer import validate_readiness_for_training


SCHEMA_VERSION = "nexus-phase4-readiness-v1"
REQUIRED_HASH_SEEDS = {"0", "1", "42"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sidecar_valid(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        return False
    declared = sidecar.read_text(encoding="ascii").split()
    return bool(declared and declared[0] == sha256_file(path))


def _verify_er3_bundle(model_dir: Path) -> tuple[bool, dict[str, Any]]:
    errors: list[str] = []
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, {"errors": ["missing manifest.json"]}
    manifest = _load(manifest_path)
    files = manifest.get("files", {})
    hashes: dict[str, str] = {}
    for name in ("config.json", "vocab.json", "weights.pt"):
        meta = files.get(name, {})
        declared_path = meta.get("path")
        path = (
            (_project_root / declared_path).resolve()
            if isinstance(declared_path, str) and declared_path
            else (model_dir / name).resolve()
        )
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        digest = sha256_file(path)
        hashes[name] = digest
        if digest != meta.get("sha256"):
            errors.append(f"{name} SHA-256 mismatch")
        expected_size = meta.get("size", meta.get("size_bytes"))
        if expected_size is not None and path.stat().st_size != int(expected_size):
            errors.append(f"{name} size mismatch")
    return not errors, {
        "run_id": manifest.get("run_id"),
        "hashes": hashes,
        "errors": errors,
    }


def evaluate_phase4(
    *,
    config: dict[str, Any],
    config_path: Path,
    dataset_manifest: dict[str, Any],
    dataset_manifest_path: Path,
    oracle: dict[str, Any],
    stage0: dict[str, Any],
    stage2_runs: list[dict[str, Any]],
    stage3: dict[str, Any],
    readiness: dict[str, Any],
    preflight: dict[str, Any],
    overfit_smoke: dict[str, Any],
    er3_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, requirement: str) -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "value": value,
            "requirement": requirement,
        })

    dataset_errors = validate_dataset_manifest(
        dataset_manifest, dataset_manifest_path.parent
    )
    check(
        "dataset",
        not dataset_errors
        and int(dataset_manifest.get("pairs_accepted", 0)) >= 5000
        and dataset_manifest.get("target_met") is True,
        {
            "pairs": dataset_manifest.get("pairs_accepted"),
            "errors": dataset_errors,
        },
        ">=5000 unique verifier-passed pairs with valid manifest",
    )

    oracle_errors = validate_oracle_artifact(oracle)
    check(
        "oracle",
        not oracle_errors,
        {"questions": oracle.get("questions_total"), "errors": oracle_errors},
        "oracle publication guard passes",
    )

    paired_n = int(stage0.get("paired_comparison", {}).get("paired_n", 0))
    stage0_pass = (
        stage0.get("status") == "VALID"
        and stage0.get("publication_guard", {}).get("status") == "PASS"
        and int(stage0.get("questions_total", 0)) == 30
        and int(stage0.get("nexus", {}).get("answered", 0)) > 0
        and int(stage0.get("rag", {}).get("answered", 0)) > 0
        and paired_n > 0
    )
    check(
        "stage0",
        stage0_pass,
        {
            "status": stage0.get("status"),
            "paired_n": paired_n,
            "nexus_answered": stage0.get("nexus", {}).get("answered"),
            "rag_answered": stage0.get("rag", {}).get("answered"),
        },
        "valid 30-case paired NEXUS/RAG baseline",
    )

    seeds = {str(run.get("python_hash_seed")) for run in stage2_runs}
    canonical_hashes = {
        str(run.get("canonical_content_sha256")) for run in stage2_runs
    }
    stage2_pass = (
        len(stage2_runs) == 3
        and seeds == REQUIRED_HASH_SEEDS
        and len(canonical_hashes) == 1
        and all(
            run.get("protocol") == "registered_stage2_v1"
            and run.get("registered_gate_status") == "PASS"
            and run.get("status") == "PASS"
            and int(run.get("questions_total", 0)) == 30
            for run in stage2_runs
        )
    )
    check(
        "stage2",
        stage2_pass,
        {"seeds": sorted(seeds), "canonical_hashes": sorted(canonical_hashes)},
        "registered PASS for seeds 0/1/42 with one canonical hash",
    )

    stage3_metrics = stage3.get("metrics", {})
    stage3_pass = (
        stage3.get("status") == "PASS"
        and int(stage3.get("total_turns", 0)) == 110
        and float(stage3_metrics.get("reference_resolution", 0.0)) >= 0.70
        and float(stage3_metrics.get("single_turn_regression", 1.0)) <= 0.02
        and float(stage3_metrics.get("dialogue_state_latency_p50_ms", 999.0)) <= 5.0
    )
    check(
        "stage3",
        stage3_pass,
        stage3_metrics,
        "110 turns; reference>=0.70, regression<=0.02, dialogue p50<=5ms",
    )

    er3_ok, er3_details = _verify_er3_bundle(er3_dir)
    check("er3_bundle", er3_ok, er3_details, "config, vocabulary and weights match manifest")

    readiness_errors = validate_readiness_for_training(
        readiness, dataset_manifest_path, config_path
    )
    check(
        "model_readiness",
        not readiness_errors
        and readiness.get("status") == "READY_FOR_TRAINING"
        and not readiness.get("blocking_checks"),
        {"status": readiness.get("status"), "errors": readiness_errors},
        "READY_FOR_TRAINING for the exact config and dataset",
    )

    config_sha = sha256_file(config_path)
    dataset_sha = dataset_manifest.get("dataset_sha256")
    preflight_pass = (
        preflight.get("status") == "PREFLIGHT_PASS"
        and preflight.get("weights_written") is False
        and preflight.get("dataset_sha256") == dataset_sha
        and preflight.get("config_sha256") == config_sha
        and int(preflight.get("parameter_count", 0)) > 0
    )
    check("preflight", preflight_pass, preflight, "exact-input CPU forward/backward pass; no weights")

    initial_loss = float(overfit_smoke.get("initial_loss", 0.0))
    final_loss = float(overfit_smoke.get("final_loss", initial_loss))
    overfit_pass = (
        overfit_smoke.get("status") == "OVERFIT_PASS"
        and overfit_smoke.get("weights_written") is False
        and overfit_smoke.get("dataset_sha256") == dataset_sha
        and overfit_smoke.get("config_sha256") == config_sha
        and final_loss < initial_loss * 0.95
    )
    check(
        "overfit_smoke",
        overfit_pass,
        {"initial_loss": initial_loss, "final_loss": final_loss},
        "loss decreases by at least 5%; no weights written",
    )

    training = config.get("training", {})
    short_training = (
        int(training.get("epochs", 999)) <= 5
        and int(training.get("early_stopping_patience", 999)) <= 3
    )
    check(
        "short_training_policy",
        short_training,
        training,
        "default training <=5 epochs with patience <=3",
    )

    source_shas = {
        str(item.get("source_sha"))
        for item in [oracle, stage0, stage3, *stage2_runs]
        if item.get("source_sha")
    }
    check(
        "source_identity",
        len(source_shas) == 1,
        sorted(source_shas),
        "all Phase 0-3 evidence comes from one code commit",
    )

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GO_FOR_REALIZER_TRAINING" if not failed else "BLOCKED",
        "checks": checks,
        "blocking_checks": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/nexus_realizer_v1.json"))
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, action="append", required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--overfit-smoke", type=Path, required=True)
    parser.add_argument("--er3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact_paths = [
        args.oracle, args.stage0, *args.stage2, args.stage3,
        args.readiness, args.preflight, args.overfit_smoke,
    ]
    result = evaluate_phase4(
        config=_load(args.config),
        config_path=args.config,
        dataset_manifest=_load(args.dataset_manifest),
        dataset_manifest_path=args.dataset_manifest,
        oracle=_load(args.oracle),
        stage0=_load(args.stage0),
        stage2_runs=[_load(path) for path in args.stage2],
        stage3=_load(args.stage3),
        readiness=_load(args.readiness),
        preflight=_load(args.preflight),
        overfit_smoke=_load(args.overfit_smoke),
        er3_dir=args.er3_dir,
    )
    result["artifact_sidecars"] = {
        str(path): _sidecar_valid(path) for path in artifact_paths
    }
    if not all(result["artifact_sidecars"].values()):
        result["blocking_checks"].append("artifact_sidecars")
        result["status"] = "BLOCKED"
    inputs = [args.config, args.dataset_manifest, *artifact_paths]
    result["inputs"] = {str(path): sha256_file(path) for path in inputs}
    result["canonical_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "status": result["status"],
                "checks": result["checks"],
                "blocking_checks": result["blocking_checks"],
                "input_hashes": sorted(result["inputs"].values()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(args.output)
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({
        "status": result["status"],
        "blocking_checks": result["blocking_checks"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if result["status"] == "GO_FOR_REALIZER_TRAINING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
