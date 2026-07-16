"""Run no-write data, model, preflight and overfit gates for the next pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.abstractive_realizer_contracts import (
    SCHEMA_VERSION, TRAINING_CONFIG_SCHEMA, load_abstractive_splits,
    normalize_answer,
)
from benchmarks.realizer_contracts import sha256_file
from benchmarks.train_nexus_realizer import (
    _batch, _encode, _loss, serialization_coverage_for_config,
)
from nexus.realizer.model import build_model, parameter_count, validate_model_config


def evaluate_preparation(
    manifest_path: Path,
    config_path: Path,
    *,
    run_overfit_smoke: bool = True,
) -> dict[str, Any]:
    manifest, splits = load_abstractive_splits(manifest_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, requirement: str) -> None:
        checks.append({
            "name": name, "passed": bool(passed), "value": value,
            "requirement": requirement,
        })

    config_errors = validate_model_config(config.get("model", {}))
    if config.get("schema_version") != TRAINING_CONFIG_SCHEMA:
        config_errors.append("unsupported abstractive training config schema")
    if config.get("data", {}).get("manifest_schema") != SCHEMA_VERSION:
        config_errors.append("dataset/config schema mismatch")
    if config.get("data", {}).get("source_format") != "comparison_plan_v3":
        config_errors.append("source_format must be comparison_plan_v3")
    if config.get("data", {}).get("target_format") != "relation_label_v2":
        config_errors.append("target_format must be relation_label_v2")
    check(
        "training_config", not config_errors, config_errors,
        "valid constrained relation-selection pilot config",
    )

    pairs = int(manifest.get("pairs_accepted", 0))
    minimum = int(config.get("data", {}).get("minimum_pairs", 1000))
    check("pair_count", pairs >= minimum, pairs, f">= {minimum} unique compositions")
    check(
        "train_size", len(splits["train"]) >= 750, len(splits["train"]),
        ">= 750 train records after family-disjoint validation split",
    )
    fraction = float(manifest.get("validation_fraction_actual", 0.0))
    check("split_balance", 0.15 <= fraction <= 0.25, fraction, "validation fraction within [0.15, 0.25]")
    check(
        "consumed_validation_quarantine",
        int(manifest.get("quarantined_source_families", 0)) > 0,
        manifest.get("quarantined_source_families"),
        "all source families from consumed v1 validation excluded",
    )
    check(
        "no_consumed_text_overlap",
        manifest.get("old_question_overlap") == 0 and manifest.get("old_answer_overlap") == 0,
        {
            "questions": manifest.get("old_question_overlap"),
            "answers": manifest.get("old_answer_overlap"),
        },
        "zero normalized question/answer overlap with v1",
    )
    check(
        "multi_evidence_only",
        manifest.get("single_candidate_target_count") == 0,
        manifest.get("single_candidate_target_count"),
        "no target equals one evidence candidate",
    )
    check(
        "atomic_claim_non_reuse",
        manifest.get("atomic_claim_reuse_count") == 0
        and int(manifest.get("atomic_claims_used", 0)) == 2 * pairs,
        {
            "used": manifest.get("atomic_claims_used"),
            "reuse": manifest.get("atomic_claim_reuse_count"),
        },
        "each atomic claim contributes to exactly one composition",
    )
    relation_counts = manifest.get("counts_by_relation", {})
    largest_relation_share = max(relation_counts.values(), default=pairs) / max(pairs, 1)
    check(
        "relation_balance", largest_relation_share <= 0.80,
        {"counts": relation_counts, "largest_share": round(largest_relation_share, 6)},
        "no relation class exceeds 80%",
    )
    check(
        "task_coverage", len(manifest.get("counts_by_task", {})) >= 2,
        manifest.get("counts_by_task"), "at least two source task families",
    )
    verified_plans = sum(
        record.get("target_verification", {}).get("relation_verified") is True
        and (
            (normalize_answer(record.get("slots", {}).get("VALUE_1", ""))
             == normalize_answer(record.get("slots", {}).get("VALUE_2", "")))
            is (record.get("composition", {}).get("relation") == "the same")
        )
        for rows in splits.values() for record in rows
    )
    check(
        "symbolic_relation_plan", verified_plans == pairs, verified_plans,
        "every neural input plan is independently verified from both values",
    )

    coverage = [
        serialization_coverage_for_config(record, config)
        for rows in splits.values() for record in rows
    ] if not config_errors else []
    coverage_summary = {
        "min": min(coverage, default=0.0),
        "mean": round(sum(coverage) / len(coverage), 6) if coverage else 0.0,
    }
    check(
        "serialization_coverage",
        coverage_summary["min"] == 1.0 and coverage_summary["mean"] == 1.0,
        coverage_summary, "verified relation plan and both values retained",
    )

    try:
        import torch
    except ImportError:
        torch = None
    check("pytorch_cpu", torch is not None, bool(torch), "CPU PyTorch is installed")
    preflight: dict[str, Any] = {"status": "NOT_RUN"}
    overfit: dict[str, Any] = {"status": "NOT_RUN"}
    if torch is not None and not config_errors:
        seed = int(config["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        model = build_model(config["model"])
        params = parameter_count(model)
        maximum = int(config["training"]["max_parameters"])
        check("parameter_budget", params <= maximum, params, f"<= {maximum}")
        examples = _encode(splits["train"][:4], config)
        source, target = _batch(examples, torch)
        model.train()
        started = time.perf_counter()
        loss = _loss(model, source, target, torch)
        loss.backward()
        grad_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        initial_loss = float(loss.detach())
        preflight = {
            "status": "PREFLIGHT_PASS" if math.isfinite(initial_loss) and grad_finite else "PREFLIGHT_FAIL",
            "initial_loss": round(initial_loss, 6),
            "gradient_finite": grad_finite,
            "parameter_count": params,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "weights_written": False,
        }
        check(
            "preflight", preflight["status"] == "PREFLIGHT_PASS"
            and initial_loss <= float(config["training"]["initial_loss_max"]),
            preflight, "finite forward/backward below initial-loss ceiling; no weights",
        )

        if run_overfit_smoke:
            torch.manual_seed(seed)
            smoke_model = build_model(config["model"])
            chosen = []
            for relation in ("different", "the same"):
                chosen.append(next(
                    record for record in splits["train"]
                    if record["composition"]["relation"] == relation
                ))
            smoke_source, smoke_target = _batch(_encode(chosen, config), torch)
            optimizer = torch.optim.AdamW(
                smoke_model.parameters(),
                lr=float(config["training"]["learning_rate"]),
                weight_decay=0.0,
            )
            steps = int(config["training"].get("overfit_smoke_steps", 30))
            losses: list[float] = []
            started = time.perf_counter()
            smoke_model.train()
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                smoke_loss = _loss(smoke_model, smoke_source, smoke_target, torch)
                smoke_loss.backward()
                torch.nn.utils.clip_grad_norm_(smoke_model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(smoke_loss.detach()))
            reduction = (losses[0] - losses[-1]) / max(losses[0], 1e-9)
            overfit = {
                "status": "OVERFIT_PASS" if reduction >= 0.15 else "OVERFIT_FAIL",
                "steps": steps,
                "initial_loss": round(losses[0], 6),
                "final_loss": round(losses[-1], 6),
                "relative_reduction": round(reduction, 6),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "weights_written": False,
            }
            check(
                "overfit_smoke", overfit["status"] == "OVERFIT_PASS", overfit,
                ">=15% loss reduction on one example per relation; no weights",
            )

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema_version": "nexus-realizer-abstractive-readiness-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_BOUNDED_PILOT" if not failed else "BLOCKED",
        "blocking_checks": failed,
        "checks": checks,
        "preflight": preflight,
        "overfit_smoke": overfit,
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": getattr(torch, "__version__", None) if torch is not None else None,
            "cuda": bool(torch.cuda.is_available()) if torch is not None else False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/distillation/realizer_abstractive_v1/manifest.json"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("training/nexus_realizer_abstractive_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-overfit-smoke", action="store_true")
    args = parser.parse_args()
    result = evaluate_preparation(
        args.manifest, args.config,
        run_overfit_smoke=not args.skip_overfit_smoke,
    )
    canonical = copy.deepcopy(result)
    canonical.pop("created_utc")
    canonical["preflight"].pop("elapsed_seconds", None)
    canonical["overfit_smoke"].pop("elapsed_seconds", None)
    for item in canonical["checks"]:
        if item["name"] in {"preflight", "overfit_smoke"} and isinstance(item["value"], dict):
            item["value"].pop("elapsed_seconds", None)
    result["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar.write_text(f"{sha256_file(args.output)}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({
        "status": result["status"],
        "blocking_checks": result["blocking_checks"],
        "canonical_sha256": result["canonical_sha256"],
    }, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_BOUNDED_PILOT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
