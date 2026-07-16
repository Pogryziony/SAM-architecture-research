"""CPU trainer and no-write preflight for NEXUS Realizer v1.

Run ``--mode preflight`` before any training.  ``--mode train`` additionally
requires a readiness artifact with status READY_FOR_TRAINING.  Repository
weight output is accepted only below the configured ``models/realizer`` root;
external output directories remain supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import (
    assert_no_split_leakage,
    canonical_json,
    validate_dataset_manifest,
    validate_distillation_record,
    sha256_file,
)
from nexus.realizer.model import build_model, parameter_count, validate_model_config
from nexus.realizer.tokenizer import ByteTokenizer


def load_training_inputs(manifest_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    abstractive = manifest.get("schema_version") == "nexus-realizer-abstractive-v1"
    if abstractive:
        from benchmarks.abstractive_realizer_contracts import validate_abstractive_manifest
        errors = validate_abstractive_manifest(manifest, manifest_path.parent)
    else:
        errors = validate_dataset_manifest(manifest, manifest_path.parent)
    errors.extend(validate_model_config(config.get("model", {})))
    if config.get("schema_version") not in {
        "nexus-realizer-training-v1", "nexus-realizer-training-v2",
        "nexus-realizer-abstractive-training-v1",
    }:
        errors.append("unsupported training config schema")
    if manifest.get("schema_version") != config.get("data", {}).get("manifest_schema"):
        errors.append("dataset/config schema mismatch")
    if errors:
        raise ValueError("invalid training inputs: " + "; ".join(errors))
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        path = manifest_path.parent / manifest["splits"][split]["path"]
        splits[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for index, record in enumerate(splits[split]):
            if abstractive:
                from benchmarks.abstractive_realizer_contracts import validate_abstractive_record
                record_errors = validate_abstractive_record(record)
            else:
                record_errors = validate_distillation_record(record)
            if record_errors:
                raise ValueError(
                    f"invalid {split} record {index}: " + "; ".join(record_errors)
                )
            if record.get("dataset_split") != split:
                raise ValueError(f"record split mismatch in {split} at index {index}")
    if abstractive:
        from benchmarks.abstractive_realizer_contracts import assert_no_source_family_leakage
        assert_no_source_family_leakage(splits)
    else:
        assert_no_split_leakage(splits)
    return manifest, config, splits


def apply_training_overrides(
    config: dict[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply validated runtime parameters to the effective training config."""

    training = config.setdefault("training", {})
    allowed = {
        "epochs", "early_stopping_patience", "batch_size",
        "learning_rate", "weight_decay", "max_parameters",
    }
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        target = "early_stopping_patience" if key == "patience" else key
        if target not in allowed:
            continue
        if target in {"epochs", "early_stopping_patience", "batch_size"}:
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{target} must be a positive integer")
        if target == "learning_rate" and float(value) <= 0:
            raise ValueError("learning_rate must be positive")
        training[target] = value
    return config


def effective_config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _evidence_units(record: dict[str, Any]) -> dict[str, list[Any]]:
    evidence = record["evidence_pack"]
    units: dict[str, list[Any]] = {"facts": [], "numbers": [], "edges": [], "sources": []}
    for item in evidence.get("node_facts", []):
        if isinstance(item, dict) and item.get("text"):
            units["facts"].append(str(item["text"]))
    for item in evidence.get("numbers", []):
        units["numbers"].append(item)
    for path in evidence.get("paths", []):
        for edge in path.get("edges", []):
            if isinstance(edge, dict):
                units["edges"].append({
                    "from": edge.get("from", ""),
                    "relation": edge.get("type", ""),
                    "to": edge.get("to", ""),
                })
    for source in evidence.get("sources", []):
        units["sources"].append(str(source))
    return units


def _serialize_with_units(record: dict[str, Any], max_bytes: int | None) -> tuple[str, dict[str, list[Any]]]:
    units = _evidence_units(record)
    prefix = "[QUESTION] " + str(record["question"]).strip() + " [EVIDENCE] "
    if max_bytes is None:
        return prefix + canonical_json(units), units
    if max_bytes < 64:
        raise ValueError("max input budget is too small")
    selected: dict[str, list[Any]] = {key: [] for key in units}
    while len((prefix + canonical_json(selected)).encode("utf-8")) > max_bytes and len(prefix) > 32:
        prefix = prefix[:-1]
    for key in ("facts", "numbers", "edges", "sources"):
        for unit in units[key]:
            candidate = {name: list(values) for name, values in selected.items()}
            candidate[key].append(unit)
            serialized = prefix + canonical_json(candidate)
            if len(serialized.encode("utf-8")) <= max_bytes:
                selected = candidate
    return prefix + canonical_json(selected), selected


def serialize_source(record: dict[str, Any], max_bytes: int | None = None) -> str:
    """Serialize compact JSON evidence, highest-value first, within budget."""
    return _serialize_with_units(record, max_bytes)[0]


def _utf8_prefix(text: str, max_bytes: int) -> str:
    """Return a valid UTF-8 prefix that fits the exact byte budget."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def serialize_grounded_source(record: dict[str, Any], max_bytes: int) -> str:
    """Put the answer-bearing evidence first and omit graph bookkeeping.

    The v1 serializer spent most of its 1024-token budget on JSON keys, edge
    metadata and repeated source paths.  For realization the complete fact is
    the important unit.  It is placed first so truncation cannot silently keep
    the question while dropping the evidence required to answer it.
    """
    from nexus.realizer.grounded import evidence_candidates

    candidates = evidence_candidates(record)
    if not candidates:
        return _utf8_prefix(
            "[EVIDENCE] Insufficient evidence.\n[QUESTION] "
            + str(record.get("question", "")).strip(),
            max_bytes,
        )
    primary = candidates[0].text
    prefix = f"[EVIDENCE] {primary}\n[QUESTION] "
    if len(prefix.encode("utf-8")) >= max_bytes:
        return _utf8_prefix(prefix, max_bytes)
    question_budget = max_bytes - len(prefix.encode("utf-8"))
    return prefix + _utf8_prefix(str(record.get("question", "")).strip(), question_budget)


def serialize_comparison_slots(record: dict[str, Any], max_bytes: int) -> str:
    """Serialize both evidence bindings without exposing the target template."""
    slots = record.get("slots", {})
    text = (
        "[TASK] compare two evidence values\n"
        f"[EVIDENCE_1] source={slots.get('SOURCE_1', '')} | "
        f"subject={slots.get('SUBJECT_1', '')} | value={slots.get('VALUE_1', '')}\n"
        f"[EVIDENCE_2] source={slots.get('SOURCE_2', '')} | "
        f"subject={slots.get('SUBJECT_2', '')} | value={slots.get('VALUE_2', '')}\n"
        f"[QUESTION] {str(record.get('question', '')).strip()}"
    )
    return _utf8_prefix(text, max_bytes)


def serialize_source_for_config(record: dict[str, Any], config: dict[str, Any]) -> str:
    model_config = config["model"]
    max_bytes = int(model_config["max_input_tokens"]) - 2
    source_format = config.get("data", {}).get("source_format", "evidence_json_v1")
    if source_format == "grounded_compact_v2":
        return serialize_grounded_source(record, max_bytes)
    if source_format == "comparison_slots_v1":
        return serialize_comparison_slots(record, max_bytes)
    if source_format != "evidence_json_v1":
        raise ValueError(f"unsupported source format: {source_format}")
    return serialize_source(record, max_bytes)


def serialization_coverage_for_config(
    record: dict[str, Any], config: dict[str, Any]
) -> float:
    """Measure retention of evidence needed by the configured source format."""
    source_format = config.get("data", {}).get("source_format", "evidence_json_v1")
    if source_format == "evidence_json_v1":
        return serialization_coverage(
            record, int(config["model"]["max_input_tokens"]) - 2
        )
    if source_format == "comparison_slots_v1":
        serialized = serialize_source_for_config(record, config)
        slots = record.get("slots", {})
        required = [
            str(slots.get(name, ""))
            for name in (
                "SOURCE_1", "VALUE_1", "SUBJECT_1",
                "SOURCE_2", "VALUE_2", "SUBJECT_2",
            )
        ]
        return 1.0 if all(value and value in serialized for value in required) else 0.0
    if source_format != "grounded_compact_v2":
        raise ValueError(f"unsupported source format: {source_format}")
    from nexus.realizer.grounded import evidence_candidates

    candidates = evidence_candidates(record)
    if not candidates:
        return 1.0
    serialized = serialize_source_for_config(record, config)
    return 1.0 if candidates[0].text in serialized else 0.0


def training_target_for_config(
    record: dict[str, Any], config: dict[str, Any]
) -> str:
    target_format = config.get("data", {}).get("target_format", "answer_text_v1")
    if target_format == "slot_template_v1":
        target = record.get("training_target")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("slot_template_v1 record has no training_target")
        return target
    if target_format != "answer_text_v1":
        raise ValueError(f"unsupported target format: {target_format}")
    return str(record["answer"])


def serialization_coverage(record: dict[str, Any], max_bytes: int) -> float:
    """Coverage of the top three ranked fact/number units (or edges)."""
    full = _evidence_units(record)
    _, compact = _serialize_with_units(record, max_bytes)
    total = min(3, len(full["facts"]) + len(full["numbers"]))
    retained = min(3, len(compact["facts"]) + len(compact["numbers"]))
    if total == 0:
        total = min(3, len(full["edges"]))
        retained = min(3, len(compact["edges"]))
    return 1.0 if total == 0 else retained / total


def _encode(records: Sequence[dict[str, Any]], config: dict[str, Any]) -> list[tuple[list[int], list[int]]]:
    tokenizer = ByteTokenizer()
    model_config = config["model"]
    return [
        (
            tokenizer.encode(
                serialize_source_for_config(record, config),
                model_config["max_input_tokens"],
            ),
            tokenizer.encode(
                training_target_for_config(record, config),
                model_config["max_output_tokens"],
            ),
        )
        for record in records
    ]


def _batch(examples: Sequence[tuple[list[int], list[int]]], torch: Any):
    if not examples:
        raise ValueError("cannot build an empty batch")
    source_length = max(len(source) for source, _ in examples)
    target_length = max(len(target) for _, target in examples)
    source = torch.zeros((len(examples), source_length), dtype=torch.long)
    target = torch.zeros((len(examples), target_length), dtype=torch.long)
    for index, (source_ids, target_ids) in enumerate(examples):
        source[index, : len(source_ids)] = torch.tensor(source_ids)
        target[index, : len(target_ids)] = torch.tensor(target_ids)
    return source, target


def _loss(model: Any, source: Any, target: Any, torch: Any):
    logits = model(source, target[:, :-1])
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target[:, 1:].reshape(-1), ignore_index=0
    )


def _assert_external_output(
    output_dir: Path,
    *,
    allow_in_repository: bool = False,
    repository_output_root: str | None = None,
) -> None:
    resolved = output_dir.resolve()
    inside_repository = resolved == _project_root or _project_root in resolved.parents
    if inside_repository and not allow_in_repository:
        raise ValueError("weight output must be outside the git repository")
    if inside_repository:
        if not repository_output_root:
            raise ValueError(
                "repository weight output requires repository_output_root"
            )
        allowed_root = (_project_root / repository_output_root).resolve()
        if resolved != allowed_root and allowed_root not in resolved.parents:
            raise ValueError(
                f"repository weight output must be under {allowed_root}"
            )


def _assert_configured_output(output_dir: Path, config: dict[str, Any]) -> None:
    policy = config.get("artifact_policy", {})
    _assert_external_output(
        output_dir,
        allow_in_repository=bool(policy.get("allow_weights_in_repository", False)),
        repository_output_root=policy.get("repository_output_root"),
    )


def validate_readiness_for_training(
    readiness: dict[str, Any], manifest_path: Path, config_path: Path,
) -> list[str]:
    errors: list[str] = []
    if readiness.get("schema_version") != "nexus-realizer-readiness-v1":
        errors.append("unsupported readiness schema")
    checks = readiness.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("missing readiness checks")
        checks = []
    if any(item.get("passed") is not True for item in checks if isinstance(item, dict)):
        errors.append("readiness contains failed checks")
    if readiness.get("blocking_checks"):
        errors.append("readiness contains blocking checks")
    canonical_payload = {
        "schema_version": readiness.get("schema_version"),
        "status": readiness.get("status"),
        "checks": checks,
        "blocking_checks": readiness.get("blocking_checks", []),
        "input_hashes": sorted(readiness.get("inputs", {}).values()),
    }
    expected = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if readiness.get("readiness_canonical_sha256") != expected:
        errors.append("readiness canonical payload hash mismatch")
    input_hashes = set(readiness.get("inputs", {}).values())
    if sha256_file(manifest_path) not in input_hashes:
        errors.append("readiness does not identify this dataset manifest")
    if sha256_file(config_path) not in input_hashes:
        errors.append("readiness does not identify this training config")
    derived_status = "READY_FOR_TRAINING" if not errors else "BLOCKED"
    if readiness.get("status") != derived_status:
        errors.append("readiness status is inconsistent with checks")
    return errors


def run(
    manifest_path: Path,
    config_path: Path,
    *,
    mode: str,
    readiness_path: Path | None = None,
    output_dir: Path | None = None,
    epoch_override: int | None = None,
    training_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest, config, splits = load_training_inputs(manifest_path, config_path)
    if epoch_override is not None and epoch_override > 0:
        training_overrides = {**(training_overrides or {}), "epochs": epoch_override}
    apply_training_overrides(config, training_overrides)
    effective_training = dict(config["training"])
    effective_hash = effective_config_sha256(config)
    config_file_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if mode == "train":
        if readiness_path is None:
            raise ValueError("--readiness is required for training")
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        readiness_errors = validate_readiness_for_training(
            readiness, manifest_path, config_path
        )
        if readiness_errors:
            raise ValueError("training blocked: " + "; ".join(readiness_errors))
        if output_dir is None:
            raise ValueError("--output-dir is required for training")
        _assert_configured_output(output_dir, config)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch unavailable; install with `pip install -e '.[train]'`") from exc

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = build_model(config["model"])
    params = parameter_count(model)
    if params > int(config["training"]["max_parameters"]):
        raise ValueError(f"parameter budget exceeded: {params}")

    train_examples = _encode(splits["train"], config)
    validation_examples = _encode(splits["validation"], config)
    coverage_values = [
        serialization_coverage_for_config(record, config)
        for records in splits.values()
        for record in records
    ]
    coverage_summary = {
        "mean": round(sum(coverage_values) / len(coverage_values), 4),
        "min": round(min(coverage_values), 4),
    }
    if mode == "overfit-smoke":
        train_examples = train_examples[: min(8, len(train_examples))]
    batch_size = min(int(config["training"]["batch_size"]), len(train_examples))
    source, target = _batch(train_examples[:batch_size], torch)
    model.train()
    first_loss = float(_loss(model, source, target, torch).detach())
    initial_loss_max = float(config["training"].get("initial_loss_max", float("inf")))
    if not math.isfinite(first_loss) or first_loss > initial_loss_max:
        raise RuntimeError(
            f"pathological initial loss {first_loss:.4f} exceeds "
            f"configured maximum {initial_loss_max:.4f}"
        )

    if mode == "preflight":
        loss = _loss(model, source, target, torch)
        loss.backward()
        return {
            "status": "PREFLIGHT_PASS",
            "dataset_sha256": manifest["dataset_sha256"],
            "config_sha256": config_file_sha,
            "parameter_count": params,
            "batch_size": batch_size,
            "source_shape": list(source.shape),
            "target_shape": list(target.shape),
            "loss": round(float(loss.detach()), 6),
            "priority_evidence_coverage": coverage_summary,
            "effective_training_config": effective_training,
            "effective_config_sha256": effective_hash,
            "weights_written": False,
        }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    steps = 50 if mode == "overfit-smoke" else int(config["training"]["epochs"])
    losses: list[float] = []
    started = time.perf_counter()
    if mode == "overfit-smoke":
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, source, target, torch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        final_loss = losses[-1]
        status = "OVERFIT_PASS" if final_loss < first_loss * 0.95 else "OVERFIT_FAIL"
        return {
            "status": status,
            "dataset_sha256": manifest["dataset_sha256"],
            "config_sha256": config_file_sha,
            "parameter_count": params,
            "initial_loss": round(first_loss, 6),
            "final_loss": round(final_loss, 6),
            "steps": steps,
            "effective_training_config": effective_training,
            "effective_config_sha256": effective_hash,
            "weights_written": False,
        }

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    patience_limit = int(config["training"]["early_stopping_patience"])
    best_validation = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0
    epochs_completed = 0
    epoch_history: list[dict[str, Any]] = []
    rng = random.Random(seed)
    total_batches = 0
    for epoch in range(steps):
        epoch_start = time.perf_counter()
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        model.train()
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []
        for offset in range(0, len(order), batch_size):
            examples = [train_examples[index] for index in order[offset: offset + batch_size]]
            batch_source, batch_target = _batch(examples, torch)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch_source, batch_target, torch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            epoch_grad_norms.append(float(grad_norm))
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            total_batches += 1
        train_loss = sum(epoch_losses) / len(epoch_losses)
        mean_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        losses.append(train_loss)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for offset in range(0, len(validation_examples), batch_size):
                val_source, val_target = _batch(validation_examples[offset: offset + batch_size], torch)
                validation_losses.append(float(_loss(model, val_source, val_target, torch)))
        validation_loss = sum(validation_losses) / len(validation_losses)
        epochs_completed = epoch + 1
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - started

        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epochs_completed
            patience = 0
            improved = True
        else:
            patience += 1
            improved = False

        epoch_info: dict[str, Any] = {
            "epoch": epochs_completed,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(validation_loss, 6),
            "best_validation_loss": round(best_validation, 6),
            "best_epoch": best_epoch,
            "learning_rate": round(current_lr, 8),
            "grad_norm_mean": round(mean_grad_norm, 4),
            "elapsed_s": round(total_elapsed, 1),
            "epoch_s": round(epoch_elapsed, 1),
            "patience": patience,
            "improved": improved,
        }
        epoch_history.append(epoch_info)

        # Per-epoch progress line
        flag = "*" if improved else " "
        print(
            f"[{flag}] epoch {epochs_completed:>3d}/{steps} | "
            f"train {train_loss:.4f} | val {validation_loss:.4f} "
            f"(best {best_validation:.4f} @ {best_epoch}) | "
            f"lr {current_lr:.2e} | grad {mean_grad_norm:.2f} | "
            f"{total_elapsed:.0f}s"
        )

        if patience >= patience_limit:
            print(f"Early stopping at epoch {epochs_completed} (patience={patience_limit})")
            break

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    final_loss = losses[-1]
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=False)
    weights = output_dir / "model.pt"
    torch.save(model.state_dict(), weights)
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    result = {
        "status": "TRAINING_COMPLETE",
        "dataset_sha256": manifest["dataset_sha256"],
        "config_sha256": config_file_sha,
        "effective_config_sha256": effective_hash,
        "effective_training_config": effective_training,
        "parameter_count": params,
        "initial_loss": round(first_loss, 6),
        "final_loss": round(final_loss, 6),
        "best_validation_loss": round(best_validation, 6),
        "epochs_completed": epochs_completed,
        "validation_examples": len(validation_examples),
        "priority_evidence_coverage": coverage_summary,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "weights": {"path": str(weights), "sha256": weights_sha},
        "epoch_history": epoch_history,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Path to dataset manifest (required unless --list-presets)")
    parser.add_argument("--config", default="training/nexus_realizer_v1.json", type=Path)
    parser.add_argument("--mode", choices=("preflight", "overfit-smoke", "train"), default="preflight")
    parser.add_argument("--preset", default=None, help="Training intensity preset (smoke/quick/pilot/standard/full)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="write an immutable JSON report plus .sha256 sidecar",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs (for quick runs)")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    if args.list_presets:
        from stack.encoder.training_presets import list_presets, get_preset
        for name in list_presets():
            p = get_preset(name)
            note = p.pop("note", "")
            print(f"  {name:12s} epochs={p['epochs']:>3d}  patience={p.get('patience', '?'):>3d}")
            if note: print(f"               {note}")
        return 0

    if args.preset:
        from stack.encoder.training_presets import apply_preset
        cli = {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }
        cli = {k: v for k, v in cli.items() if v is not None}
        preset_params = apply_preset(args.preset, model_type="realizer", cli_overrides=cli)
        training_overrides = preset_params
        print(f"Preset '{args.preset}': {training_overrides}")
    else:
        training_overrides = {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }
    try:
        if args.manifest is None:
            raise ValueError("--manifest is required unless --list-presets is used")
        result = run(
            args.manifest, args.config, mode=args.mode,
            readiness_path=args.readiness, output_dir=args.output_dir,
            training_overrides=training_overrides,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2
    if args.report is not None:
        sidecar = args.report.with_suffix(args.report.suffix + ".sha256")
        if args.report.exists() or sidecar.exists():
            raise FileExistsError(f"refusing to overwrite: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        digest = hashlib.sha256(args.report.read_bytes()).hexdigest()
        sidecar.write_text(f"{digest}  {args.report.name}\n", encoding="ascii")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"].endswith("PASS") or result["status"] == "TRAINING_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
