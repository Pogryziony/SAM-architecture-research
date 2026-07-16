"""NEXUS Realizer v2 trainer — generation-aware validation, checkpointing, and evaluation.

Extends v1 training with per-epoch generation on a validation subset,
generation-quality-based checkpoint selection, safety early stopping,
and per-epoch checkpoint saving at configurable epochs (default: 1, 3, 5).

Usage:
    python benchmarks/train_nexus_realizer_v2.py --mode pilot --manifest <path> --epochs 3
    python benchmarks/train_nexus_realizer_v2.py --mode train --manifest <path> --readiness <path> --epochs 5 --output-dir models/realizer/run_001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Keep metadata-only CLI operations usable on machines without PyTorch.
if __name__ == "__main__" and "--list-presets" in sys.argv:
    from stack.encoder.training_presets import get_preset, list_presets

    for _name in list_presets():
        _preset = get_preset(_name)
        _note = _preset.pop("note", "")
        print(
            f"  {_name:12s} epochs={_preset['epochs']:>3d}  "
            f"patience={_preset.get('patience', '?'):>3d}"
        )
        if _note:
            print(f"               {_note}")
    raise SystemExit(0)

import torch

from benchmarks.train_nexus_realizer import (
    load_training_inputs,
    _encode,
    _batch,
    _assert_configured_output,
    _loss,
    apply_training_overrides,
    effective_config_sha256,
    validate_readiness_for_training,
    serialize_source_for_config,
    serialization_coverage_for_config,
    training_target_for_config,
)
from nexus.realizer.model import build_model, parameter_count, validate_model_config
from nexus.realizer.tokenizer import ByteTokenizer
from nexus.realizer.decoder import DecoderConfig, decode_to_text, compute_repetition_rates
from nexus.realizer.grounded import (
    answer_similarity,
    grounding_score,
    evidence_candidates,
    realize_grounded,
    token_f1,
)
from benchmarks.abstractive_realizer_contracts import (
    SLOT_NAMES, materialize_slot_template,
)

# Try to import psutil for memory tracking (optional)
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _peak_rss_mb() -> float | None:
    """Return current RSS in MB if psutil is available.

    The value is sampled after expensive phases and the maximum sample is
    tracked by the caller.  It is intentionally not labelled as an operating
    system peak because ``psutil`` does not expose that portably.
    """
    if not _HAS_PSUTIL:
        return None
    try:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _scheduler_total_epochs(config: dict[str, Any], run_epochs: int) -> int:
    """Keep short checkpoint runs on the preregistered LR schedule."""
    configured = int(config["training"].get("scheduler_total_epochs", run_epochs))
    return max(configured, run_epochs, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Generation-aware metrics on a validation subset
# ═══════════════════════════════════════════════════════════════════════════


def _generate_and_score(
    model: Any,
    val_records: list[dict[str, Any]],
    config: dict[str, Any],
    decoder_cfg: DecoderConfig,
    max_samples: int = 20,
) -> dict[str, Any]:
    """Generate answers and compute quality metrics on validation subset."""
    tokenizer = ByteTokenizer()
    readable = 0
    repetitive = 0
    empty = 0
    eos_count = 0
    total_len = 0
    rep_3gram_sum = 0.0
    total = 0
    neural_exact = 0
    final_exact = 0
    neural_similarity = 0.0
    final_similarity = 0.0
    neural_token_f1 = 0.0
    final_token_f1 = 0.0
    neural_grounded = 0
    fallback_count = 0
    neural_outputs: set[str] = set()
    final_outputs: set[str] = set()
    slot_placeholder_exact = 0
    relation_correct = 0
    slot_mode = config.get("data", {}).get("target_format") == "slot_template_v1"

    model.eval()
    with torch.no_grad():
        for i, record in enumerate(val_records[:max_samples]):
            source_ids = tokenizer.encode(
                serialize_source_for_config(record, config),
                config["model"]["max_input_tokens"],
            )
            text, diag = decode_to_text(model, source_ids, tokenizer, decoder_cfg)
            template_reference = training_target_for_config(record, config)
            if slot_mode:
                placeholders_ok = all(
                    text.count(f"[{name}]") == 1 for name in SLOT_NAMES
                )
                expected_relation = str(record.get("composition", {}).get("relation", ""))
                opposite = "different" if expected_relation == "the same" else "the same"
                relation_ok = bool(
                    expected_relation and expected_relation in text and opposite not in text
                )
                slot_placeholder_exact += placeholders_ok
                relation_correct += relation_ok
                final_text = materialize_slot_template(text, record.get("slots", {}))
                realization_fallback = False
                supported = placeholders_ok and relation_ok
            else:
                realization = realize_grounded(record, text)
                final_text = realization.answer
                realization_fallback = realization.fallback_used
                supported = grounding_score(
                    text, evidence_candidates(record)
                ) >= 0.72
            reference = str(record.get("answer", "")).strip()
            total += 1
            neural_outputs.add(text)
            final_outputs.add(final_text)
            neural_exact += " ".join(text.casefold().split()) == " ".join(template_reference.casefold().split())
            final_exact += " ".join(final_text.casefold().split()) == " ".join(reference.casefold().split())
            neural_similarity += answer_similarity(text, template_reference)
            final_similarity += answer_similarity(final_text, reference)
            neural_token_f1 += token_f1(text, template_reference)
            final_token_f1 += token_f1(final_text, reference)
            neural_grounded += supported
            fallback_count += realization_fallback

            # Categorize
            words = text.split()
            if not text or len(words) < 2:
                empty += 1
            else:
                trigrams = [tuple(words[j:j + 3]) for j in range(len(words) - 2)]
                if trigrams and len(set(trigrams)) / len(trigrams) < 0.4:
                    repetitive += 1
                else:
                    readable += 1

            if diag["eos_reached"]:
                eos_count += 1
            total_len += diag.get("token_count", len(text))
            rep_3gram_sum += compute_repetition_rates(
                tokenizer.encode(text, decoder_cfg.max_length),
            ).get("rep_3gram", 0)

    n = max(total, 1)
    return {
        "coherent_rate": round(readable / n, 4),
        "coherent": readable,
        "partial": 0,
        "repetitive": repetitive,
        "empty": empty,
        "eos_rate": round(eos_count / n, 4),
        "avg_length": round(total_len / n, 1),
        "rep_3gram_mean": round(rep_3gram_sum / n, 4),
        "neural_exact_match_rate": round(neural_exact / n, 4),
        "neural_similarity_mean": round(neural_similarity / n, 4),
        "neural_token_f1_mean": round(neural_token_f1 / n, 4),
        "neural_grounded_rate": round(neural_grounded / n, 4),
        "neural_unique_outputs": len(neural_outputs),
        "neural_uniqueness_ratio": round(len(neural_outputs) / n, 4),
        "answer_exact_match_rate": round(final_exact / n, 4),
        "answer_similarity_mean": round(final_similarity / n, 4),
        "answer_token_f1_mean": round(final_token_f1 / n, 4),
        "answer_unique_outputs": len(final_outputs),
        "fallback_rate": round(fallback_count / n, 4),
        "slot_placeholder_exact_rate": round(slot_placeholder_exact / n, 4),
        "relation_accuracy": round(relation_correct / n, 4),
        "samples": total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint persistence
# ═══════════════════════════════════════════════════════════════════════════


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: Path,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    learning_rate: float,
    gen_metrics: dict[str, Any],
    effective_config: dict[str, Any],
    effective_config_hash: str,
    dataset_sha256: str,
    config_sha256: str,
    parameter_count: int,
    elapsed_seconds: float,
    epoch_seconds: float,
    decoder_cfg: DecoderConfig,
    sample_predictions: list[dict[str, Any]] | None = None,
    stop_reason: str = "epoch_complete",
) -> dict[str, Any]:
    """Save a training checkpoint and record its SHA-256.

    Does NOT overwrite existing checkpoints — raises FileExistsError if the
    checkpoint directory already contains weights.
    """
    import torch as _torch

    ckpt_dir = output_dir / f"checkpoint_epoch_{epoch:03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weights_path = ckpt_dir / "model.pt"
    if weights_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {weights_path}")

    # Save model weights
    _torch.save(model.state_dict(), weights_path)
    weights_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    weights_size = weights_path.stat().st_size

    # Save optimizer and scheduler state
    opt_path = ckpt_dir / "optimizer.pt"
    _torch.save(optimizer.state_dict(), opt_path)
    opt_sha = hashlib.sha256(opt_path.read_bytes()).hexdigest()

    sched_path = ckpt_dir / "scheduler.pt"
    _torch.save(scheduler.state_dict(), sched_path)
    sched_sha = hashlib.sha256(sched_path.read_bytes()).hexdigest()

    # Build checkpoint manifest
    checkpoint_info: dict[str, Any] = {
        "schema_version": "nexus-realizer-checkpoint-v1",
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "validation_loss": round(validation_loss, 6),
        "learning_rate": round(learning_rate, 8),
        "gen_metrics": gen_metrics,
        "weights": {
            "path": weights_path.as_posix(),
            "sha256": weights_sha,
            "size_bytes": weights_size,
        },
        "optimizer_sha256": opt_sha,
        "scheduler_sha256": sched_sha,
        "optimizer": {
            "path": opt_path.as_posix(),
            "sha256": opt_sha,
            "stored_in_git": False,
        },
        "scheduler": {
            "path": sched_path.as_posix(),
            "sha256": sched_sha,
            "stored_in_git": False,
        },
        "effective_config_sha256": effective_config_hash,
        "effective_training_config": effective_config,
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "parameter_count": parameter_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "epoch_seconds": round(epoch_seconds, 3),
        "decoder_config": {
            "strategy": decoder_cfg.strategy,
            "repetition_penalty": decoder_cfg.repetition_penalty,
            "no_repeat_ngram_size": decoder_cfg.no_repeat_ngram_size,
            "max_length": decoder_cfg.max_length,
        },
        "stop_reason": stop_reason,
        "peak_rss_mb": _peak_rss_mb(),
    }

    if sample_predictions is not None:
        preds_path = ckpt_dir / "sample_predictions.json"
        preds_path.write_text(
            json.dumps(sample_predictions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_info["sample_predictions_sha256"] = hashlib.sha256(
            preds_path.read_bytes()
        ).hexdigest()
        checkpoint_info["sample_predictions_count"] = len(sample_predictions)

    # Save manifest with SHA-256 sidecar
    manifest_path = ckpt_dir / "manifest.json"

    # Add files metadata for existing test compatibility
    checkpoint_info["files"] = {
        "model.pt": {
            "sha256": weights_sha,
            "size": weights_size,
            "stored_in_git": False,
            "repository_eligible": True,
            "promotion_status": "unreviewed",
        }
    }

    manifest_path.write_text(
        json.dumps(checkpoint_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    # SHA-256 sidecar
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    sidecar.write_text(f"{manifest_sha}  {manifest_path.name}\n", encoding="ascii")

    checkpoint_info["manifest_sha256"] = manifest_sha
    return checkpoint_info


def _generate_sample_predictions(
    model: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    decoder_cfg: DecoderConfig,
    max_samples: int = 10,
) -> list[dict[str, Any]]:
    """Generate per-example predictions with evidence and ground truth."""
    tokenizer = ByteTokenizer()
    predictions: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for i, record in enumerate(records[:max_samples]):
            source_ids = tokenizer.encode(
                serialize_source_for_config(record, config),
                config["model"]["max_input_tokens"],
            )
            text, diag = decode_to_text(model, source_ids, tokenizer, decoder_cfg)
            slot_mode = config.get("data", {}).get("target_format") == "slot_template_v1"
            if slot_mode:
                final_answer = materialize_slot_template(text, record.get("slots", {}))
                realization_payload = {
                    "strategy": "neural_slot_template",
                    "answer": final_answer,
                    "fallback_used": False,
                    "slot_placeholders_complete": all(
                        text.count(f"[{name}]") == 1 for name in SLOT_NAMES
                    ),
                }
            else:
                realization = realize_grounded(record, text)
                final_answer = realization.answer
                realization_payload = realization.to_dict()
            rep_rates = compute_repetition_rates(
                tokenizer.encode(text, decoder_cfg.max_length)
            )
            predictions.append({
                "index": i,
                "id": record.get("id", f"sample_{i}"),
                "question": record["question"],
                "generated_text": text,
                "answer": final_answer,
                "realization": realization_payload,
                "ground_truth": record.get("answer", ""),
                "evidence_keys": list(record.get("evidence_pack", {}).keys()) if isinstance(record.get("evidence_pack"), dict) else [],
                "eos_reached": diag.get("eos_reached", False),
                "token_count": diag.get("token_count", 0),
                "rep_3gram": rep_rates.get("rep_3gram", 0),
                "rep_2gram": rep_rates.get("rep_2gram", 0),
            })
    return predictions


def train_v2(
    manifest_path: Path,
    config_path: Path,
    *,
    mode: str = "pilot",
    readiness_path: Path | None = None,
    output_dir: Path | None = None,
    epoch_override: int | None = None,
    training_overrides: dict[str, Any] | None = None,
    decoder_config: DecoderConfig | None = None,
    gen_val_samples: int = 20,
    checkpoint_epochs: list[int] | None = None,
    sample_predictions_count: int = 10,
) -> dict[str, Any]:
    """Run generation-aware training with per-epoch checkpointing.

    Args:
        mode: "pilot" (no readiness required) or "train" (requires readiness).
        epoch_override: Override config epochs.
        decoder_config: Decoder for generation-aware validation.
        gen_val_samples: Number of validation records for per-epoch generation.
        checkpoint_epochs: Epochs at which to save checkpoints (default: [1, 3, 5]).
        sample_predictions_count: Number of per-example predictions per checkpoint.
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [1, 3, 5]
    manifest, config, splits = load_training_inputs(manifest_path, config_path)
    if epoch_override is not None and epoch_override > 0:
        training_overrides = {**(training_overrides or {}), "epochs": epoch_override}
    apply_training_overrides(config, training_overrides)
    effective_training = dict(config["training"])
    effective_hash = effective_config_sha256(config)

    if mode == "train":
        if readiness_path is None:
            raise ValueError("--readiness is required for train mode")
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        errors = validate_readiness_for_training(readiness, manifest_path, config_path)
        if errors:
            raise ValueError("training blocked: " + "; ".join(errors))
    if output_dir is not None:
        _assert_configured_output(output_dir, config)

    if decoder_config is None:
        decoder_config = DecoderConfig(
            strategy="greedy", repetition_penalty=1.2, no_repeat_ngram_size=3,
            max_length=min(
                int(config["training"].get("generation_max_tokens", 128)),
                int(config["model"]["max_output_tokens"]),
            ),
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch unavailable") from exc

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = build_model(config["model"])
    params = parameter_count(model)

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

    bs = min(int(config["training"]["batch_size"]), len(train_examples))
    source, target = _batch(train_examples[:bs], torch)
    model.train()
    first_loss = float(_loss(model, source, target, torch).detach())
    initial_loss_max = float(config["training"].get("initial_loss_max", float("inf")))
    if not math.isfinite(first_loss) or first_loss > initial_loss_max:
        raise RuntimeError(
            f"pathological initial loss {first_loss:.4f} exceeds "
            f"configured maximum {initial_loss_max:.4f}"
        )

    # Validation records for generation-aware checks (same subset every epoch)
    rng_val = random.Random(seed + 1)
    val_records = list(splits["validation"])
    rng_val.shuffle(val_records)
    val_subset = val_records[:gen_val_samples]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    steps = max(int(config["training"]["epochs"]), 1)
    # Cap at 12 for v2 unless explicitly overridden
    if mode == "pilot" and steps > 12:
        steps = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_scheduler_total_epochs(config, steps),
    )

    patience_limit = int(config["training"]["early_stopping_patience"])
    best_val_loss = float("inf")
    best_gen_quality = -1.0
    best_state = None
    best_epoch = -1
    patience = 0
    gen_patience = 0
    epoch_history: list[dict[str, Any]] = []
    saved_checkpoints: list[dict[str, Any]] = []
    rng = random.Random(seed)
    started = time.perf_counter()
    total_batches = 0
    safety_stop_reason = ""
    # Filter checkpoint epochs to those within training range
    active_checkpoint_epochs = [e for e in checkpoint_epochs if 1 <= e <= steps]

    print(f"V2 {'pilot' if mode == 'pilot' else 'train'}: {len(train_examples)} train, "
          f"{len(val_records)} val, {params} params, {steps} epochs, bs={bs}")
    print(f"Decoder: {decoder_config.strategy}"
          + (f"_rp{decoder_config.repetition_penalty}" if decoder_config.repetition_penalty > 1 else "")
          + (f"_ng{decoder_config.no_repeat_ngram_size}" if decoder_config.no_repeat_ngram_size else ""))
    print(f"Initial loss: {first_loss:.4f}")
    print()

    for epoch in range(steps):
        epoch_start = time.perf_counter()
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        model.train()
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []

        for offset in range(0, len(order), bs):
            examples = [train_examples[index] for index in order[offset: offset + bs]]
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
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validation loss
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for offset in range(0, len(validation_examples), bs):
                val_source, val_target = _batch(validation_examples[offset: offset + bs], torch)
                validation_losses.append(float(_loss(model, val_source, val_target, torch)))
        validation_loss = sum(validation_losses) / len(validation_losses)

        # Generation-aware quality check
        gen_metrics = _generate_and_score(model, val_subset, config, decoder_config, gen_val_samples)

        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - started

        # Select neural checkpoints by label-aware validation quality.  The
        # grounded fallback is reported separately and cannot hide collapse.
        gen_quality = (
            0.45 * gen_metrics["neural_token_f1_mean"]
            + 0.25 * gen_metrics["neural_similarity_mean"]
            + 0.20 * gen_metrics["neural_grounded_rate"]
            + 0.10 * gen_metrics["neural_uniqueness_ratio"]
            - 0.20 * gen_metrics["rep_3gram_mean"]
        )

        val_improved = validation_loss < best_val_loss - 1e-6
        gen_improved = gen_quality > best_gen_quality + 0.01

        if val_improved:
            best_val_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1

        if gen_improved:
            best_gen_quality = gen_quality
            gen_patience = 0
        else:
            gen_patience += 1

        improved_flag = "*" if val_improved else " "
        gen_flag = "+" if gen_improved else " "

        epoch_info: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(validation_loss, 6),
            "best_validation_loss": round(best_val_loss, 6),
            "best_epoch": best_epoch,
            "learning_rate": round(current_lr, 8),
            "grad_norm_mean": round(mean_grad_norm, 4),
            "elapsed_s": round(total_elapsed, 1),
            "epoch_s": round(epoch_elapsed, 1),
            "patience": patience,
            "gen_patience": gen_patience,
            "val_improved": val_improved,
            "gen_improved": gen_improved,
            "gen_metrics": gen_metrics,
            "gen_quality": round(gen_quality, 4),
        }
        epoch_history.append(epoch_info)

        slot_gate_failures: list[str] = []
        if (
            config.get("data", {}).get("target_format") == "slot_template_v1"
            and epoch + 1 == 1
        ):
            gates = config.get("gates", {}).get("epoch_1_continue", {})
            if gen_metrics["answer_exact_match_rate"] < float(
                gates.get("materialized_exact_match_min", 0.0)
            ):
                slot_gate_failures.append("materialized_exact_match")
            if gen_metrics["slot_placeholder_exact_rate"] < float(
                gates.get("slot_placeholder_exact_min", 0.0)
            ):
                slot_gate_failures.append("slot_placeholder_exact")
            if gen_metrics["relation_accuracy"] < float(
                gates.get("relation_accuracy_min", 0.0)
            ):
                slot_gate_failures.append("relation_accuracy")
            epoch_info["epoch_1_continue_gate"] = {
                "passed": not slot_gate_failures,
                "failures": slot_gate_failures,
            }

        # Save checkpoint at designated epochs
        current_epoch = epoch + 1
        if output_dir is not None and current_epoch in active_checkpoint_epochs:
            sample_preds = _generate_sample_predictions(
                model, val_subset, config, decoder_config, sample_predictions_count,
            )
            stop_reason = (
                "epoch_1_quality_gate_failed" if slot_gate_failures
                else "early_stopping" if patience >= patience_limit
                else "gen_regression" if gen_patience >= 3
                else "epoch_complete"
            )
            ckpt_info = _save_checkpoint(
                model, optimizer, scheduler, output_dir, current_epoch,
                train_loss, validation_loss, current_lr,
                gen_metrics, effective_training, effective_hash,
                manifest["dataset_sha256"],
                hashlib.sha256(config_path.read_bytes()).hexdigest(),
                params, time.perf_counter() - started, epoch_elapsed,
                decoder_config, sample_preds, stop_reason,
            )
            saved_checkpoints.append(ckpt_info)
            print(f"    checkpoint saved: epoch {current_epoch} "
                  f"({ckpt_info['weights']['sha256'][:12]}...) "
                  f"size={ckpt_info['weights']['size_bytes']}B")

        print(
            f"[{improved_flag}{gen_flag}] epoch {epoch + 1:>3d}/{steps} | "
            f"train {train_loss:.4f} | val {validation_loss:.4f} "
            f"(best {best_val_loss:.4f} @ {best_epoch}) | "
            f"neural f1 {gen_metrics['neural_token_f1_mean']:.0%} "
            f"ground {gen_metrics['neural_grounded_rate']:.0%} "
            f"unique {gen_metrics['neural_uniqueness_ratio']:.0%} "
            f"fallback {gen_metrics['fallback_rate']:.0%} "
            f"rep3 {gen_metrics['rep_3gram_mean']:.2f} "
            f"eos {gen_metrics['eos_rate']:.0%} | "
            f"{total_elapsed:.0f}s"
        )

        # Safety stops
        current_rss = _peak_rss_mb()
        max_rss = float(config["training"].get("max_peak_rss_mb", float("inf")))
        if current_rss is not None and current_rss > max_rss:
            print(
                f"STOP: sampled RSS {current_rss:.1f} MB exceeds "
                f"configured budget {max_rss:.1f} MB"
            )
            safety_stop_reason = "memory_budget_exceeded"
            break
        if not (math.isfinite(validation_loss) and math.isfinite(train_loss)):
            print("STOP: non-finite loss detected")
            safety_stop_reason = "non_finite_loss"
            break
        if slot_gate_failures:
            safety_stop_reason = "epoch_1_quality_gate_failed"
            print("STOP: epoch 1 slot-quality gate failed: " + ", ".join(slot_gate_failures))
            break
        if gen_metrics["repetitive"] > gen_val_samples * 0.5:
            print(f"STOP: excessive repetition ({gen_metrics['repetitive']}/{gen_val_samples})")
            safety_stop_reason = "excessive_repetition"
            break
        if (
            epoch + 1 >= 2
            and gen_metrics["neural_uniqueness_ratio"] < 0.10
            and gen_metrics["neural_token_f1_mean"] < 0.10
        ):
            print("STOP: neural mode collapse detected by text-level metrics")
            safety_stop_reason = "mode_collapse"
            break
        if gen_patience >= 3:
            print(f"STOP: generation quality regressed for {gen_patience} epochs")
            safety_stop_reason = "generation_regression"
            break
        if patience >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1}")
            safety_stop_reason = "early_stopping"
            break

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)

    result: dict[str, Any] = {
        "status": (
            "ABSTRACTIVE_REALIZER_PILOT_STOPPED"
            if safety_stop_reason and config.get("data", {}).get("target_format") == "slot_template_v1"
            else "V2_TRAINING_COMPLETE" if mode == "train"
            else "V2_PILOT_COMPLETE"
        ),
        "safety_stop_reason": safety_stop_reason,
        "dataset_sha256": manifest["dataset_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "effective_config_sha256": effective_hash,
        "effective_training_config": effective_training,
        "parameter_count": params,
        "initial_loss": round(first_loss, 6),
        "final_loss": round(epoch_history[-1]["train_loss"], 6),
        "best_validation_loss": round(best_val_loss, 6),
        "best_epoch": best_epoch,
        "epochs_completed": len(epoch_history),
        "best_gen_quality": round(best_gen_quality, 4),
        "decoder_config": {
            "strategy": decoder_config.strategy,
            "repetition_penalty": decoder_config.repetition_penalty,
            "no_repeat_ngram_size": decoder_config.no_repeat_ngram_size,
        },
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "gen_val_samples": gen_val_samples,
        "priority_evidence_coverage": coverage_summary,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "epoch_history": epoch_history,
        "peak_rss_mb": _peak_rss_mb(),
        "checkpoint_epochs": active_checkpoint_epochs,
        "saved_checkpoints": saved_checkpoints,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save best model weights
        weights = output_dir / "model_best.pt"
        if not weights.exists():
            torch.save(best_state, weights)
        weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        result["best_weights"] = {
            "path": weights.as_posix(),
            "sha256": weights_sha,
            "epoch": best_epoch,
            "stored_in_git": False,
        }
        # Save run manifest
        run_manifest_path = output_dir / "run_manifest.json"
        run_manifest_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        run_manifest_sha = hashlib.sha256(run_manifest_path.read_bytes()).hexdigest()
        sidecar = run_manifest_path.with_suffix(run_manifest_path.suffix + ".sha256")
        sidecar.write_text(f"{run_manifest_sha}  {run_manifest_path.name}\n", encoding="ascii")
        result["run_manifest_sha256"] = run_manifest_sha

        # Also save effective config
        eff_config_path = output_dir / "effective_config.json"
        eff_config_path.write_text(
            json.dumps({
                "effective_training_config": effective_training,
                "effective_config_sha256": effective_hash,
                "decoder_config": result["decoder_config"],
                "model_config": config["model"],
                "dataset_sha256": manifest["dataset_sha256"],
                "config_sha256": result["config_sha256"],
                "seed": seed,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        eff_config_sha = hashlib.sha256(eff_config_path.read_bytes()).hexdigest()
        eff_sidecar = eff_config_path.with_suffix(eff_config_path.suffix + ".sha256")
        eff_sidecar.write_text(f"{eff_config_sha}  {eff_config_path.name}\n", encoding="ascii")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    from stack.encoder.training_presets import apply_preset, list_presets, get_preset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Path to dataset manifest (required unless --list-presets)")
    parser.add_argument("--config", default="training/nexus_realizer_v2.json", type=Path)
    parser.add_argument("--mode", choices=("pilot", "train"), default="pilot")
    parser.add_argument("--preset", default=None, help="Training intensity preset (smoke/quick/pilot/standard/full)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override preset/patience")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--gen-val-samples", type=int, default=20, help="Validation records for per-epoch generation")
    parser.add_argument("--checkpoint-epochs", type=str, default=None,
                        help="Comma-separated epochs to save checkpoints (default: 1,3,5)")
    parser.add_argument("--sample-predictions", type=int, default=10,
                        help="Number of per-example predictions per checkpoint")
    parser.add_argument("--decoder-strategy", default=None, choices=("greedy", "beam", "sample"))
    parser.add_argument("--rep-penalty", type=float, default=None, help="Repetition penalty (>1.0)")
    parser.add_argument("--no-repeat-ngram", type=int, default=None, help="No-repeat n-gram size (0=off)")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    if args.list_presets:
        for name in list_presets():
            p = get_preset(name)
            note = p.pop("note", "")
            print(f"  {name:12s} epochs={p['epochs']:>3d}  patience={p.get('patience', '?'):>3d}")
            if note:
                print(f"               {note}")
        return 0

    # Apply preset overrides
    if args.preset:
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

    decoder_file_config = json.loads(args.config.read_text(encoding="utf-8"))
    dc = DecoderConfig(
        strategy=args.decoder_strategy or training_overrides.get("decoder_strategy", "greedy"),
        repetition_penalty=(
            args.rep_penalty
            if args.rep_penalty is not None
            else float(training_overrides.get("rep_penalty", 1.2))
        ),
        no_repeat_ngram_size=(
            args.no_repeat_ngram
            if args.no_repeat_ngram is not None
            else int(training_overrides.get("no_repeat_ngram", 3))
        ),
        beam_width=args.beam_width,
        temperature=args.temperature,
        top_k=args.top_k,
        # Per-epoch generation remains bounded even when the training contract
        # retains a handful of much longer reference answers.  Final answers
        # are never truncated because Grounded Realizer can copy the complete
        # supported evidence sentence.
        max_length=min(
            int(decoder_file_config["training"].get(
                "generation_max_tokens", 128
            )),
            int(decoder_file_config["model"]["max_output_tokens"]),
        ),
    )

    try:
        if args.manifest is None:
            raise ValueError("--manifest is required unless --list-presets is used")
        ckpt_epochs = None
        if args.checkpoint_epochs is not None:
            ckpt_epochs = [int(e.strip()) for e in args.checkpoint_epochs.split(",") if e.strip()]
        result = train_v2(
            args.manifest, args.config,
            mode=args.mode,
            readiness_path=args.readiness,
            output_dir=args.output_dir,
            training_overrides=training_overrides,
            decoder_config=dc,
            gen_val_samples=args.gen_val_samples,
            checkpoint_epochs=ckpt_epochs,
            sample_predictions_count=args.sample_predictions,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2

    if args.output_dir is not None:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if "COMPLETE" in result["status"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
