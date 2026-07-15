"""NEXUS Realizer v2 trainer — generation-aware validation and improved decoding.

Extends v1 training with per-epoch generation on a validation subset,
generation-quality-based checkpoint selection, and safety early stopping.

Usage:
    python benchmarks/train_nexus_realizer_v2.py --mode pilot --manifest <path> --epochs 3
    python benchmarks/train_nexus_realizer_v2.py --mode train --manifest <path> --readiness <path> --epochs 12
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
from typing import Any, Sequence

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from benchmarks.train_nexus_realizer import (
    load_training_inputs,
    serialization_coverage,
    _encode,
    _batch,
    _assert_external_output,
    _loss,
    validate_readiness_for_training,
)
from nexus.realizer.model import build_model, parameter_count, validate_model_config
from nexus.realizer.tokenizer import ByteTokenizer
from nexus.realizer.decoder import DecoderConfig, decode_to_text, compute_repetition_rates


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
    from benchmarks.train_nexus_realizer import serialize_source

    coherent = 0
    partial_count = 0
    repetitive = 0
    empty = 0
    eos_count = 0
    total_len = 0
    rep_3gram_sum = 0.0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, record in enumerate(val_records[:max_samples]):
            source_ids = tokenizer.encode(
                serialize_source(record, config["model"]["max_input_tokens"] - 2),
                config["model"]["max_input_tokens"],
            )
            text, diag = decode_to_text(model, source_ids, tokenizer, decoder_cfg)
            total += 1

            # Categorize
            words = text.split()
            if not text or len(words) < 2:
                empty += 1
            else:
                trigrams = [tuple(words[j:j + 3]) for j in range(len(words) - 2)]
                if trigrams and len(set(trigrams)) / len(trigrams) < 0.4:
                    repetitive += 1
                elif trigrams and len(set(trigrams)) / len(trigrams) < 0.7:
                    partial_count += 1
                else:
                    coherent += 1

            if diag["eos_reached"]:
                eos_count += 1
            total_len += diag.get("token_count", len(text))
            rep_3gram_sum += compute_repetition_rates(
                tokenizer.encode(text, decoder_cfg.max_length),
            ).get("rep_3gram", 0)

    n = max(total, 1)
    return {
        "coherent_rate": round(coherent / n, 4),
        "coherent": coherent,
        "partial": partial_count,
        "repetitive": repetitive,
        "empty": empty,
        "eos_rate": round(eos_count / n, 4),
        "avg_length": round(total_len / n, 1),
        "rep_3gram_mean": round(rep_3gram_sum / n, 4),
        "samples": total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# V2 training run
# ═══════════════════════════════════════════════════════════════════════════


def train_v2(
    manifest_path: Path,
    config_path: Path,
    *,
    mode: str = "pilot",
    readiness_path: Path | None = None,
    output_dir: Path | None = None,
    epoch_override: int | None = None,
    decoder_config: DecoderConfig | None = None,
    gen_val_samples: int = 20,
) -> dict[str, Any]:
    """Run generation-aware training.

    Args:
        mode: "pilot" (no readiness required) or "train" (requires readiness).
        epoch_override: Override config epochs.
        decoder_config: Decoder for generation-aware validation.
        gen_val_samples: Number of validation records for per-epoch generation.
    """
    manifest, config, splits = load_training_inputs(manifest_path, config_path)
    if epoch_override is not None and epoch_override > 0:
        config["training"]["epochs"] = epoch_override

    if mode == "train":
        if readiness_path is None:
            raise ValueError("--readiness is required for train mode")
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        errors = validate_readiness_for_training(readiness, manifest_path, config_path)
        if errors:
            raise ValueError("training blocked: " + "; ".join(errors))
    if output_dir is not None:
        _assert_external_output(output_dir)

    if decoder_config is None:
        decoder_config = DecoderConfig(
            strategy="greedy", repetition_penalty=1.2, no_repeat_ngram_size=3,
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
        serialization_coverage(record, config["model"]["max_input_tokens"] - 2)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    patience_limit = 3 if mode == "pilot" else int(config["training"]["early_stopping_patience"])
    best_val_loss = float("inf")
    best_gen_quality = -1.0
    best_state = None
    best_epoch = -1
    patience = 0
    gen_patience = 0
    epoch_history: list[dict[str, Any]] = []
    rng = random.Random(seed)
    started = time.perf_counter()
    total_batches = 0

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

        # Composite quality score: coherent_rate - rep_rate
        gen_quality = gen_metrics["coherent_rate"] - gen_metrics["rep_3gram_mean"]

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

        print(
            f"[{improved_flag}{gen_flag}] epoch {epoch + 1:>3d}/{steps} | "
            f"train {train_loss:.4f} | val {validation_loss:.4f} "
            f"(best {best_val_loss:.4f} @ {best_epoch}) | "
            f"gen coh {gen_metrics['coherent_rate']:.0%} "
            f"rep3 {gen_metrics['rep_3gram_mean']:.2f} "
            f"eos {gen_metrics['eos_rate']:.0%} | "
            f"{total_elapsed:.0f}s"
        )

        # Safety stops
        if not (math.isfinite(validation_loss) and math.isfinite(train_loss)):
            print("STOP: non-finite loss detected")
            break
        if gen_metrics["repetitive"] > gen_val_samples * 0.5:
            print(f"STOP: excessive repetition ({gen_metrics['repetitive']}/{gen_val_samples})")
            break
        if gen_patience >= 3:
            print(f"STOP: generation quality regressed for {gen_patience} epochs")
            break
        if patience >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)

    result: dict[str, Any] = {
        "status": "V2_TRAINING_COMPLETE" if mode == "train" else "V2_PILOT_COMPLETE",
        "dataset_sha256": manifest["dataset_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
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
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        weights = output_dir / "model.pt"
        torch.save(model.state_dict(), weights)
        weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        result["weights"] = {"path": str(weights), "sha256": weights_sha}
        (output_dir / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    from stack.encoder.training_presets import apply_preset, list_presets, get_preset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", default="training/nexus_realizer_v1.json", type=Path)
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
    parser.add_argument("--decoder-strategy", default="greedy", choices=("greedy", "beam", "sample"))
    parser.add_argument("--rep-penalty", type=float, default=1.2, help="Repetition penalty (>1.0)")
    parser.add_argument("--no-repeat-ngram", type=int, default=3, help="No-repeat n-gram size (0=off)")
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
        cli = {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size}
        cli = {k: v for k, v in cli.items() if v is not None}
        preset_params = apply_preset(args.preset, model_type="realizer", cli_overrides=cli)
        if args.epochs is None:
            args.epochs = preset_params["epochs"]
        if args.patience is None and "patience" in preset_params:
            args.patience = preset_params["patience"]
        if args.batch_size is None and "batch_size" in preset_params:
            args.batch_size = preset_params["batch_size"]
        print(f"Preset '{args.preset}': epochs={args.epochs}, patience={args.patience}, batch_size={args.batch_size}")

    dc = DecoderConfig(
        strategy=args.decoder_strategy,
        repetition_penalty=args.rep_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        beam_width=args.beam_width,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    try:
        result = train_v2(
            args.manifest, args.config,
            mode=args.mode,
            readiness_path=args.readiness,
            output_dir=args.output_dir,
            epoch_override=args.epochs,
            decoder_config=dc,
            gen_val_samples=args.gen_val_samples,
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
