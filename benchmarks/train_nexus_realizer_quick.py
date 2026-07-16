"""Quick training runner for NEXUS Realizer v1 — fast iteration with verbose logging.

Defaults to 5 epochs, batch-size 8, and detailed per-batch logging for
the first epoch.  Designed for rapid experimentation; not for production
training runs.

Usage:
    python benchmarks/train_nexus_realizer_quick.py --manifest <path>
    python benchmarks/train_nexus_realizer_quick.py --manifest <path> --epochs 3
    python benchmarks/train_nexus_realizer_quick.py --manifest <path> --verbose-batches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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
from benchmarks.train_nexus_realizer import (
    load_training_inputs,
    serialization_coverage,
    _encode,
    _batch,
    _assert_configured_output,
    _loss,
)
from nexus.realizer.model import build_model, parameter_count, validate_model_config
from nexus.realizer.tokenizer import ByteTokenizer


def quick_train(
    manifest_path: Path,
    config_path: Path,
    *,
    epochs: int = 5,
    batch_size: int | None = None,
    output_dir: Path | None = None,
    verbose_batches: bool = False,
) -> dict[str, Any]:
    """Run a quick training loop with verbose per-epoch and optional per-batch logging."""
    manifest, config, splits = load_training_inputs(manifest_path, config_path)

    # Override for quick runs
    config["training"]["epochs"] = epochs
    config["training"]["early_stopping_patience"] = min(
        config["training"]["early_stopping_patience"], max(epochs // 2, 1),
    )
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size

    if output_dir is not None:
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    steps = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    patience_limit = int(config["training"]["early_stopping_patience"])
    best_validation = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0
    epoch_history: list[dict[str, Any]] = []
    rng = random.Random(seed)
    started = time.perf_counter()
    total_batches = 0

    print(f"Quick train: {len(train_examples)} train, {len(validation_examples)} val, "
          f"{params} params, {epochs} epochs, batch_size={bs}")
    print(f"Initial loss: {first_loss:.4f}")
    print(f"Evidence coverage — mean: {coverage_summary['mean']:.2%}, min: {coverage_summary['min']:.2%}")
    print()

    for epoch in range(steps):
        epoch_start = time.perf_counter()
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        model.train()
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []

        batch_count = (len(order) + bs - 1) // bs
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

            if verbose_batches and ((offset // bs) % max(1, batch_count // 10) == 0):
                pct = (offset // bs + 1) / batch_count * 100
                print(f"  batch {offset // bs + 1:>4d}/{batch_count} ({pct:.0f}%) | "
                      f"loss {loss.detach():.4f} | grad {grad_norm:.2f}")

        train_loss = sum(epoch_losses) / len(epoch_losses)
        mean_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validation
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for offset in range(0, len(validation_examples), bs):
                val_source, val_target = _batch(validation_examples[offset: offset + bs], torch)
                validation_losses.append(float(_loss(model, val_source, val_target, torch)))
        validation_loss = sum(validation_losses) / len(validation_losses)
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - started

        improved = validation_loss < best_validation - 1e-6
        if improved:
            best_validation = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1

        epoch_info: dict[str, Any] = {
            "epoch": epoch + 1,
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

        flag = "*" if improved else " "
        loss_delta = ""
        if epoch > 0:
            prev_train = epoch_history[-2]["train_loss"]
            loss_delta = f" (d {train_loss - prev_train:+.4f})"
        print(
            f"[{flag}] epoch {epoch + 1:>3d}/{steps} | "
            f"train {train_loss:.4f}{loss_delta} | val {validation_loss:.4f} "
            f"(best {best_validation:.4f} @ {best_epoch}) | "
            f"lr {current_lr:.2e} | grad {mean_grad_norm:.2f} | "
            f"{total_elapsed:.0f}s"
        )

        if patience >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1} (patience={patience_limit})")
            break

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")

    model.load_state_dict(best_state)
    final_loss = epoch_history[-1]["train_loss"]
    result: dict[str, Any] = {
        "status": "QUICK_TRAIN_COMPLETE",
        "dataset_sha256": manifest["dataset_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "parameter_count": params,
        "initial_loss": round(first_loss, 6),
        "final_loss": round(final_loss, 6),
        "best_validation_loss": round(best_validation, 6),
        "best_epoch": best_epoch,
        "epochs_completed": len(epoch_history),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
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
    else:
        result["weights"] = None

    print(f"\nDone. Best val {best_validation:.4f} @ epoch {best_epoch}. "
          f"Total {result['elapsed_seconds']:.1f}s")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", default="training/nexus_realizer_v1.json", type=Path)
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", type=Path, default=None, help="External output dir")
    parser.add_argument("--verbose-batches", action="store_true", help="Log per-batch progress")
    args = parser.parse_args()

    try:
        result = quick_train(
            args.manifest, args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            verbose_batches=args.verbose_batches,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2

    if args.output_dir is None:
        # Print full JSON for programmatic consumption
        print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
