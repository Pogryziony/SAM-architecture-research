"""Diagnose NEXUS Realizer decoding quality on a validation-derived diagnostic set.

Evaluates an existing checkpoint with a matrix of decoder configurations.
Reports teacher-forced and free-running metrics for every configuration.

Usage:
    python benchmarks/diagnose_realizer_decoding.py \
        --checkpoint <path> --manifest <path> --config training/nexus_realizer_v1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from nexus.realizer.model import build_model, parameter_count
from nexus.realizer.tokenizer import ByteTokenizer
from nexus.realizer.decoder import (
    DecoderConfig,
    decode_to_text,
    compute_repetition_rates,
)

# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic set builder
# ═══════════════════════════════════════════════════════════════════════════


def _load_diagnostic_set(
    manifest_path: Path, max_examples: int = 50,
) -> list[dict[str, Any]]:
    """Load a stratified sample from the validation split for diagnostics."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_path = manifest_path.parent / manifest["splits"]["validation"]["path"]
    records = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines() if line]
    return records[:max_examples]


def _encode_diagnostic(
    record: dict[str, Any], config: dict[str, Any],
) -> tuple[list[int], list[int], str]:
    """Encode a diagnostic record returning source ids, target ids, and answer text."""
    tokenizer = ByteTokenizer()
    from benchmarks.train_nexus_realizer import serialize_source

    source_ids = tokenizer.encode(
        serialize_source(record, config["model"]["max_input_tokens"] - 2),
        config["model"]["max_input_tokens"],
    )
    target_ids = tokenizer.encode(record["answer"], config["model"]["max_output_tokens"])
    return source_ids, target_ids, record["answer"]


# ═══════════════════════════════════════════════════════════════════════════
# Teacher-forced metrics
# ═══════════════════════════════════════════════════════════════════════════


def _teacher_forced_metrics(
    model: Any, source_batch: torch.Tensor, target_batch: torch.Tensor,
) -> dict[str, float]:
    """Compute teacher-forced loss, token accuracy, and sequence accuracy."""
    with torch.no_grad():
        logits = model(source_batch, target_batch[:, :-1])
        # Loss (same as training)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_batch[:, 1:].reshape(-1),
            ignore_index=0,
        ).item()

        # Token accuracy (excluding padding)
        predictions = logits.argmax(dim=-1)
        targets = target_batch[:, 1:]
        mask = targets != 0
        token_correct = (predictions == targets) & mask
        token_acc = token_correct.sum().item() / max(mask.sum().item(), 1)

        # Sequence accuracy (entire target must match)
        seq_correct = 0
        for b in range(target_batch.shape[0]):
            b_mask = mask[b]
            if b_mask.sum() == 0:
                continue
            if (predictions[b][b_mask] == targets[b][b_mask]).all():
                seq_correct += 1
        seq_acc = seq_correct / target_batch.shape[0]

    return {"tf_loss": round(loss, 4), "tf_token_acc": round(token_acc, 4), "tf_seq_acc": round(seq_acc, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# Free-running metrics
# ═══════════════════════════════════════════════════════════════════════════


def _categorize_output(text: str) -> str:
    """Categorize a generated text as coherent, partial, repetitive, or empty."""
    if not text or len(text) < 3:
        return "empty"
    words = text.split()
    if len(words) < 3:
        return "empty"
    # Check repetition
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if trigrams:
        unique_ratio = len(set(trigrams)) / len(trigrams)
        if unique_ratio < 0.4:
            return "repetitive"
        if unique_ratio < 0.7:
            return "partial"
    return "coherent"


def _evaluate_decoder(
    model: Any,
    records: list[dict[str, Any]],
    train_config: dict[str, Any],
    decoder_config: DecoderConfig,
    max_examples: int = 50,
) -> dict[str, Any]:
    """Evaluate a decoder configuration on the diagnostic set."""
    tokenizer = ByteTokenizer()
    results = {
        "decoder": {
            "strategy": decoder_config.strategy,
            "temperature": decoder_config.temperature,
            "top_k": decoder_config.top_k,
            "top_p": decoder_config.top_p,
            "beam_width": decoder_config.beam_width,
            "repetition_penalty": decoder_config.repetition_penalty,
            "no_repeat_ngram_size": decoder_config.no_repeat_ngram_size,
        },
        "outputs": [],
    }

    outputs = []
    eos_count = 0
    total_len = 0
    rep_2gram_sum = 0.0
    rep_3gram_sum = 0.0
    rep_4gram_sum = 0.0
    entropy_sum = 0.0
    categories = {"coherent": 0, "partial": 0, "repetitive": 0, "empty": 0}

    t0 = time.perf_counter()
    for i, record in enumerate(records[:max_examples]):
        source_ids, target_ids, gold_answer = _encode_diagnostic(record, train_config)

        text, diag = decode_to_text(model, source_ids, tokenizer, decoder_config)

        rep = compute_repetition_rates(
            tokenizer.encode(text, decoder_config.max_length),
        )
        cat = _categorize_output(text)
        categories[cat] += 1
        eos_count += 1 if diag["eos_reached"] else 0
        total_len += diag.get("token_count", len(text))
        rep_2gram_sum += rep.get("rep_2gram", 0)
        rep_3gram_sum += rep.get("rep_3gram", 0)
        rep_4gram_sum += rep.get("rep_4gram", 0)
        entropy_sum += diag.get("entropy_mean", 0)

        outputs.append({
            "index": i,
            "question": record["question"][:120],
            "gold": gold_answer[:200],
            "generated": text[:300],
            "category": cat,
            "eos_reached": diag["eos_reached"],
            "token_count": diag.get("token_count", 0),
            "rep_2gram": rep.get("rep_2gram", 0),
            "rep_3gram": rep.get("rep_3gram", 0),
        })

    n = max(len(outputs), 1)
    elapsed = time.perf_counter() - t0

    results["outputs"] = outputs
    results["metrics"] = {
        "eos_rate": round(eos_count / n, 4),
        "avg_length": round(total_len / n, 1),
        "rep_2gram_mean": round(rep_2gram_sum / n, 4),
        "rep_3gram_mean": round(rep_3gram_sum / n, 4),
        "rep_4gram_mean": round(rep_4gram_sum / n, 4),
        "entropy_mean": round(entropy_sum / n, 4),
        "coherent": categories["coherent"],
        "partial": categories["partial"],
        "repetitive": categories["repetitive"],
        "empty": categories["empty"],
        "coherent_rate": round(categories["coherent"] / n, 4),
        "elapsed_s": round(elapsed, 1),
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def _default_decoder_matrix() -> list[DecoderConfig]:
    """Recommended initial decoder configurations for the matrix test."""
    return [
        DecoderConfig(strategy="greedy"),
        DecoderConfig(strategy="beam", beam_width=3),
        DecoderConfig(strategy="beam", beam_width=5, length_penalty=0.6),
        DecoderConfig(strategy="greedy", repetition_penalty=1.15, no_repeat_ngram_size=3),
        DecoderConfig(strategy="greedy", repetition_penalty=1.2, no_repeat_ngram_size=3),
        DecoderConfig(strategy="beam", beam_width=3, repetition_penalty=1.15, no_repeat_ngram_size=3),
        DecoderConfig(strategy="sample", temperature=0.8, top_k=20, seed=42),
        DecoderConfig(strategy="sample", temperature=0.8, top_p=0.9, seed=42),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="model.pt path")
    parser.add_argument("--manifest", required=True, type=Path, help="distillation dataset manifest")
    parser.add_argument("--config", default="training/nexus_realizer_v1.json", type=Path)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Save full JSON results")
    args = parser.parse_args()

    train_config = json.loads(args.config.read_text(encoding="utf-8"))
    model = build_model(train_config["model"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    params = parameter_count(model)
    print(f"Model: {params} params, checkpoint: {args.checkpoint}")

    records = _load_diagnostic_set(args.manifest, args.max_examples)
    print(f"Diagnostic set: {len(records)} validation records")

    # Teacher-forced metrics on entire set
    tokenizer = ByteTokenizer()
    from benchmarks.train_nexus_realizer import serialize_source, _batch

    batch_source, batch_target = _batch([
        (
            tokenizer.encode(
                serialize_source(r, train_config["model"]["max_input_tokens"] - 2),
                train_config["model"]["max_input_tokens"],
            ),
            tokenizer.encode(r["answer"], train_config["model"]["max_output_tokens"]),
        )
        for r in records
    ], torch)
    tf = _teacher_forced_metrics(model, batch_source, batch_target)
    print(f"\nTeacher-forced: loss={tf['tf_loss']} token_acc={tf['tf_token_acc']} seq_acc={tf['tf_seq_acc']}")

    # Decoder matrix
    configs = _default_decoder_matrix()
    print(f"\nDecoder matrix: {len(configs)} configurations on {len(records)} examples\n")

    all_results = {
        "checkpoint": str(args.checkpoint),
        "model_params": params,
        "teacher_forced": tf,
        "diagnostic_examples": len(records),
        "configurations": [],
    }

    for ci, dc in enumerate(configs):
        label = (
            f"{dc.strategy}"
            + (f"_b{dc.beam_width}" if dc.strategy == "beam" and dc.beam_width > 1 else "")
            + (f"_lp{dc.length_penalty}" if dc.length_penalty else "")
            + (f"_rp{dc.repetition_penalty}" if dc.repetition_penalty > 1.0 else "")
            + (f"_ng{dc.no_repeat_ngram_size}" if dc.no_repeat_ngram_size else "")
            + (f"_t{dc.temperature}" if dc.temperature != 1.0 else "")
            + (f"_tk{dc.top_k}" if dc.top_k else "")
            + (f"_tp{dc.top_p}" if dc.top_p else "")
        )
        result = _evaluate_decoder(model, records, train_config, dc, args.max_examples)
        m = result["metrics"]
        print(
            f"[{ci+1:>2d}] {label:<45s} | "
            f"coh {m['coherent']:>2d}/{m['coherent']+m['partial']+m['repetitive']+m['empty']} "
            f"({m['coherent_rate']:.0%}) | "
            f"eos {m['eos_rate']:.0%} | "
            f"rep3 {m['rep_3gram_mean']:.2f} | "
            f"len {m['avg_length']:.0f} | "
            f"{m['elapsed_s']:.0f}s"
        )
        result["config_label"] = label
        all_results["configurations"].append(result)

    # Find best configuration by coherent rate
    best = max(all_results["configurations"], key=lambda c: c["metrics"]["coherent_rate"])
    print(f"\nBest: {best['config_label']} (coherent rate: {best['metrics']['coherent_rate']:.0%})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
