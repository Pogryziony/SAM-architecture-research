"""Evaluate trained Realizer checkpoints on the validation set."""
from __future__ import annotations

import json
import random
import sys
import time
import torch
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.train_nexus_realizer import load_training_inputs, serialize_source
from nexus.realizer.model import build_model
from nexus.realizer.tokenizer import ByteTokenizer
from nexus.realizer.decoder import DecoderConfig, decode_to_text, compute_repetition_rates


def evaluate_checkpoint(weights_path: Path, val_records: list, model_cfg: dict,
                        max_input: int, decoder_cfg: DecoderConfig,
                        sample_size: int = 100, seed: int = 42):
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    rng = random.Random(seed)
    shuffled = list(val_records)
    rng.shuffle(shuffled)
    sample = shuffled[:sample_size]

    tokenizer = ByteTokenizer()
    unique_texts = set()
    total_len = 0
    eos_count = 0
    reps = []
    examples = []

    started = time.perf_counter()
    with torch.no_grad():
        for i, record in enumerate(sample):
            source_ids = tokenizer.encode(
                serialize_source(record, max_input - 2), max_input
            )
            text, diag = decode_to_text(model, source_ids, tokenizer, decoder_cfg)
            unique_texts.add(text)
            total_len += diag.get("token_count", 0)
            if diag["eos_reached"]:
                eos_count += 1
            rep = compute_repetition_rates(tokenizer.encode(text, decoder_cfg.max_length))
            reps.append(rep.get("rep_3gram", 0))
            if i < 5:
                examples.append({
                    "question": record["question"][:80],
                    "ground_truth": record.get("answer", "")[:80],
                    "generated": text[:80]
                })

    elapsed = time.perf_counter() - started
    n = len(sample)
    text_counts = Counter()
    for rec in sample:
        source_ids = tokenizer.encode(
            serialize_source(rec, max_input - 2), max_input
        )
        text, _ = decode_to_text(model, source_ids, tokenizer, decoder_cfg)
        text_counts[text] += 1

    return {
        "samples": n,
        "unique_outputs": len(unique_texts),
        "uniqueness_ratio": len(unique_texts) / n,
        "eos_rate": eos_count / n,
        "avg_length": total_len / n,
        "avg_rep_3gram": sum(reps) / len(reps) if reps else 0.0,
        "latency_p50_ms": elapsed / n * 1000,
        "top_outputs": text_counts.most_common(5),
        "examples": examples,
    }


def main():
    manifest_path = Path("data/distillation/realizer_v1/manifest.json")
    config_path = Path("training/nexus_realizer_v1.json")
    manifest, config, splits = load_training_inputs(manifest_path, config_path)
    val_records = splits["validation"]
    model_cfg = config["model"]
    max_input = model_cfg["max_input_tokens"]

    decoder_cfg = DecoderConfig(
        strategy="greedy", repetition_penalty=1.2, no_repeat_ngram_size=3
    )

    run_dir = Path("models/realizer/run_20260716T100428Z")

    for ckpt_epoch in [1, 3]:
        weights_path = run_dir / f"checkpoint_epoch_{ckpt_epoch:03d}" / "model.pt"
        if not weights_path.is_file():
            print(f"Epoch {ckpt_epoch}: checkpoint not found")
            continue

        print(f"\n{'='*60}")
        print(f"EPOCH {ckpt_epoch} EVALUATION")
        print(f"{'='*60}")

        result = evaluate_checkpoint(
            weights_path, val_records, model_cfg, max_input, decoder_cfg,
            sample_size=100, seed=20260711 + ckpt_epoch,
        )

        print(f"Samples evaluated: {result['samples']}")
        print(f"Unique outputs: {result['unique_outputs']}/{result['samples']} "
              f"({result['uniqueness_ratio']:.1%})")
        print(f"EOS rate: {result['eos_rate']:.1%}")
        print(f"Avg length (tokens): {result['avg_length']:.1f}")
        print(f"Avg rep_3gram: {result['avg_rep_3gram']:.4f}")
        print(f"Latency p50: {result['latency_p50_ms']:.1f}ms")
        print(f"\nTop 5 most common outputs:")
        for text, count in result["top_outputs"]:
            print(f"  [{count}x] {repr(text[:60])}")
        print(f"\nSample generations:")
        for i, ex in enumerate(result["examples"]):
            print(f"  Q{i}: {ex['question']}")
            print(f"  GT{i}: {ex['ground_truth']}")
            print(f"  GEN{i}: {repr(ex['generated'])}")
            print()

        # Save results
        results_dir = Path("benchmarks/results/realizer/run_20260716T100428Z")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"eval_epoch_{ckpt_epoch:03d}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        import hashlib
        sidecar = out_path.with_suffix(out_path.suffix + ".sha256")
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        sidecar.write_text(f"{digest}  {out_path.name}\n", encoding="ascii")
        print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
