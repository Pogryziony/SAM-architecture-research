"""Full 1434-record evaluation of a trained Realizer checkpoint.

Reports raw neural metrics and grounded/hybrid metrics separately.
"""
import json
import hashlib
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from benchmarks.train_nexus_realizer import load_training_inputs, serialize_source_for_config
from nexus.realizer.model import build_model
from nexus.realizer.tokenizer import ByteTokenizer
from nexus.realizer.decoder import DecoderConfig, decode_to_text, compute_repetition_rates
from nexus.realizer.grounded import (
    realize_grounded, answer_similarity, token_f1,
    grounding_diagnostics, evidence_candidates,
)


def evaluate_full(weights_path: Path, run_id: str, epoch: int):
    manifest_path = Path("data/distillation/realizer_v1/manifest.json")
    config_path = Path("training/nexus_realizer_v2.json")
    manifest, config, splits = load_training_inputs(manifest_path, config_path)
    val_records = splits["validation"]
    total = len(val_records)

    model_cfg = config["model"]
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    tokenizer = ByteTokenizer()
    decoder_cfg = DecoderConfig(
        strategy="greedy", repetition_penalty=1.2,
        no_repeat_ngram_size=3, max_length=int(model_cfg["max_output_tokens"]),
    )
    max_input = model_cfg["max_input_tokens"]

    # Neural metrics
    neural_exact = 0; neural_similarity = 0.0; neural_token_f1_sum = 0.0
    neural_grounded = 0; neural_texts = Counter(); neural_unique = set()
    rep_3gram_sum = 0.0; rep_2gram_sum = 0.0; eos_count = 0
    empty_count = 0; total_len = 0
    grounding_scores = []; continuous_scores = []; grounding_failures = Counter()
    unsupported_number_count = 0; unsupported_identifier_count = 0
    unsupported_token_count = 0; unreadable_count = 0

    # Grounded/hybrid
    grounded_exact = 0; grounded_similarity = 0.0; grounded_token_f1 = 0.0
    grounded_outputs = set(); fallback_count = 0
    evidence_copy_count = 0; neural_used_count = 0

    per_example = []; latencies = []

    started = time.perf_counter()
    with torch.no_grad():
        for i, record in enumerate(val_records):
            t0 = time.perf_counter()
            source_ids = tokenizer.encode(
                serialize_source_for_config(record, config), max_input,
            )
            neural_text, diag = decode_to_text(model, source_ids, tokenizer, decoder_cfg)
            realization = realize_grounded(record, neural_text)
            grounded_text = realization.answer
            reference = str(record.get("answer", "")).strip()
            latencies.append((time.perf_counter() - t0) * 1000)

            # Neural
            ref_norm = " ".join(reference.casefold().split())
            neural_norm = " ".join(neural_text.casefold().split())
            neural_exact += (neural_norm == ref_norm)
            neural_similarity += answer_similarity(neural_text, reference)
            neural_token_f1_sum += token_f1(neural_text, reference)
            candidates = evidence_candidates(record)
            grounding = grounding_diagnostics(neural_text, candidates)
            neural_grounded += grounding.score >= 0.72
            grounding_scores.append(grounding.score)
            continuous_scores.append(grounding.continuous_support_score)
            if grounding.rejection_reason:
                grounding_failures[grounding.rejection_reason] += 1
            unsupported_number_count += bool(grounding.unsupported_numbers)
            unsupported_identifier_count += bool(grounding.unsupported_identifiers)
            unsupported_token_count += bool(grounding.unsupported_tokens)
            unreadable_count += not grounding.readable
            neural_texts[neural_text] += 1
            neural_unique.add(neural_text)
            rep_rates = compute_repetition_rates(
                tokenizer.encode(neural_text, decoder_cfg.max_length),
            )
            rep_3gram_sum += rep_rates.get("rep_3gram", 0)
            rep_2gram_sum += rep_rates.get("rep_2gram", 0)
            if diag["eos_reached"]:
                eos_count += 1
            if not neural_text or len(neural_text.split()) < 2:
                empty_count += 1
            total_len += diag.get("token_count", len(neural_text))

            # Grounded
            grd_norm = " ".join(grounded_text.casefold().split())
            grounded_exact += (grd_norm == ref_norm)
            grounded_similarity += answer_similarity(grounded_text, reference)
            grounded_token_f1 += token_f1(grounded_text, reference)
            grounded_outputs.add(grounded_text)
            fallback_count += realization.fallback_used
            if realization.strategy == "evidence_copy":
                evidence_copy_count += 1
            elif realization.strategy == "neural_grounded":
                neural_used_count += 1

            if i < 50:
                per_example.append({
                    "id": record.get("id"),
                    "question": record.get("question", "")[:120],
                    "reference": reference[:120],
                    "neural_prediction": neural_text[:120],
                    "grounded_answer": grounded_text[:120],
                    "fallback_used": realization.fallback_used,
                    "rejection_reason": realization.rejection_reason,
                    "neural_grounding": round(grounding.score, 4),
                    "grounding_diagnostics": grounding.to_dict(),
                })

    elapsed = time.perf_counter() - started
    n = max(total, 1)
    duplicates = sum(c - 1 for c in neural_texts.values())
    max_single_pct = max(neural_texts.values()) / n if neural_texts else 0
    top_outputs = neural_texts.most_common(10)
    sorted_latency = sorted(latencies)
    p50 = sorted_latency[len(sorted_latency) // 2]
    def distribution(values):
        ordered = sorted(values)
        if not ordered:
            return {key: 0.0 for key in ("min", "mean", "p10", "p25", "p50", "p75", "p90")}
        def q(fraction):
            return ordered[min(int((len(ordered) - 1) * fraction), len(ordered) - 1)]
        return {
            "min": round(ordered[0], 6),
            "mean": round(sum(ordered) / len(ordered), 6),
            "p10": round(q(0.10), 6), "p25": round(q(0.25), 6),
            "p50": round(q(0.50), 6), "p75": round(q(0.75), 6),
            "p90": round(q(0.90), 6),
        }

    result = {
        "schema_version": "nexus-realizer-checkpoint-eval-v2",
        "checkpoint": "epoch_" + str(epoch),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "source_tree_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], text=True,
        ).strip(),
        "dataset_sha256": manifest["dataset_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "label_use": "scoring_only",
        "samples": total,
        "raw_neural": {
            "exact_match_rate": round(neural_exact / n, 6),
            "token_f1_mean": round(neural_token_f1_sum / n, 6),
            "similarity_mean": round(neural_similarity / n, 6),
            "grounded_rate": round(neural_grounded / n, 6),
            "grounding_failure_rate": round(1.0 - neural_grounded / n, 6),
            "grounding_score_distribution": distribution(grounding_scores),
            "continuous_support_distribution": distribution(continuous_scores),
            "grounding_failure_reasons": dict(sorted(grounding_failures.items())),
            "unsupported_number_rate": round(unsupported_number_count / n, 6),
            "unsupported_identifier_rate": round(unsupported_identifier_count / n, 6),
            "unsupported_token_rate": round(unsupported_token_count / n, 6),
            "unreadable_rate": round(unreadable_count / n, 6),
            "uniqueness_ratio": round(len(neural_unique) / n, 6),
            "unique_outputs": len(neural_unique),
            "duplicates": duplicates,
            "max_single_output_pct": round(max_single_pct, 6),
            "eos_rate": round(eos_count / n, 6),
            "empty_output_rate": round(empty_count / n, 6),
            "rep_3gram_mean": round(rep_3gram_sum / n, 6),
            "rep_2gram_mean": round(rep_2gram_sum / n, 6),
            "avg_length": round(total_len / n, 1),
            "latency_p50_ms": round(p50, 6),
            "most_common_outputs": [
                [text[:80], count] for text, count in top_outputs
            ],
        },
        "grounded_hybrid": {
            "exact_match_rate": round(grounded_exact / n, 6),
            "token_f1_mean": round(grounded_token_f1 / n, 6),
            "similarity_mean": round(grounded_similarity / n, 6),
            "hallucination_rate": round(1.0 - grounded_exact / n, 6),
            "fallback_rate": round(fallback_count / n, 6),
            "evidence_copy_rate": round(evidence_copy_count / n, 6),
            "neural_used_rate": round(neural_used_count / n, 6),
            "unique_outputs": len(grounded_outputs),
            "uniqueness_ratio": round(len(grounded_outputs) / n, 6),
            "latency_p50_ms": round(p50, 6),
        },
        "per_example": per_example,
    }

    # Save
    out_dir = Path("benchmarks/results/realizer") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_epoch_{:03d}.json".format(epoch)
    sidecar = out_path.with_suffix(out_path.suffix + ".sha256")
    if out_path.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite: " + str(out_path))
    out_path.write_text(
        json.dumps(result, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    sidecar.write_text(digest + "  " + out_path.name + "\n", encoding="ascii")

    # Print summary
    print("FULL EVALUATION EPOCH {} ({} records)".format(epoch, total))
    print("=" * 60)
    print("RAW NEURAL:")
    for k, v in result["raw_neural"].items():
        if k != "most_common_outputs":
            print("  {}: {}".format(k, v))
    print("  Most common outputs:")
    for text, count in top_outputs[:5]:
        print("    [{}x] {}".format(count, repr(text[:60])))
    print("GROUNDED/HYBRID:")
    for k, v in result["grounded_hybrid"].items():
        print("  {}: {}".format(k, v))
    print("Saved: " + str(out_path))
    return result


if __name__ == "__main__":
    evaluate_full(
        Path("models/realizer/v2_20260716T131357Z/checkpoint_epoch_001/model.pt"),
        "v2_20260716T131357Z", 1,
    )
