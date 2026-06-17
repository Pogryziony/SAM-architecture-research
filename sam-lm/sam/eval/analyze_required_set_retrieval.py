"""Analyze required-set retrieval coverage for Experiment 0.10.

Computes:
  - any_required_present@K, all_required_present@K, coverage@K
  - Per-hop breakdowns (1-hop, 2-hop, 3-hop)
  - Per-example diagnostic JSONL with failure types

Usage:
    python -m sam.eval.analyze_required_set_retrieval \
      --data-dir data/synthetic_dense \
      --retriever-checkpoint <path_to_best_dual_encoder> \
      --topk 64 \
      --output experiments/debug/required_set_retrieval_0_10.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..data.dataset import Tokenizer, QADataset, build_kb_tensors, collate_qa
from ..model.sam_core import SamModel, DualEncoderWrapper
from ..training.train_retrieval import DualEncoderRetriever, QueryEncoder
from ..utils.config import load_config
from ..utils.seed import seed_everything
from ..eval.metrics import compute_required_set_metrics


def _pick_device(cfg_device: str) -> str:
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def load_dual_encoder(ckpt_path: str, tokenizer, device: str = "cpu"):
    """Load standalone dual encoder for retrieval."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ms = state.get("model_state", state)

    enc = QueryEncoder(
        vocab_size=tokenizer.vocab_size, d_model=256,
        n_layers=3, n_heads=4, d_ff=1024, query_dim=256,
        max_seq_len=64, pad_id=tokenizer.pad
    )
    num_slots = ms["slot_emb.weight"].shape[0]
    enc.load_state_dict(
        {k.replace("encoder.", ""): v for k, v in ms.items() if k.startswith("encoder.")},
        strict=False
    )

    model = DualEncoderRetriever(enc, ms["slot_emb.weight"].shape[1], num_slots)
    model.load_state_dict(ms, strict=False)
    model.to(device)
    model.eval()
    return model


def analyze_required_set(
    data_dir: str,
    retriever_ckpt: str,
    topk: int = 64,
    limit: Optional[int] = None,
    device: str = "cpu",
    sam_config: Optional[str] = None,
):
    """Run required-set retrieval diagnostics."""
    tokenizer = Tokenizer.from_dir(data_dir)
    k_values: Tuple[int, ...] = (1, 3, 8, 16, 32, 64)
    k_values = tuple(k for k in k_values if k <= topk)
    max_k = max(k_values)

    # Load dual encoder
    print(f"[analyze_required_set] Loading dual encoder from {retriever_ckpt}")
    dual_model = load_dual_encoder(retriever_ckpt, tokenizer, device)
    slot_emb = dual_model.slot_emb.weight  # [num_slots, D]

    # Load test data
    test_ds = QADataset(data_dir, "test", tokenizer, kind="qa",
                        open_book=False, max_seq_len=64)
    n_examples = min(limit or len(test_ds), len(test_ds))
    print(f"[analyze_required_set] Analyzing {n_examples} test examples, topK={topk}")

    # Also load KB for fact text lookup
    from ..data.dataset import load_jsonl
    kb = load_jsonl(os.path.join(data_dir, "kb.jsonl"))
    kb_by_slot: Dict[int, Dict] = {}
    for rec in kb:
        kb_by_slot[rec["slot_id"]] = rec

    # Collect data
    required_slots_global: List[List[int]] = []
    retrieved_topk_global: List[List[int]] = []
    retrieval_scores_global: List[List[float]] = []
    hops_global: List[int] = []
    per_example: List[Dict[str, Any]] = []

    # Failure type counts
    failure_counts: Dict[str, int] = defaultdict(int)

    for idx in range(n_examples):
        ex = test_ds[idx]
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        prompt_len = ex["prompt_len"]
        prompt_lens = torch.tensor([prompt_len], device=device)

        req_raw = ex["required_slots"]
        if isinstance(req_raw, torch.Tensor):
            req_slots = [int(s) for s in req_raw if int(s) >= 0]
        else:
            req_slots = [int(s) for s in req_raw if int(s) >= 0]

        hops = int(ex["hops"])

        # Retrieve topK
        with torch.no_grad():
            q, _ = dual_model(input_ids, prompt_lens)
            scores_all = q @ slot_emb.t()  # [1, num_slots]
            top_scores, top_slots = scores_all.topk(min(topk, scores_all.size(-1)), dim=-1)

        ret_slots = top_slots[0].tolist()
        ret_scores = [round(float(s), 4) for s in top_scores[0].tolist()]

        required_slots_global.append(req_slots)
        retrieved_topk_global.append(ret_slots)
        retrieval_scores_global.append(ret_scores)
        hops_global.append(hops)

        # Build per-example detail
        req_set = set(req_slots)
        ret_set = set(ret_slots)
        retrieved_required = [s for s in ret_slots if s in req_set]
        missing_required = [s for s in req_slots if s not in ret_set]

        # Required facts
        required_facts = []
        for s in req_slots:
            if s in kb_by_slot:
                required_facts.append(kb_by_slot[s].get("text", ""))
            else:
                required_facts.append(f"<unknown slot {s}>")

        detail = {
            "question": tokenizer.decode(input_ids[0, :prompt_len].tolist()).strip(),
            "gold_answer": tokenizer.decode(ex.get("answer_ids", torch.tensor([])).tolist()).strip(),
            "reasoning_hops": hops,
            "task_type": ex.get("task_type", "unknown"),
            "required_slots": req_slots,
            "retrieved_topk": ret_slots,
            "retrieval_scores_topk": ret_scores,
            "required_facts": required_facts,
            "retrieved_required_slots": retrieved_required,
            "missing_required_slots": missing_required,
            "required_count": len(req_slots),
            "retrieved_required_count": len(retrieved_required),
            "missing_required_count": len(missing_required),
        }

        # Per-K breakdown
        for k in k_values:
            ret_k_set = set(ret_slots[:k])
            ret_req_k = [s for s in ret_slots[:k] if s in req_set]
            miss_k = [s for s in req_slots if s not in ret_k_set]
            n_retrieved_k = len(ret_req_k)

            detail[f"all_required_present_at_{k}"] = (
                len(req_slots) > 0 and n_retrieved_k == len(req_slots)
            )
            detail[f"any_required_present_at_{k}"] = n_retrieved_k > 0
            detail[f"required_slot_coverage_at_{k}"] = (
                n_retrieved_k / max(len(req_slots), 1)
            )
            detail[f"retrieved_required_at_{k}"] = ret_req_k
            detail[f"missing_required_at_{k}"] = miss_k

        # Determine failure type
        all_at_8 = detail.get(f"all_required_present_at_{8}", False)
        all_at_32 = detail.get(f"all_required_present_at_{32}", False)
        all_at_64 = detail.get(f"all_required_present_at_{64}", False)

        if len(req_slots) == 0:
            failure = "no_required_slots"
        elif all_at_8:
            failure = "none_all_present"
        elif all_at_32:
            failure = "ranked_too_low"
        elif all_at_64:
            failure = "ranked_beyond_64"
        else:
            if detail.get(f"any_required_present_at_{64}", False):
                failure = "missing_required_slot"
            else:
                failure = "no_required_in_topk"

        detail["failure_type"] = failure
        failure_counts[failure] += 1

        per_example.append(detail)

    # Compute aggregate metrics
    results = compute_required_set_metrics(
        required_slots_global, retrieved_topk_global, hops_global, k_values,
    )
    results["num_examples"] = n_examples
    results["topk"] = topk
    results["failure_type_counts"] = dict(failure_counts)
    results["data_dir"] = data_dir
    results["retriever_checkpoint"] = retriever_ckpt

    # Print summary table
    print(f"\n{'='*80}")
    print(f"REQUIRED-SET RETRIEVAL DIAGNOSTICS ({n_examples} examples, topK={topk})")
    print(f"{'='*80}")

    header = "K".ljust(6) + "any_req@K".ljust(14) + "all_req@K".ljust(14) + \
             "coverage@K".ljust(14) + "1-hop all@K".ljust(14) + \
             "2-hop all@K".ljust(14) + "3-hop all@K".ljust(14)
    print(header)
    print("-" * len(header))

    for k in k_values:
        prefix = f"at_{k}"
        any_r = results.get(f"any_required_present_{prefix}", 0)
        all_r = results.get(f"all_required_present_{prefix}", 0)
        cov = results.get(f"required_slot_coverage_{prefix}", 0)
        all_1 = results.get(f"all_required_single_hop_{prefix}", 0)
        all_2 = results.get(f"all_required_two_hop_{prefix}", 0)
        all_3 = results.get(f"all_required_three_hop_{prefix}", 0)
        row = (f"{k}".ljust(6) +
               f"{any_r:.4f}".ljust(14) +
               f"{all_r:.4f}".ljust(14) +
               f"{cov:.4f}".ljust(14) +
               f"{all_1:.4f}".ljust(14) +
               f"{all_2:.4f}".ljust(14) +
               f"{all_3:.4f}".ljust(14))
        print(row)

    print(f"\nMean required count: {results.get('mean_required_count', 0):.4f}")
    print(f"Failure type distribution:")
    for ft, count in sorted(failure_counts.items()):
        print(f"  {ft}: {count} ({100*count/max(n_examples,1):.1f}%)")

    return results, per_example


def main():
    ap = argparse.ArgumentParser(
        description="Analyze required-set retrieval coverage for Experiment 0.10")
    ap.add_argument("--data-dir", required=True,
                    help="Path to synthetic data directory")
    ap.add_argument("--retriever-checkpoint", required=True,
                    help="Path to dual encoder checkpoint")
    ap.add_argument("--topk", type=int, default=64,
                    help="Maximum K for retrieval (default: 64)")
    ap.add_argument("--output", default="experiments/debug/required_set_retrieval_0_10.json",
                    help="Output JSON summary path")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of examples to analyze")
    ap.add_argument("--device", default="auto",
                    help="Device to use (auto/cuda/cpu)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = _pick_device(args.device)

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data directory not found: {args.data_dir}")
        sys.exit(1)

    if not os.path.exists(args.retriever_checkpoint):
        print(f"ERROR: retriever checkpoint not found: {args.retriever_checkpoint}")
        sys.exit(1)

    results, per_example = analyze_required_set(
        data_dir=args.data_dir,
        retriever_ckpt=args.retriever_checkpoint,
        topk=args.topk,
        limit=args.limit,
        device=device,
    )

    # Save summary JSON
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Strip per_example from summary
    summary = {k: v for k, v in results.items() if k != "per_example"}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {args.output}")

    # Save per-example JSONL
    jsonl_path = args.output.replace(".json", "_misses.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in per_example:
            f.write(json.dumps(ex) + "\n")
    print(f"Per-example details saved to {jsonl_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
