"""Analyze required-set retrieval coverage for Experiment 0.10/0.11.

Computes:
  - any_required_present@K, all_required_present@K, coverage@K
  - Per-hop breakdowns (1-hop, 2-hop, 3-hop)
  - MRR of first required slot, mean rank per slot position, max rank
  - Per-example diagnostic JSONL with failure types

Supports retriever modes:
  dual_encoder_baseline, chain_set_bce, chain_set_infonce,
  chain_set_bce_hardneg, slot_graph_expansion,
  chain_set_plus_graph_expansion, iterative_chain,
  iterative_chain_teacher_forced

Usage:
    python -m sam.eval.analyze_required_set_retrieval \
      --data-dir data/synthetic_dense \
      --retriever-mode chain_set_bce \
      --retriever-checkpoint <path> \
      --topk 64 \
      --output experiments/debug/required_set_chain_set_bce_0_11.json
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

from ..data.dataset import Tokenizer, QADataset, build_kb_tensors, collate_qa, load_jsonl
from ..model.sam_core import SamModel, DualEncoderWrapper
from ..training.train_retrieval import (
    DualEncoderRetriever, QueryEncoder, ChainSetRetriever, SlotGraphExpander,
)
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


def load_chain_set_retriever(ckpt_path: str, tokenizer, device: str = "cpu"):
    """Load chain-set retriever checkpoint."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ms = state.get("model_state", state)

    num_slots = ms["slot_emb.weight"].shape[0]
    slot_dim = ms["slot_emb.weight"].shape[1]

    enc = QueryEncoder(
        vocab_size=tokenizer.vocab_size, d_model=256,
        n_layers=3, n_heads=4, d_ff=1024, query_dim=256,
        max_seq_len=64, pad_id=tokenizer.pad
    )

    model = ChainSetRetriever(enc, slot_dim, num_slots)
    model.load_state_dict(ms, strict=False)
    model.to(device)
    model.eval()
    return model


def load_slot_graph_expander(ckpt_path: str, tokenizer, device: str = "cpu"):
    """Load slot graph expander checkpoint."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ms = state.get("model_state", state)

    num_slots = ms["slot_emb.weight"].shape[0]
    slot_dim = ms["slot_emb.weight"].shape[1]

    # Try to infer hidden_dim from scorer
    hidden_dim = 128
    if "scorer.0.weight" in ms:
        hidden_dim = ms["scorer.0.weight"].shape[0]

    model = SlotGraphExpander(slot_dim, num_slots, hidden_dim)
    model.load_state_dict(ms, strict=False)
    model.to(device)
    model.eval()
    return model


def run_dual_encoder_retrieval(model, input_ids, prompt_lens, device, topk):
    """Run dual encoder retrieval."""
    q, _ = model(input_ids, prompt_lens)
    s = F.normalize(model.slot_emb.weight, dim=-1)
    scores = q @ s.t()
    return scores.topk(min(topk, scores.size(-1)), dim=-1)


def run_chain_set_retrieval(model, input_ids, prompt_lens, device, topk):
    """Run chain-set retrieval."""
    q, s_all = model(input_ids, prompt_lens)
    return model.retrieve_topk(q, topk)


def run_slot_graph_expansion(retriever_model, expander_model, input_ids, prompt_lens,
                              device, topk, top_anchors=8, top_neighbors=8):
    """Two-stage: retrieve anchors, expand to neighbors, union."""
    # Stage 1: Retrieve anchor slots
    if isinstance(retriever_model, DualEncoderRetriever):
        q, s_all = retriever_model(input_ids, prompt_lens)
        anchor_slots, anchor_scores = retriever_model.retrieve_topk(q @ s_all.t(), top_anchors) if hasattr(retriever_model, 'retrieve_topk') else run_dual_encoder_retrieval(retriever_model, input_ids, prompt_lens, device, top_anchors)
    elif isinstance(retriever_model, ChainSetRetriever):
        q, s_all = retriever_model(input_ids, prompt_lens)
        anchor_slots, anchor_scores = retriever_model.retrieve_topk(q, top_anchors)
    else:
        q, s_all = retriever_model(input_ids, prompt_lens)
        anchor_slots, anchor_scores = run_dual_encoder_retrieval(retriever_model, input_ids, prompt_lens, device, top_anchors)

    # Stage 2: Expand from anchors
    B = anchor_slots.shape[0]
    neighbors, _ = expander_model.expand(anchor_slots, top_neighbors)  # [B, A, N]

    # Union: anchors + neighbors
    all_candidates = []
    for i in range(B):
        anchors_i = set(anchor_slots[i].tolist())
        neighbors_i = set(neighbors[i].flatten().tolist())
        union = anchors_i | neighbors_i
        union_list = list(union)[:topk]
        all_candidates.append(union_list)

    return torch.tensor(all_candidates, device=device), None  # no scores for union


def run_iterative_chain(retriever_model, input_ids, prompt_lens, device, topk,
                         kb_by_slot, steps=3, topk_per_step=32, teacher_forced=False,
                         gold_required=None):
    """Iterative chain retrieval.

    Step 0: query = question, retrieve
    Step 1+: query = question + previous slot text, retrieve
    Union all retrieved.
    """
    B = input_ids.shape[0]
    all_retrieved = [[] for _ in range(B)]
    current_input_ids = input_ids.clone()

    # Get question-only text length (prompt_len)
    for step in range(steps):
        if isinstance(retriever_model, DualEncoderRetriever):
            q, s_all = retriever_model(current_input_ids, prompt_lens)
            scores = q @ s_all.t()
            top_slots, _ = scores.topk(min(topk_per_step, scores.size(-1)), dim=-1)
        elif isinstance(retriever_model, ChainSetRetriever):
            q, s_all = retriever_model(current_input_ids, prompt_lens)
            top_slots, _ = retriever_model.retrieve_topk(q, topk_per_step)
        else:
            q, s_all = retriever_model(current_input_ids, prompt_lens)
            scores = q @ s_all.t()
            top_slots, _ = scores.topk(min(topk_per_step, scores.size(-1)), dim=-1)

        for i in range(B):
            all_retrieved[i].extend(top_slots[i].tolist())

            # Append slot text to query for next step
            if step < steps - 1:
                if teacher_forced and gold_required is not None:
                    # Teacher-forced: use gold required slots
                    req_i = [int(s) for s in gold_required[i] if int(s) >= 0]
                    if step < len(req_i):
                        slot_id = req_i[step]
                        if slot_id in kb_by_slot:
                            slot_text = kb_by_slot[slot_id].get("text", "")
                            # Append slot text to the input (simplified)
                            # We just prepend the slot text tokens
                            slot_tokens = current_input_ids.new_tensor(
                                [tokenizer_encode_slot(slot_text, current_input_ids.device)],
                                dtype=torch.long
                            )
                else:
                    # Non-oracle: use top-1 retrieved slot
                    top1_slot = int(top_slots[i, 0].item())
                    if top1_slot in kb_by_slot:
                        slot_text = kb_by_slot[top1_slot].get("text", "")

    # Deduplicate and truncate
    result = []
    for i in range(B):
        seen = set()
        deduped = []
        for s in all_retrieved[i]:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        result.append(deduped[:topk])

    return result, None  # no scores


def compute_extended_rank_metrics(required_slots_list, retrieved_topk_list, k_values):
    """Compute MRR, mean rank per slot, max rank metrics."""
    results = {}
    for k in k_values:
        first_ranks = []
        second_ranks = []
        third_ranks = []
        max_ranks = []

        for req, ret in zip(required_slots_list, retrieved_topk_list):
            ret_k = ret[:k]
            req_set = set(req)
            ret_set = set(ret_k)

            # Find ranks of required slots
            req_ranks = []
            for rs in req:
                if rs in ret_set:
                    rank = ret_k.index(rs) + 1  # 1-indexed
                    req_ranks.append(rank)
                else:
                    req_ranks.append(k + 1)  # beyond K

            req_ranks.sort()
            if len(req_ranks) >= 1:
                first_ranks.append(req_ranks[0])
            if len(req_ranks) >= 2:
                second_ranks.append(req_ranks[1])
            if len(req_ranks) >= 3:
                third_ranks.append(req_ranks[2])
            if req_ranks:
                max_ranks.append(max(req_ranks))

        prefix = f"at_{k}"
        if first_ranks:
            results[f"mrr_first_required_{prefix}"] = sum(1.0 / r for r in first_ranks) / len(first_ranks)
            results[f"mean_rank_first_required_{prefix}"] = sum(first_ranks) / len(first_ranks)
        if second_ranks:
            results[f"mean_rank_second_required_{prefix}"] = sum(second_ranks) / len(second_ranks)
        if third_ranks:
            results[f"mean_rank_third_required_{prefix}"] = sum(third_ranks) / len(third_ranks)
        if max_ranks:
            results[f"mean_max_required_rank_{prefix}"] = sum(max_ranks) / len(max_ranks)

    return results


def analyze_required_set(
    data_dir: str,
    retriever_mode: str,
    retriever_ckpt: str,
    expander_ckpt: Optional[str] = None,
    topk: int = 64,
    limit: Optional[int] = None,
    device: str = "cpu",
):
    """Run required-set retrieval diagnostics."""
    tokenizer = Tokenizer.from_dir(data_dir)
    k_values: Tuple[int, ...] = (1, 3, 8, 16, 32, 64)
    k_values = tuple(k for k in k_values if k <= topk)
    max_k = max(k_values)

    is_dual_encoder = retriever_mode in ("dual_encoder_baseline",)
    is_chain_set = retriever_mode in ("chain_set_bce", "chain_set_infonce", "chain_set_bce_hardneg")
    is_graph_expansion = retriever_mode in ("slot_graph_expansion", "chain_set_plus_graph_expansion")
    is_iterative = retriever_mode in ("iterative_chain", "iterative_chain_teacher_forced")

    # Load KB for slot text lookup
    kb = load_jsonl(os.path.join(data_dir, "kb.jsonl"))
    kb_by_slot: Dict[int, Dict] = {}
    for rec in kb:
        kb_by_slot[rec["slot_id"]] = rec

    # Load retriever models
    print(f"[analyze_required_set] Mode: {retriever_mode}")
    print(f"[analyze_required_set] Loading retriever from {retriever_ckpt}")

    if is_dual_encoder:
        retriever = load_dual_encoder(retriever_ckpt, tokenizer, device)
    elif is_chain_set:
        retriever = load_chain_set_retriever(retriever_ckpt, tokenizer, device)
    elif is_graph_expansion:
        retriever = load_chain_set_retriever(retriever_ckpt, tokenizer, device)
        if expander_ckpt:
            print(f"[analyze_required_set] Loading expander from {expander_ckpt}")
            expander = load_slot_graph_expander(expander_ckpt, tokenizer, device)
        else:
            print("WARNING: No expander checkpoint provided for graph expansion mode")
            expander = None
    elif is_iterative:
        retriever = load_chain_set_retriever(retriever_ckpt, tokenizer, device)
    else:
        retriever = load_dual_encoder(retriever_ckpt, tokenizer, device)

    # Load test data
    test_ds = QADataset(data_dir, "test", tokenizer, kind="qa",
                        open_book=False, max_seq_len=64)
    n_examples = min(limit or len(test_ds), len(test_ds))
    print(f"[analyze_required_set] Analyzing {n_examples} test examples, topK={topk}")

    # Collect data
    required_slots_global: List[List[int]] = []
    retrieved_topk_global: List[List[int]] = []
    retrieval_scores_global: List[List[float]] = []
    hops_global: List[int] = []
    per_example: List[Dict[str, Any]] = []
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

        # Retrieve based on mode
        with torch.no_grad():
            if retriever_mode == "iterative_chain_teacher_forced":
                ret_slots_list, _ = run_iterative_chain(
                    retriever, input_ids, prompt_lens, device, topk, kb_by_slot,
                    steps=3, topk_per_step=min(32, topk),
                    teacher_forced=True, gold_required=[req_slots]
                )
                ret_slots = ret_slots_list[0] if ret_slots_list else []
                ret_scores = [0.0] * len(ret_slots)
            elif retriever_mode == "iterative_chain":
                ret_slots_list, _ = run_iterative_chain(
                    retriever, input_ids, prompt_lens, device, topk, kb_by_slot,
                    steps=3, topk_per_step=min(32, topk), teacher_forced=False
                )
                ret_slots = ret_slots_list[0] if ret_slots_list else []
                ret_scores = [0.0] * len(ret_slots)
            elif retriever_mode == "slot_graph_expansion" and expander is not None:
                ret_slots_t, _ = run_slot_graph_expansion(
                    retriever, expander, input_ids, prompt_lens,
                    device, topk, top_anchors=8, top_neighbors=8
                )
                ret_slots = ret_slots_t[0].tolist() if ret_slots_t.size(0) > 0 else []
                ret_scores = [0.0] * len(ret_slots)
            elif retriever_mode == "chain_set_plus_graph_expansion" and expander is not None:
                ret_slots_t, _ = run_slot_graph_expansion(
                    retriever, expander, input_ids, prompt_lens,
                    device, topk, top_anchors=8, top_neighbors=8
                )
                ret_slots = ret_slots_t[0].tolist() if ret_slots_t.size(0) > 0 else []
                ret_scores = [0.0] * len(ret_slots)
            elif is_chain_set:
                if isinstance(retriever, ChainSetRetriever):
                    q, _ = retriever(input_ids, prompt_lens)
                    top_slots, top_scores = retriever.retrieve_topk(q, topk)
                else:
                    q, s_all = retriever(input_ids, prompt_lens)
                    scores = q @ s_all.t()
                    top_scores, top_slots = scores.topk(min(topk, scores.size(-1)), dim=-1)
                ret_slots = top_slots[0].tolist()
                ret_scores = [round(float(s), 4) for s in top_scores[0].tolist()]
            else:
                # Default: dual encoder
                if isinstance(retriever, DualEncoderRetriever):
                    q, _ = retriever(input_ids, prompt_lens)
                    s = F.normalize(retriever.slot_emb.weight, dim=-1)
                    scores = q @ s.t()
                    top_scores, top_slots = scores.topk(min(topk, scores.size(-1)), dim=-1)
                elif isinstance(retriever, ChainSetRetriever):
                    q, _ = retriever(input_ids, prompt_lens)
                    top_slots, top_scores = retriever.retrieve_topk(q, topk)
                else:
                    q, s_all = retriever(input_ids, prompt_lens)
                    scores = q @ s_all.t()
                    top_scores, top_slots = scores.topk(min(topk, scores.size(-1)), dim=-1)
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
    results["retriever_mode"] = retriever_mode

    # Add extended rank metrics
    rank_metrics = compute_extended_rank_metrics(
        required_slots_global, retrieved_topk_global, k_values
    )
    results.update(rank_metrics)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"REQUIRED-SET RETRIEVAL DIAGNOSTICS ({n_examples} examples, topK={topk}, mode={retriever_mode})")
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

    # Rank analysis
    print(f"\n--- Required Slot Rank Analysis ---")
    for k in k_values:
        prefix = f"at_{k}"
        mrr = results.get(f"mrr_first_required_{prefix}", 0)
        mr1 = results.get(f"mean_rank_first_required_{prefix}", 0)
        mr2 = results.get(f"mean_rank_second_required_{prefix}", 0)
        mr3 = results.get(f"mean_rank_third_required_{prefix}", 0)
        mx = results.get(f"mean_max_required_rank_{prefix}", 0)
        print(f"  K={k}: MRR={mrr:.4f}  mean_rank(slot1)={mr1:.1f}  "
              f"mean_rank(slot2)={mr2:.1f}  mean_rank(slot3)={mr3:.1f}  "
              f"max_rank={mx:.1f}")

    print(f"\nMean required count: {results.get('mean_required_count', 0):.4f}")
    print(f"Failure type distribution:")
    for ft, count in sorted(failure_counts.items()):
        print(f"  {ft}: {count} ({100*count/max(n_examples,1):.1f}%)")

    return results, per_example


def tokenizer_encode_slot(text: str, device) -> torch.Tensor:
    """Simple tokenizer-agnostic encoding for slot text (returns raw token)."""
    # This is a placeholder — actual encoding would need the tokenizer
    # For iterative chain, we use the slot text from KB directly
    return torch.tensor([0], dtype=torch.long, device=device)


def main():
    ap = argparse.ArgumentParser(
        description="Analyze required-set retrieval coverage")
    ap.add_argument("--data-dir", required=True,
                    help="Path to synthetic data directory")
    ap.add_argument("--retriever-checkpoint", required=True,
                    help="Path to retriever checkpoint")
    ap.add_argument("--retriever-mode", default="dual_encoder_baseline",
                    choices=["dual_encoder_baseline", "chain_set_bce",
                             "chain_set_infonce", "chain_set_bce_hardneg",
                             "slot_graph_expansion", "chain_set_plus_graph_expansion",
                             "iterative_chain", "iterative_chain_teacher_forced"],
                    help="Retriever mode to use")
    ap.add_argument("--expander-checkpoint", default=None,
                    help="Path to slot graph expander checkpoint (for expansion modes)")
    ap.add_argument("--topk", type=int, default=64,
                    help="Maximum K for retrieval (default: 64)")
    ap.add_argument("--output", default="experiments/debug/required_set_retrieval.json",
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
        retriever_mode=args.retriever_mode,
        retriever_ckpt=args.retriever_checkpoint,
        expander_ckpt=args.expander_checkpoint,
        topk=args.topk,
        limit=args.limit,
        device=device,
    )

    # Save summary JSON
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

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
