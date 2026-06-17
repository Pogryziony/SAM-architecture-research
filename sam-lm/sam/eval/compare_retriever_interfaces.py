"""Compare standalone dual encoder retrieval vs SAM external text query retrieval.

Usage:
    python -m sam.eval.compare_retriever_interfaces \
      --data-dir data/synthetic_dense \
      --retriever-checkpoint experiments/exp_0_6/retrieval_dual_encoder/checkpoint.pt \
      --sam-config configs/sam_retrieved_external_text_dense.yaml \
      --limit 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..data.dataset import Tokenizer, QADataset, build_kb_tensors, collate_qa
from ..model.sam_core import SamModel, DualEncoderWrapper
from ..training.train_retrieval import DualEncoderRetriever, QueryEncoder
from ..utils.config import load_config
from ..utils.seed import seed_everything
from torch.utils.data import DataLoader


def load_standalone_dual(
    ckpt_path: str, tokenizer, device: str = "cpu"
) -> DualEncoderRetriever:
    """Load standalone dual encoder from checkpoint."""
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


def load_sam_with_retriever(
    sam_config_path: str, retriever_ckpt_path: str, data_dir: str, device: str = "cpu"
) -> Tuple[SamModel, Tokenizer]:
    """Load SAM model with dual encoder retriever, same as train_sam does."""
    tokenizer = Tokenizer.from_dir(data_dir)
    cfg = load_config(sam_config_path)

    mem_cfg = cfg.model.get("memory", {})
    if hasattr(mem_cfg, 'to_dict'):
        mem_cfg = mem_cfg.to_dict()
    else:
        mem_cfg = dict(mem_cfg)

    model = SamModel(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.model.get("d_model", 384),
        n_layers=cfg.model.get("n_layers", 6),
        n_heads=cfg.model.get("n_heads", 6),
        d_ff=cfg.model.get("d_ff", 1536),
        dropout=cfg.model.get("dropout", 0.0),
        max_seq_len=cfg.model.get("max_seq_len", 128),
        memory_every=cfg.model.get("memory_every", 3),
        memory_query=cfg.model.get("memory_query", "tokenwise"),
        memory_integration=cfg.model.get("memory_integration", "gated_sum"),
        memory_cfg=mem_cfg,
        pad_id=tokenizer.pad,
    )

    n_subkeys = mem_cfg.get("num_subkeys", 64)
    slot_value_token, _ = build_kb_tensors(data_dir, n_subkeys * n_subkeys, tokenizer)

    retriever = DualEncoderWrapper(retriever_ckpt_path, tokenizer, device)
    model._retrieval_k = cfg.get("topK", 8)
    model.set_kb(slot_value_token, retriever=retriever)
    model.to(device)
    model.eval()

    return model, tokenizer


def compute_standalone_topk(
    standalone: DualEncoderRetriever,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    k: int = 32,
) -> Tuple[List[int], List[float], torch.Tensor, torch.Tensor]:
    """Get topK from standalone dual encoder."""
    with torch.no_grad():
        q, all_s = standalone(input_ids, prompt_lens)  # q: [1, D], all_s: [num_slots, D]
        scores = q @ all_s.t()  # [1, num_slots]
        sv, si = scores.topk(k, dim=-1)
        return si[0].tolist(), sv[0].tolist(), q, all_s


def compute_sam_topk(
    model: SamModel,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    k: int = 32,
) -> Tuple[List[int], List[float], torch.Tensor, torch.Tensor]:
    """Get topK from SAM external text query path."""
    with torch.no_grad():
        q_text = model._retriever.encode_text(input_ids, prompt_lens)
        s_frozen = F.normalize(model._slot_emb_frozen, dim=-1).to(q_text.device)
        scores = q_text @ s_frozen.t()
        sv, si = scores.topk(k, dim=-1)
        return si[0].tolist(), sv[0].tolist(), q_text, s_frozen


def compare_retrievers(
    data_dir: str,
    retriever_ckpt: str,
    sam_config: str,
    limit: int = 200,
    device: str = "cpu",
):
    """Compare standalone dual encoder vs SAM external text query retrieval."""
    tokenizer = Tokenizer.from_dir(data_dir)

    print(f"Loading standalone dual encoder from {retriever_ckpt}...")
    standalone = load_standalone_dual(retriever_ckpt, tokenizer, device)

    print(f"Loading SAM with retriever...")
    sam_model, _ = load_sam_with_retriever(sam_config, retriever_ckpt, data_dir, device)

    # Compute checkpoint and embedding hashes
    with open(retriever_ckpt, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()

    standalone_slot_hash = hashlib.sha256(
        standalone.slot_emb.weight.detach().cpu().numpy().tobytes()
    ).hexdigest()

    sam_slot_hash = hashlib.sha256(
        sam_model._slot_emb_frozen.detach().cpu().numpy().tobytes()
    ).hexdigest()

    print(f"  Standalone slot hash: {standalone_slot_hash[:16]}...")
    print(f"  SAM frozen slot hash: {sam_slot_hash[:16]}...")
    print(f"  Slot hashes match: {standalone_slot_hash == sam_slot_hash}")

    # Prepare data
    test_ds = QADataset(data_dir, "test", tokenizer, kind="qa",
                        open_book=False, max_seq_len=64)  # Match dual encoder seq len
    limit = min(limit, len(test_ds))

    results = []
    stats = {
        "total": 0,
        "identical_top8": 0,
        "identical_top32": 0,
        "standalone_recall_at_1": 0,
        "standalone_recall_at_8": 0,
        "standalone_recall_at_32": 0,
        "sam_recall_at_1": 0,
        "sam_recall_at_8": 0,
        "sam_recall_at_32": 0,
        "slot_hash_match": standalone_slot_hash == sam_slot_hash,
        "ckpt_hash": ckpt_hash,
        "standalone_slot_hash": standalone_slot_hash,
        "sam_slot_hash": sam_slot_hash,
        "question_text_mismatches": 0,
        "score_correlation": 0.0,
    }

    score_pairs = []

    for idx in range(limit):
        ex = test_ds[idx]
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        prompt_len = ex["prompt_len"]
        prompt_lens = torch.tensor([prompt_len], device=device)

        req_raw = ex["required_slots"]
        if isinstance(req_raw, torch.Tensor):
            req_slots = [int(s) for s in req_raw if int(s) >= 0]
        else:
            req_slots = [int(s) for s in req_raw if int(s) >= 0]

        if not req_slots:
            continue

        # Also encode with SAM's dataset to compare
        sam_ex = QADataset(data_dir, "test", tokenizer, kind="qa",
                           open_book=False, max_seq_len=128)
        sam_input_ids = sam_ex[idx]["input_ids"].unsqueeze(0).to(device)
        sam_prompt_lens = torch.tensor([sam_ex[idx]["prompt_len"]], device=device)

        # Standalone retrieval
        sa_top32, sa_scores, sa_q, sa_s = compute_standalone_topk(
            standalone, input_ids, prompt_lens, k=32)
        sa_top8 = sa_top32[:8]

        # SAM retrieval (same sequence length as standalone)
        sam_top32_same, sam_scores_same, sam_q, sam_s = compute_sam_topk(
            sam_model, input_ids, prompt_lens, k=32)
        sam_top8_same = sam_top32_same[:8]

        # SAM retrieval (SAM's own sequence length)
        sam_top32_own, sam_scores_own, sam_q2, sam_s2 = compute_sam_topk(
            sam_model, sam_input_ids, sam_prompt_lens, k=32)
        sam_top8_own = sam_top32_own[:8]

        # Check question text
        q_text_standalone = tokenizer.decode(
            input_ids[0, :prompt_len].tolist())
        q_text_sam = tokenizer.decode(
            sam_input_ids[0, :sam_ex[idx]["prompt_len"]].tolist())
        text_match = (q_text_standalone == q_text_sam)
        if not text_match:
            stats["question_text_mismatches"] += 1

        # Recall checks
        sa_hit_1 = any(s in sa_top32[:1] for s in req_slots)
        sa_hit_8 = any(s in sa_top8 for s in req_slots)
        sa_hit_32 = any(s in sa_top32 for s in req_slots)

        sam_hit_1 = any(s in sam_top8_same[:1] for s in req_slots)
        sam_hit_8 = any(s in sam_top8_same for s in req_slots)
        sam_hit_32 = any(s in sam_top32_same for s in req_slots)

        sam_hit_8_own = any(s in sam_top8_own for s in req_slots)
        sam_hit_32_own = any(s in sam_top32_own for s in req_slots)

        stats["total"] += 1
        stats["standalone_recall_at_1"] += int(sa_hit_1)
        stats["standalone_recall_at_8"] += int(sa_hit_8)
        stats["standalone_recall_at_32"] += int(sa_hit_32)
        stats["sam_recall_at_1"] += int(sam_hit_1)
        stats["sam_recall_at_8"] += int(sam_hit_8)
        stats["sam_recall_at_32"] += int(sam_hit_32)

        # TopK identity
        if sa_top8 == sam_top8_same:
            stats["identical_top8"] += 1
        if sa_top32 == sam_top32_same:
            stats["identical_top32"] += 1

        # Score correlation
        for i in range(min(32, len(sa_scores))):
            score_pairs.append((sa_scores[i], sam_scores_same[i]))

        result = {
            "idx": idx,
            "question_text_match": text_match,
            "prompt_len_standalone": prompt_len,
            "prompt_len_sam": sam_ex[idx]["prompt_len"],
            "required_slots": req_slots,
            "standalone_top8": sa_top8,
            "standalone_top32": sa_top32,
            "standalone_scores_top8": [round(float(s), 4) for s in sa_scores[:8]],
            "sam_top8_same_seq": sam_top8_same,
            "sam_top32_same_seq": sam_top32_same,
            "sam_scores_top8_same_seq": [round(float(s), 4) for s in sam_scores_same[:8]],
            "sam_top8_own_seq": sam_top8_own,
            "sam_top32_own_seq": sam_top32_own,
            "standalone_hit_8": sa_hit_8,
            "standalone_hit_32": sa_hit_32,
            "sam_hit_8": sam_hit_8,
            "sam_hit_32": sam_hit_32,
            "sam_own_hit_8": sam_hit_8_own,
            "sam_own_hit_32": sam_hit_32_own,
            "top8_identical": sa_top8 == sam_top8_same,
            "top32_identical": sa_top32 == sam_top32_same,
        }
        results.append(result)

    # Compute stats ratios
    total = max(stats["total"], 1)
    stats["standalone_recall_at_1"] /= total
    stats["standalone_recall_at_8"] /= total
    stats["standalone_recall_at_32"] /= total
    stats["sam_recall_at_1"] /= total
    stats["sam_recall_at_8"] /= total
    stats["sam_recall_at_32"] /= total
    stats["identical_top8_pct"] = stats["identical_top8"] / total * 100
    stats["identical_top32_pct"] = stats["identical_top32"] / total * 100
    stats["question_text_mismatch_pct"] = stats["question_text_mismatches"] / total * 100

    # Score correlation
    if score_pairs:
        x = torch.tensor([p[0] for p in score_pairs])
        y = torch.tensor([p[1] for p in score_pairs])
        stats["score_correlation"] = float(
            torch.corrcoef(torch.stack([x, y]))[0, 1].item()
        )

    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS ({total} examples)")
    print(f"{'='*60}")
    print(f"Slot hashes match: {stats['slot_hash_match']}")
    print(f"Question text mismatches: {stats['question_text_mismatches']}/{total} ({stats['question_text_mismatch_pct']:.1f}%)")
    print(f"Score correlation: {stats['score_correlation']:.6f}")
    print(f"")
    print(f"Top8 identical: {stats['identical_top8']}/{total} ({stats['identical_top8_pct']:.1f}%)")
    print(f"Top32 identical: {stats['identical_top32']}/{total} ({stats['identical_top32_pct']:.1f}%)")
    print(f"")
    print(f"Standalone Recall@1:  {stats['standalone_recall_at_1']:.4f}")
    print(f"Standalone Recall@8:  {stats['standalone_recall_at_8']:.4f}")
    print(f"Standalone Recall@32: {stats['standalone_recall_at_32']:.4f}")
    print(f"")
    print(f"SAM Recall@1 (same seq):  {stats['sam_recall_at_1']:.4f}")
    print(f"SAM Recall@8 (same seq):  {stats['sam_recall_at_8']:.4f}")
    print(f"SAM Recall@32 (same seq): {stats['sam_recall_at_32']:.4f}")

    # Save results
    os.makedirs("experiments/debug", exist_ok=True)
    with open("experiments/debug/retriever_interface_comparison_0_8.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    with open("experiments/debug/retriever_interface_comparison_0_8_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to experiments/debug/retriever_interface_comparison_0_8.*")

    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Compare standalone dual encoder vs SAM external text query retrieval")
    ap.add_argument("--data-dir", default="data/synthetic_dense")
    ap.add_argument("--retriever-checkpoint", required=True)
    ap.add_argument("--sam-config", default="configs/sam_retrieved_external_text_dense.yaml")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    compare_retrievers(
        data_dir=args.data_dir,
        retriever_ckpt=args.retriever_checkpoint,
        sam_config=args.sam_config,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
