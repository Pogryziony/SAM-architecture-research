"""Verify slot ID alignment between dataset and product-key memory.

Usage:
    python -m sam.eval.inspect_slots --data-dir data/synthetic --limit 50
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Set

import torch

from ..data.dataset import load_jsonl, Tokenizer, build_kb_tensors
from ..model.product_key_memory import ProductKeyMemory


def inspect_slots(data_dir: str, limit: int = 50):
    tokenizer = Tokenizer.from_dir(data_dir)
    kb = load_jsonl(os.path.join(data_dir, "kb.jsonl"))
    print(f"=== Slot Inspection: {data_dir} ===\n")

    # Build KB -> PKM mapping
    num_subkeys = 1024  # standard for 1M slots
    total_slots = num_subkeys * num_subkeys
    slot_value_token, num_live = build_kb_tensors(data_dir, total_slots, tokenizer)
    print(f"Total PKM slots: {total_slots}")
    print(f"Live slots (from KB): {num_live}")
    print(f"Dead slots: {total_slots - num_live}")
    print(f"Dead ratio: {(total_slots - num_live) / total_slots:.4f}")

    # Check slot ID continuity
    slot_ids = sorted(rec["slot_id"] for rec in kb)
    max_slot = max(slot_ids) if slot_ids else 0
    print(f"Max KB slot ID: {max_slot}")
    print(f"KB slot IDs contiguous 0..{len(slot_ids)-1}? {slot_ids == list(range(len(slot_ids)))}")

    # Check which slots have valid value tokens
    valid_slots = (slot_value_token >= 0).nonzero(as_tuple=True)[0]
    invalid_slots = (slot_value_token < 0).nonzero(as_tuple=True)[0]
    print(f"Slots with valid value tokens: {len(valid_slots)}")
    print(f"Slots with invalid value tokens: {len(invalid_slots)}")

    # Show first few slot mappings
    print("\nFirst 10 slot mappings:")
    for i in range(min(10, total_slots)):
        token = slot_value_token[i].item()
        token_str = tokenizer.inv.get(int(token), "?") if token >= 0 else "DEAD"
        print(f"  slot_{i:06d} -> token={token} ({token_str})")

    # Check dataset examples' required_slots exist in KB
    issues = []
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        examples = load_jsonl(path)
        print(f"\n--- {split}: checking {min(len(examples), limit)} examples ---")
        for i, ex in enumerate(examples[:limit]):
            for slot_str in ex["required_slots"]:
                slot_id = int(slot_str.split("_")[1])
                if slot_id >= total_slots:
                    issues.append(f"  {split}[{i}]: slot {slot_str} ({slot_id}) >= total_slots ({total_slots})")
                elif slot_value_token[slot_id] < 0:
                    issues.append(f"  {split}[{i}]: slot {slot_str} ({slot_id}) has invalid value token (-1)")
            for fo in ex["facts"]:
                fact_slot_id = int(fo["slot_id"].split("_")[1])
                if fact_slot_id != int(slot_str.split("_")[1]) and slot_str == fo["slot_id"]:
                    pass  # same slot appears in both fields

    if issues:
        print(f"\n!!! {len(issues)} SLOT ISSUES FOUND !!!")
        for issue in issues[:20]:
            print(issue)
    else:
        print("\nNo slot alignment issues found.")

    # Check retrieval behavior on a sample
    print("\n=== Retrieval sampling (random keys) ===")
    pkm = ProductKeyMemory(
        num_subkeys=min(64, num_subkeys),
        key_dim=32,
        value_dim=48,
        top_a=8,
        top_b=8,
        top_k=4,
    )
    # Initialize with random keys
    query = torch.randn(2, 64)
    slots, scores = pkm.retrieve_topk(query, k=8)
    print(f"Sample query -> retrieved slots: {slots[0].tolist()}")
    print(f"Scores: {scores[0].tolist()}")


def main():
    ap = argparse.ArgumentParser(description="Verify slot ID alignment.")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: {args.data_dir} not found.")
        sys.exit(1)

    inspect_slots(args.data_dir, args.limit)


if __name__ == "__main__":
    main()
