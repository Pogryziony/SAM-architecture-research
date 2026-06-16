"""Inspect retrieval split: dataset size, filtering, unique slots, examples-per-slot.

Usage:
    python -m sam.eval.inspect_retrieval_split --data-dir data/synthetic
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

from ..data.dataset import load_jsonl, Tokenizer


def inspect_retrieval_split(data_dir: str):
    tokenizer = Tokenizer.from_dir(data_dir)
    kb = load_jsonl(os.path.join(data_dir, "kb.jsonl"))
    kb_by_id = {rec["slot_id"]: rec for rec in kb}

    print(f"=== Retrieval Split Inspection: {data_dir} ===\n")
    print(f"KB records: {len(kb)}")
    print(f"KB slot IDs: {min(r['slot_id'] for r in kb)}..{max(r['slot_id'] for r in kb)}")

    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"\n{split}: FILE NOT FOUND")
            continue

        examples = load_jsonl(path)
        print(f"\n--- {split} ({len(examples)} examples) ---")

        # Check all required slots exist in KB
        missing = 0
        slot_counter = Counter()
        for ex in examples:
            for s in ex["required_slots"]:
                sid = int(s.split("_")[1])
                slot_counter[sid] += 1
                if sid not in kb_by_id:
                    missing += 1

        unique_slots = len(slot_counter)
        print(f"  Unique required slots: {unique_slots}")
        print(f"  Missing slots in KB: {missing}")
        print(f"  Examples per slot: min={min(slot_counter.values())}, "
              f"max={max(slot_counter.values())}, "
              f"mean={sum(slot_counter.values())/unique_slots:.1f}")
        print(f"  Required slots per example: min=1, max={max(len(ex['required_slots']) for ex in examples)}")

        # Task-type distribution
        task_counts = Counter(ex["task_type"] for ex in examples)
        print(f"  Task types: {dict(task_counts)}")

        # Check answer token coverage
        answer_tokens = set()
        unk_answers = 0
        for ex in examples:
            ids = tokenizer.encode(ex["answer"])
            if not ids or ids[0] == tokenizer.unk:
                unk_answers += 1
            else:
                answer_tokens.update(ids)
        print(f"  Unknown answer tokens: {unk_answers}/{len(examples)}")
        print(f"  Unique answer tokens: {len(answer_tokens)}")

        # First example detail
        ex0 = examples[0]
        print(f"\n  First example:")
        print(f"    question: {ex0['question'][:80]}...")
        print(f"    answer: {ex0['answer']}")
        print(f"    required_slots: {ex0['required_slots']}")
        print(f"    task_type: {ex0['task_type']}")
        print(f"    reasoning_hops: {ex0['reasoning_hops']}")

    print(f"\n=== Summary ===")
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(path):
            examples = load_jsonl(path)
            print(f"  {split}: {len(examples)} examples")


def main():
    ap = argparse.ArgumentParser(description="Inspect retrieval split.")
    ap.add_argument("--data-dir", default="data/synthetic")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: {args.data_dir} not found.")
        sys.exit(1)

    inspect_retrieval_split(args.data_dir)


if __name__ == "__main__":
    main()
