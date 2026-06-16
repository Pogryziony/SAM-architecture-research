"""Inspect synthetic dataset examples for slot alignment, answer leakage, and task validity.

Usage:
    python -m sam.eval.inspect_dataset --data-dir data/synthetic --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List

from ..data.dataset import load_jsonl, Tokenizer


def inspect_dataset(data_dir: str, limit: int = 50):
    """Validate dataset integrity and print diagnostic information."""
    print(f"=== Dataset Inspection: {data_dir} ===\n")

    # Load metadata
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Metadata: seed={meta['seed']}, entity_sep={meta['entity_separation']}")
        print(f"Splits: train={meta['counts']['train']}, val={meta['counts']['val']}, test={meta['counts']['test']}")
        print(f"Slots: {meta['num_slots']}, Vocab: {meta['vocab_size']}")
        print()

    tokenizer = Tokenizer.from_dir(data_dir)
    kb = load_jsonl(os.path.join(data_dir, "kb.jsonl"))
    kb_by_id = {rec["slot_id"]: rec for rec in kb}

    issues = Counter()
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue

        examples = load_jsonl(path)
        print(f"--- {split} ({len(examples)} examples) ---")

        task_counts = Counter()
        hop_counts = Counter()
        slot_set = set()
        answer_set = set()

        for i, ex in enumerate(examples):
            task_counts[ex["task_type"]] += 1
            hop_counts[ex["reasoning_hops"]] += 1
            answer_set.add(ex["answer"])

            # Check slot alignment
            for slot_str in ex["required_slots"]:
                slot_id = int(slot_str.split("_")[1])
                slot_set.add(slot_id)
                if slot_id not in kb_by_id:
                    issues["missing_slot_in_kb"] += 1
                    if i < 3:
                        print(f"  WARNING: slot {slot_str} not in KB!")
                else:
                    kb_rec = kb_by_id[slot_id]
                    if kb_rec["embedding_target"] != ex["answer"]:
                        # The required slot's embedding target may differ from answer
                        # (multi-hop: only last fact's target = answer)
                        pass

            # Check question/answer leakage
            q_lower = ex["question"].lower()
            a_lower = ex["answer"].lower()
            if a_lower in q_lower:
                issues["answer_in_question"] += 1
                if i < 5:
                    print(f"  WARNING: answer '{ex['answer']}' appears in question!")

            # Check token coverage
            a_ids = tokenizer.encode(ex["answer"])
            if not a_ids or a_ids[0] == tokenizer.unk:
                issues["answer_token_missing"] += 1

            # Print detailed examples
            if i < limit:
                print(f"\n  [{i}] task={ex['task_type']} hops={ex['reasoning_hops']}")
                print(f"    Q: {ex['question']}")
                print(f"    A: '{ex['answer']}'")
                print(f"    Required slots: {ex['required_slots']}")
                for fo in ex["facts"]:
                    direct = " [addressable]" if fo["slot_id"] in ex.get("addressable_slots", []) else ""
                    print(f"    Fact {fo['slot_id']}: {fo['text']} (target={fo['embedding_target']}){direct}")
        print(f"\n  Task distribution: {dict(task_counts)}")
        print(f"  Hop distribution: {dict(hop_counts)}")
        print(f"  Unique slots referenced: {len(slot_set)}")
        print(f"  Unique answers: {len(answer_set)}")

    print(f"\n=== Issues found ===")
    for issue, count in issues.most_common():
        print(f"  {issue}: {count}")

    # KB statistics
    print(f"\n=== KB Statistics ===")
    print(f"  Total records: {len(kb)}")
    targets = Counter(rec["embedding_target"] for rec in kb)
    print(f"  Unique targets: {len(targets)}")
    print(f"  Top targets: {targets.most_common(5)}")

    # Check KB slot ID continuity
    slot_ids = sorted(rec["slot_id"] for rec in kb)
    if slot_ids != list(range(len(slot_ids))):
        print(f"  WARNING: Slot IDs are not contiguous 0..{len(slot_ids)-1}!")

    # Check fact/example slot_id format
    print(f"\n=== Slot ID Alignment ===")
    for split in ["train"]:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(path):
            examples = load_jsonl(path)
            ex = examples[0]
            print(f"  Example required_slots format: {ex['required_slots'][:3]}")
            print(f"  Example fact slot_id format: {ex['facts'][0]['slot_id']}")
    print(f"  KB slot_id format: {kb[0]['slot_id']} (type={type(kb[0]['slot_id']).__name__})")


def main():
    ap = argparse.ArgumentParser(description="Inspect synthetic dataset.")
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: {args.data_dir} not found. Run python -m sam.data.synthetic_facts first.")
        sys.exit(1)

    inspect_dataset(args.data_dir, args.limit)


if __name__ == "__main__":
    main()
