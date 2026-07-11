"""Quick diagnostic script for Entity Ranker V3."""
import json
import time
from pathlib import Path

t0 = time.time()
# Load splits
train = [json.loads(line) for line in Path("stack/encoder/data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
val = [json.loads(line) for line in Path("stack/encoder/data/val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"Loaded train={len(train)} val={len(val)} in {time.time()-t0:.1f}s")

t0 = time.time()
from benchmarks.run_benchmark import build_benchmark_graph
graph, graph_meta = build_benchmark_graph()
print(f"Built graph: nodes={graph_meta.get('nodes','?')} edges={graph_meta.get('edges','?')} in {time.time()-t0:.1f}s")

t0 = time.time()
from stack.encoder.canonical_mapping import build_canonical_mapping, export_canonical_mapping_metadata
mapping = build_canonical_mapping(graph)
meta = export_canonical_mapping_metadata(mapping, graph)
print(f"Canonical mapping: {meta['mapped_nodes']} mapped, {meta['unique_canonical_targets']} targets in {time.time()-t0:.1f}s")

# Quick candidate ceiling
from stack.encoder.trivial_baseline import candidate_pool
from stack.encoder.train_ranker_v3 import build_evaluation_group

val_groups = []
absent_count = 0
absent_questions = 0
for record in val:
    pool = candidate_pool(record["question"], graph)
    group = build_evaluation_group(
        str(record.get("id", "")), str(record["question"]),
        [str(e) for e in record.get("entities", [])],
        [str(item["node_id"]) for item in pool],
        "validation", graph
    )
    if group is not None:
        val_groups.append(group)
    else:
        absent_questions += 1

total_gold = sum(len(set(g["positive_ids"])) for g in val_groups)
ceiling_hits = sum(len(set(g["positive_ids"]) & set(g["candidate_ids"])) for g in val_groups)
print(f"Validation: {len(val_groups)} groups, {total_gold} gold entities")
print(f"  Questions with absent gold: {absent_questions}")
print(f"  Raw candidate ceiling: {ceiling_hits}/{total_gold} = {ceiling_hits/total_gold:.4f}")

from stack.encoder.canonical_mapping import apply_canonical_mapping
canon_ceiling = 0
for g in val_groups:
    mapped = apply_canonical_mapping(list(g["candidate_ids"]), mapping, top_k=max(1, len(g["candidate_ids"])))
    canon_ceiling += len(set(g["positive_ids"]) & set(mapped))
print(f"  Canonical candidate ceiling: {canon_ceiling}/{total_gold} = {canon_ceiling/total_gold:.4f}")

# Per-question breakdown
from collections import Counter
categories = Counter()
for g in val_groups:
    pos = set(g["positive_ids"])
    cand = set(g["candidate_ids"])
    mapped_cand = set(apply_canonical_mapping(list(cand), mapping, top_k=max(1, len(cand))))
    
    if not (pos & cand):
        categories["gold_absent_from_candidates"] += 1
    elif not (pos & mapped_cand):
        categories["gold_present_but_canonical_lost"] += 1
    else:
        categories["gold_in_canonical_candidates"] += 1

print(f"\nPer-question category breakdown:")
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")

print(f"\nDiagnostic complete.")
