"""Per-question failure ledger for Entity Ranker V3 validation."""
import json
from pathlib import Path
from collections import Counter, defaultdict

val = [json.loads(line) for line in Path("stack/encoder/data/val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

from benchmarks.run_benchmark import build_benchmark_graph
graph, _ = build_benchmark_graph()

from stack.encoder.canonical_mapping import build_canonical_mapping, apply_canonical_mapping
from stack.encoder.trivial_baseline import candidate_pool
from stack.encoder.train_ranker_v3 import build_evaluation_group

mapping = build_canonical_mapping(graph)

# Load the trained model
artifacts = sorted(Path("models/encoder").glob("entity_ranker_v3_*"))
if artifacts:
    model_dir = str(artifacts[-1])
    print(f"Using model: {model_dir}")
else:
    print("No trained model found!")
    exit(1)

from stack.encoder.entity_ranker_v3 import load_ranker_v3
model, tokenizer, config = load_ranker_v3(model_dir)
model.eval()

from stack.encoder.entity_text import build_entity_text

# Build groups
val_groups = []
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

categories = Counter()
stage_ceilings = {"graph": 0, "candidate_pool": 0, "canonical": 0, "ranker": 0}
total_gold = sum(len(set(g["positive_ids"])) for g in val_groups)

import torch

for g in val_groups:
    pos = set(g["positive_ids"])
    cand = set(g["candidate_ids"])
    
    # Graph ceiling: gold exists in graph?
    graph_ok = all(graph.get_node(eid) is not None for eid in pos)
    for eid in pos:
        if graph.get_node(eid) is not None:
            stage_ceilings["graph"] += 1
    
    # Candidate pool ceiling: gold in raw candidates?
    for eid in pos:
        if eid in cand:
            stage_ceilings["candidate_pool"] += 1
    
    # Canonical ceiling: gold in canonical-mapped candidates?
    mapped_cand = set(apply_canonical_mapping(list(cand), mapping, top_k=max(1, len(cand))))
    for eid in pos:
        if eid in mapped_cand:
            stage_ceilings["canonical"] += 1
    
    # Ranker: predict and check
    offsets, indices = tokenizer.tokenize_batch([g["question"]])
    q_offsets = torch.tensor(offsets[:-1], dtype=torch.long)
    q_indices = torch.tensor(indices, dtype=torch.long)
    cand_texts = [build_entity_text(cid, graph) for cid in g["candidate_ids"]]
    
    with torch.no_grad():
        scores = model(q_indices, q_offsets, cand_texts, tokenizer)
    ranked = [g["candidate_ids"][i] for i in torch.argsort(scores[0], descending=True).tolist()]
    canon_ranked = apply_canonical_mapping(ranked, mapping, top_k=10)
    
    for eid in pos:
        if eid in set(canon_ranked):
            stage_ceilings["ranker"] += 1
    
    # Categorize per question
    gold_in_cand = pos & cand
    gold_in_mapped = pos & mapped_cand
    gold_in_ranked = pos & set(canon_ranked)
    
    if not gold_in_cand:
        categories["gold_absent_from_candidate_pool"] += 1
    elif not gold_in_mapped:
        categories["gold_absent_from_canonical_mapping"] += 1
    elif not gold_in_ranked:
        # Check if gold is present but ranked below 10
        if pos & set(apply_canonical_mapping(ranked, mapping, top_k=max(1, len(ranked)))):
            categories["gold_ranked_below_10"] += 1
        else:
            categories["gold_lost_in_canonicalization"] += 1
    else:
        categories["success"] += 1

print(f"\n=== Per-question failure categories ({len(val_groups)} questions) ===")
for cat, cnt in sorted(categories.items()):
    print(f"  {cat}: {cnt}")

print(f"\n=== Stage ceilings (out of {total_gold} gold entities) ===")
for stage, hits in stage_ceilings.items():
    print(f"  {stage}: {hits}/{total_gold} = {hits/total_gold:.4f}")
