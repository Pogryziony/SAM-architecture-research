"""Deep diagnostic: why canonical mapping ceiling is so low."""
import json
from pathlib import Path
from collections import Counter

# Load data
val = [json.loads(line) for line in Path("stack/encoder/data/val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

from benchmarks.run_benchmark import build_benchmark_graph
graph, graph_meta = build_benchmark_graph()

from stack.encoder.canonical_mapping import build_canonical_mapping, apply_canonical_mapping, export_canonical_mapping_metadata
from stack.encoder.trivial_baseline import candidate_pool
from stack.encoder.train_ranker_v3 import build_evaluation_group

mapping = build_canonical_mapping(graph)
meta = export_canonical_mapping_metadata(mapping, graph)

print("=== Canonical targets (22 unique) ===")
targets = sorted(set(mapping.values()))
for t in targets:
    sources = [k for k, v in mapping.items() if v == t]
    print(f"  {t}: {len(sources)} sources, e.g. {sources[:3]}")

print("\n=== Gold entity diagnosis ===")
absent_gold = []
raw_present_canon_lost = []
canon_present = []

for record in val:
    qid = record.get("id", "")
    question = record["question"]
    entities = [str(e) for e in record.get("entities", [])]
    pool = candidate_pool(question, graph)
    cand_ids = [str(item["node_id"]) for item in pool]
    cand_set = set(cand_ids)
    mapped_cand_set = set(apply_canonical_mapping(cand_ids, mapping, top_k=max(1, len(cand_ids))))
    
    for eid in entities:
        if eid not in cand_set:
            absent_gold.append((qid, eid, question[:80]))
        elif eid not in mapped_cand_set:
            # Check what it maps to
            canonical = mapping.get(eid, eid)
            raw_present_canon_lost.append((qid, eid, canonical, question[:80]))
        else:
            canon_present.append((qid, eid, question[:80]))

print(f"\nAbsent from candidate pool: {len(absent_gold)}")
for qid, eid, q in absent_gold[:10]:
    print(f"  Q={qid}: gold={eid}, question='{q}'")
    # Check if this entity exists in the graph
    node = graph.get_node(eid)
    if node:
        print(f"    -> Node EXISTS in graph, type={node.type}")
        # Check why it's not in candidates
    else:
        print(f"    -> Node NOT FOUND in graph!")

print(f"\nPresent in raw but lost in canonical: {len(raw_present_canon_lost)}")
for qid, eid, canonical, q in raw_present_canon_lost[:10]:
    print(f"  Q={qid}: gold={eid} -> canonical={canonical}, question='{q}'")

print(f"\nPresent in canonical candidates: {len(canon_present)}")
print(f"\nTotal gold: {len(absent_gold) + len(raw_present_canon_lost) + len(canon_present)}")

# Check: which gold entities are NOT in the mapping at all?
print("\n=== Gold entities NOT in canonical mapping ===")
all_gold = set()
for record in val:
    for eid in record.get("entities", []):
        all_gold.add(str(eid))
unmapped_gold = [eid for eid in all_gold if eid not in mapping]
print(f"Unmapped gold: {len(unmapped_gold)}/{len(all_gold)}")
for eid in sorted(unmapped_gold)[:20]:
    node = graph.get_node(eid)
    ntype = getattr(node, "type", "?") if node else "MISSING"
    print(f"  {eid} (type={ntype})")
