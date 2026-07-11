"""Diagnose the 12 questions with gold absent from candidate pool."""
import json
from pathlib import Path
from stack.encoder.trivial_baseline import candidate_pool

val = [json.loads(line) for line in Path("stack/encoder/data/val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
from benchmarks.run_benchmark import build_benchmark_graph
graph, _ = build_benchmark_graph()

print("=== Questions with absent gold ===")
for record in val:
    pool = candidate_pool(record["question"], graph)
    cand_ids = set(str(item["node_id"]) for item in pool)
    gold = set(str(e) for e in record.get("entities", []))
    missing = gold - cand_ids
    if missing:
        qid = record.get("id", "")
        question = record["question"]
        print(f"\nQ={qid}: '{question}'")
        print(f"  Gold: {gold}")
        print(f"  Missing from candidates: {missing}")
        print(f"  Pool size: {len(cand_ids)}")
        for eid in missing:
            node = graph.get_node(eid)
            if node:
                print(f"  Entity {eid}: type={node.type}, aliases={getattr(node, 'aliases', [])[:5]}")
                props = getattr(node, "properties", {}) or {}
                print(f"    display_name={props.get('display_name', '')}, key_finding={str(props.get('key_finding', ''))[:80]}")
