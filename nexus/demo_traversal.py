"""
End-to-end traversal demo for NEXUS.

Demonstrates two pipelines:
  1. question -> query parser -> graph traversal -> evidence builder
  2. question -> answer_question() -> full reasoning pipeline with model + verifier

The last query uses the complete NEXUS pipeline: parse → traverse → evidence
→ prompt → model → verify.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure UTF-8 output on Windows terminals with restricted code pages
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from nexus.graph.store import InMemoryGraphStore
from nexus.graph.traversal import traverse_with_intent
from nexus.query.parser import parse_question
from nexus.reasoning.evidence_builder import build_evidence
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import DummyModel
from nexus.reasoning.verifier import Verifier
from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR


def demo():
    """Run the traversal demo with both the evidence pipeline and full answer pipeline."""
    
    # Step 1: Populate the graph
    print("=" * 60)
    print("NEXUS Traversal Demo -- Parser -> Traversal -> Evidence")
    print("=" * 60)
    print()
    
    graph = InMemoryGraphStore()
    graph = populate_graph(EXPERIMENTS_DIR, graph)
    
    print(f"Graph: {graph.node_count} nodes, {graph.edge_count} edges")
    print()
    
    # Step 2: Test questions — first 4 use parser+traversal+evidence pipeline
    queries = [
        "What was the key finding of the chain-aware retrieval experiment?",
        "Why did the project pivot to NEXUS?",
        "What concept does the oracle memory experiment validate?",
        "What showed that the selector is the bottleneck?",
    ]
    
    for i, question in enumerate(queries, 1):
        print(f"--- Query {i} ---")
        print(f"Q: {question}")
        
        # Step 3: Parse the question — spot entities, detect intent
        parsed = parse_question(question, graph, cutoff=0.6)
        print(f"Intent: {parsed.intent} (direction={parsed.direction})")
        print(f"Entities found: {parsed.entity_ids}")
        
        if not parsed.entity_ids:
            print("  No entities found — skipping traversal")
            print()
            continue
        
        # Show matched entities with types
        for eid in parsed.entity_ids:
            node = graph.get_node(eid)
            if node:
                print(f"  -> {node.id} ({node.type})")
        
        # Step 4: Traverse — use intent-aware traversal
        query_entities = set(parsed.entity_ids)
        paths = traverse_with_intent(
            graph=graph,
            entry_nodes=parsed.entity_ids,
            query_entities=query_entities,
            intent=parsed.intent,
            max_depth=4,
            beam_width=5,
        )
        
        if not paths:
            print("  No paths found")
            print()
            continue
        
        print(f"  Found {len(paths)} path(s):")
        for j, path in enumerate(paths[:3], 1):
            print(f"  Path {j} (score: {path.score:.3f}):")
            if path.steps:
                print(f"    {path.steps[0].from_node}")
                for step in path.steps:
                    arrow = "<--" if step.reversed else "--"
                    print(f"      {arrow}[{step.edge.type}] (conf: {step.edge.confidence:.2f})--> {step.to_node}")
        
        # Step 5: Build structured evidence
        evidence_json = build_evidence(question, paths, graph, max_paths=3, question_intent=parsed.intent)
        print(f"\n  Evidence pack ({len(evidence_json)} chars, {len(paths)} paths):")
        # Print just the facts summary
        import json
        evidence = json.loads(evidence_json)
        # Print curated node facts first (highest confidence)
        node_facts = evidence.get("node_facts", [])
        if node_facts:
            print(f"  Node facts (curated):")
            for nf in node_facts:
                text = nf["text"]
                if len(text) > 150:
                    text = text[:147] + "..."
                print(f"    [HIGH] {text}")
        for fact in evidence.get("facts", []):
            print(f"    - {fact}")
        print(f"  Sources: {len(evidence.get('sources', []))} unique")
        print()
    
    # ── Query 5: Full answer_question pipeline ──
    print("=" * 60)
    print("Query 5 — Full NEXUS Pipeline (parse → traverse → evidence → prompt → model → verify)")
    print("=" * 60)
    print()
    
    question5 = "What does the realistic distractor experiment depend on?"
    print(f"Q: {question5}")
    print()
    
    model = DummyModel()
    verifier = Verifier(hallucination_threshold=0.2)
    
    result = answer_question(
        question5,
        graph,
        model=model,
        verifier=verifier,
        max_depth=4,
        beam_width=5,
        max_paths=5,
    )
    
    parsed = result["parsed_query"]
    if parsed:
        print(f"Intent: {parsed.intent}")
        print(f"Entities found: {parsed.entity_ids}")
        for eid in parsed.entity_ids:
            node = graph.get_node(eid)
            if node:
                print(f"  -> {node.id} ({node.type})")
    
    print(f"\nPaths found: {result['path_count']}")
    
    # Show the prompt (first 500 chars)
    ep = result["evidence_pack"]
    if ep:
        facts = ep.get("facts", [])
        print(f"Evidence facts: {len(facts)}")
        for fact in facts[:5]:
            print(f"  - {fact}")
    
    print(f"\n{'─' * 40}")
    print("Model Answer:")
    print(result["answer"])
    print(f"{'─' * 40}")
    
    v = result["verification"]
    if v:
        status = "PASS" if v.passed else "FAIL"
        print(f"\nVerification: {status}")
        print(f"  Supported claims: {v.supported_count}")
        print(f"  Unsupported claims: {len(v.unsupported_claims)}")
        print(f"  Hallucination rate: {v.hallucination_rate:.2f}")
        if v.unsupported_claims:
            print("  Unsupported claims:")
            for claim in v.unsupported_claims:
                print(f"    - {claim[:120]}{'...' if len(claim) > 120 else ''}")
    
    # ── Summary ──
    print()
    print("=" * 60)
    print("Graph Stats")
    print("=" * 60)
    print(f"Nodes: {graph.node_count}")
    print(f"Edges: {graph.edge_count}")
    stats = graph.stats()
    for node_type, count in sorted(stats["node_types"].items()):
        print(f"  {node_type}: {count}")


if __name__ == "__main__":
    demo()
