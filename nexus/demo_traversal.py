"""
End-to-end traversal demo for NEXUS.

Demonstrates: question -> entity lookup -> graph traversal -> evidence -> answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus.graph.store import InMemoryGraphStore
from nexus.graph.traversal import beam_search, traverse_with_intent
from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR


def demo():
    """Run the traversal demo on the populated experiment graph."""
    
    # Step 1: Populate the graph
    print("=" * 60)
    print("NEXUS Traversal Demo")
    print("=" * 60)
    print()
    
    graph = InMemoryGraphStore()
    graph = populate_graph(EXPERIMENTS_DIR, graph)
    
    print(f"Graph: {graph.node_count} nodes, {graph.edge_count} edges")
    print()
    
    # Step 2: Define some test questions with known entities
    queries = [
        {
            "question": "What was the key finding of the chain-aware retrieval experiment?",
            "entities": ["Exp_0_11_ChainRetrieval"],
            "expected_path": ["Exp_0_11_ChainRetrieval"],
        },
        {
            "question": "What experiments led to the decision to pivot to NEXUS?",
            "entities": ["Decision_PivotToNEXUS"],
            "expected_path": ["Decision_PivotToNEXUS", "Concept_SelectorBottleneck"],
        },
        {
            "question": "What concept does the oracle memory experiment validate?",
            "entities": ["Exp_0_6_Validation"],
            "expected_path": ["Exp_0_6_Validation", "Concept_OracleMemory"],
        },
        {
            "question": "What experiment showed that the selector is the bottleneck?",
            "entities": ["Concept_SelectorBottleneck"],
            "expected_path": ["Concept_SelectorBottleneck", "Exp_0_12_Selection"],
        },
        {
            "question": "Show me the full experiment dependency chain.",
            "entities": ["Exp_0_13B_RealisticDistractors"],
            "expected_path": ["Exp_0_13B_RealisticDistractors", "Exp_0_13A_NoisyMemory"],
        },
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"--- Query {i} ---")
        print(f"Q: {query['question']}")
        print(f"Entry entities: {query['entities']}")
        
        # Step 3: Resolve entity names to node IDs
        entry_nodes = []
        for entity_name in query["entities"]:
            node_id = graph.find_entity(entity_name)
            if node_id:
                entry_nodes.append(node_id)
                node = graph.get_node(node_id)
                print(f"  Found: {node.id} ({node.type})")
            else:
                print(f"  NOT FOUND: {entity_name}")
        
        if not entry_nodes:
            print("  No entry nodes found — skipping traversal")
            print()
            continue
        
        # Step 4: Traverse the graph
        paths = beam_search(
            graph=graph,
            start_nodes=entry_nodes,
            query_entities=set(query["entities"]),
            max_depth=4,
            beam_width=5,
        )
        
        if not paths:
            print("  No paths found")
            print()
            continue
        
        # Step 5: Show results
        print(f"  Found {len(paths)} path(s):")
        for j, path in enumerate(paths[:3], 1):
            print(f"  Path {j} (score: {path.score:.3f}):")
            if path.steps:
                print(f"    {path.steps[0].from_node}")
                for step in path.steps:
                    direction = "<--" if step.reversed else "--"
                    print(f"      {direction}[{step.edge.type}] (conf: {step.edge.confidence:.2f})--> {step.to_node}")
        
        # Step 6: Build evidence from top path
        top_path = paths[0]
        if top_path.steps:
            evidence_nodes = []
            seen = set()
            for step in top_path.steps:
                if step.from_node not in seen:
                    node = graph.get_node(step.from_node)
                    if node:
                        evidence_nodes.append(node)
                        seen.add(step.from_node)
                if step.to_node not in seen:
                    node = graph.get_node(step.to_node)
                    if node:
                        evidence_nodes.append(node)
                        seen.add(step.to_node)
            
            print(f"  Evidence ({len(evidence_nodes)} nodes):")
            for node in evidence_nodes:
                desc = node.properties.get("description", "") or node.properties.get("key_finding", "") or node.properties.get("title", "")
                if desc:
                    print(f"    [{node.type}] {node.id}: {desc[:100]}")
        
        print()
    
    # Step 7: Summary
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
