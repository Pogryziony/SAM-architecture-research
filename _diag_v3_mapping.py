"""Diagnose why Decision_PivotToNEXUS maps to Concept_ArchitectureWorks."""
from benchmarks.run_benchmark import build_benchmark_graph
from stack.encoder.canonical_mapping import _parent_ids, _is_canonical_id, _find_canonical, build_canonical_mapping

graph, _ = build_benchmark_graph()

# Check Decision_PivotToNEXUS
node_id = "Decision_PivotToNEXUS"
node = graph.get_node(node_id)
print(f"Node: {node_id}")
print(f"  Type: {node.type if node else 'MISSING'}")
print(f"  Properties: {node.properties if node else 'N/A'}")

parents = _parent_ids(node_id, graph)
print(f"  Parent IDs: {parents}")
print(f"  Parent canonical check: {[(p, _is_canonical_id(p)) for p in parents]}")

canonical = _find_canonical(node_id, graph)
print(f"  Find canonical: {canonical}")

# Check outgoing edges
outgoing = graph.get_outgoing(node_id)
print(f"  Outgoing edges ({len(outgoing)}):")
for e in outgoing[:10]:
    target_node = graph.get_node(e.target)
    print(f"    -> {e.target} (type={e.type}, target_type={getattr(target_node, 'type', '?')})")

# Check incoming edges
incoming = graph.get_incoming(node_id)
print(f"  Incoming edges ({len(incoming)}):")
for e in incoming[:10]:
    print(f"    <- {e.source} (type={e.type})")

# Check mapping
mapping = build_canonical_mapping(graph)
print(f"\n  In mapping: {mapping.get(node_id, 'NOT IN MAPPING')}")

# Let's also check Concept_ArchitectureWorks
print(f"\nConcept_ArchitectureWorks:")
ca = graph.get_node("Concept_ArchitectureWorks")
if ca:
    print(f"  Type: {ca.type}")
    print(f"  Properties: {ca.properties}")
    print(f"  Outgoing: {[(e.target, e.type) for e in graph.get_outgoing('Concept_ArchitectureWorks')[:5]]}")

# Check what maps TO Decision_PivotToNEXUS (children)
children = [k for k, v in mapping.items() if v == "Decision_PivotToNEXUS"]
print(f"\n  Nodes mapping TO Decision_PivotToNEXUS: {children}")

# Check Property_Child-like nodes
for nid in sorted(graph._nodes.keys()):
    node = graph.get_node(nid)
    props = getattr(node, "properties", {}) or {}
    if isinstance(props, dict):
        parent = str(props.get("parent_entity", props.get("parent_id", "")))
        if parent == "Decision_PivotToNEXUS":
            print(f"\n  Property child of Decision_PivotToNEXUS: {nid} (type={node.type})")
