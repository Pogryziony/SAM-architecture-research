"""
NEXUS ingestion pipeline -- walks project docs and experiments,
extracts entities and relations, and populates the knowledge graph.

Usage:
    python -m nexus.ingestion.ingest_docs
    python -m nexus.ingestion.ingest_docs --dirs sam-lm/docs sam-lm/experiments
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph import Node, Edge
from nexus.graph.store import InMemoryGraphStore
from nexus.ingestion.entity_extractor import extract_from_markdown
from nexus.ingestion.relation_extractor import extract_relations
from nexus.ingestion.normalizer import canonicalize
from nexus.ingestion.deduplicator import merge_entity_lists


def _slugify(name: str) -> str:
    """Create a normalized ID from an entity name using the canonicalizer."""
    return canonicalize(name)


def _make_node(entity: dict) -> Node:
    """Convert an entity dict from the extractor to a Node."""
    name = entity["name"]
    node_id = _slugify(name)
    return Node(
        id=node_id,
        type=entity.get("type", "Entity"),
        properties={
            "name": name,
            "display_name": name,
        },
        sources=[entity.get("source", "")],
    )


def _supplement_entities(text: str, source_path: str) -> list[dict]:
    """
    Extract additional entities that the base entity_extractor misses.
    
    Handles experiment references, model names, metrics, config files,
    gates, and bold-emphasized concepts.
    """
    entities = []
    seen = set()

    def add(name: str, etype: str, line: int = 0):
        key = (name, etype)
        if key not in seen:
            seen.add(key)
            entities.append({
                "name": name, "type": etype,
                "source": source_path, "line": line,
            })

    # Experiment references: "Experiment 0.X", "Experiment 0.XY"
    for m in re.finditer(
        r'(?:Experiment|Exp)[_\s]*([\d]+(?:\.[\d]+[A-Z]?)?)',
        text, re.IGNORECASE
    ):
        line = text[:m.start()].count('\n') + 1
        add(f"Experiment {m.group(1)}", "Experiment", line)

    # SAM running modes
    for pattern, etype in [
        (r'\b(core_only)\b', "Experiment"),
        (r'\b(oracle_memory)\b', "Experiment"),
        (r'\b(retrieved_memory)\b', "Experiment"),
        (r'\b(random_memory)\b', "Experiment"),
        (r'\b(oracle_text_memory)\b', "Experiment"),
        (r'\b(retrieved_memory_external_text_query)\b', "Experiment"),
        (r'\b(oracle_filter)\b', "Concept"),
        (r'\b(learned_selector)\b', "Concept"),
        (r'\b(chain_set)\b', "Concept"),
        (r'\b(dual_encoder)\b', "Concept"),
        (r'\b(product_key_memory)\b', "Concept"),
        (r'\b(gated_integration)\b', "Concept"),
        (r'\b(gated_sum)\b', "Concept"),
        (r'\b(InfoNCE)\b', "Concept"),
        (r'\b(BCE loss)\b', "Concept"),
        (r'\b(slot_selector)\b', "Concept"),
        (r'\b(retrieval_augmented_generation)\b', "Concept"),
        (r'\b(dense baseline)\b', "Concept"),
        (r'\b(dense open.book)\b', "Concept"),
    ]:
        for m in re.finditer(pattern, text):
            line = text[:m.start()].count('\n') + 1
            add(m.group(1), etype, line)

    # Metric mentions with values
    for m in re.finditer(
        r'((?:[\w_]+_)?(?:recall|accuracy|precision|F1|coverage|loss|MRR)(?:@\d+)?(?:_\w+)?)\s*=\s*([\d]+(?:\.[\d]+)?%?)',
        text, re.IGNORECASE
    ):
        line = text[:m.start()].count('\n') + 1
        add(m.group(1), "Metric", line)

    # Config file references
    for m in re.finditer(r'configs/([\w_]+\.ya?ml)', text):
        line = text[:m.start()].count('\n') + 1
        add(m.group(1), "CodeFile", line)

    # Python source references
    for m in re.finditer(r'(sam/[\w/]+\.py)', text):
        line = text[:m.start()].count('\n') + 1
        add(m.group(1), "CodeFile", line)

    # Gate mentions
    for m in re.finditer(
        r'Gate\s+([A-Z\d]+)\s*[^a-z]*?([—–-]\s*(.+?))?(?:\n|$|\.)',
        text, re.IGNORECASE
    ):
        line = text[:m.start()].count('\n') + 1
        gate_id = f"Gate {m.group(1)}"
        if m.group(3):
            gate_id += f": {m.group(3).strip()}"
        add(gate_id, "Decision", line)

    # Bold concepts **X**
    for m in re.finditer(r'\*\*([^*\n]{3,60})\*\*', text):
        bold = m.group(1).strip()
        if not any(w in bold.lower() for w in ('the ', 'and ', 'but ', 'that ', 'this ')):
            line = text[:m.start()].count('\n') + 1
            add(bold, "Concept", line)

    return entities


def ingest_directory(
    directory: Path,
    graph: InMemoryGraphStore,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Walk a directory, extract entities and relations from all .md files,
    and add them to the graph.

    Returns (nodes_added, edges_added).
    """
    md_files = sorted(directory.rglob("*.md"))
    nodes_added = 0
    edges_added = 0

    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, PermissionError) as e:
            if verbose:
                print(f"  Skipping {md_file}: {e}")
            continue

        if not text.strip():
            continue

        rel_path = str(md_file.relative_to(_project_root))

        # Extract entities from markdown (includes deduplication)
        entities = extract_from_markdown(text, rel_path)
        # Supplement with additional entity patterns
        supplement = _supplement_entities(text, rel_path)
        entities = merge_entity_lists([entities, supplement])

        if verbose and entities:
            print(f"  {rel_path}: {len(entities)} entities")

        # Add entity nodes to graph
        for entity in entities:
            node = _make_node(entity)
            if not graph.has_node(node.id):
                graph.add_node(node)
                nodes_added += 1
            else:
                existing = graph.get_node(node.id)
                if existing and rel_path not in existing.sources:
                    existing.sources.append(rel_path)

        # Extract relations (also returns newly discovered entities)
        relations, new_entities = extract_relations(text, rel_path, entities)
        
        # Add any entities discovered during relation extraction
        for entity in new_entities:
            node = _make_node(entity)
            if not graph.has_node(node.id):
                graph.add_node(node)
                nodes_added += 1
            else:
                existing = graph.get_node(node.id)
                if existing and rel_path not in existing.sources:
                    existing.sources.append(rel_path)
        
        if verbose and relations:
            print(f"    -> {len(relations)} relations")

        # Add edges to graph
        for rel in relations:
            source_id = _slugify(rel["source_name"])
            target_id = _slugify(rel["target_name"])

            if not graph.has_node(source_id) or not graph.has_node(target_id):
                continue

            edge = Edge(
                type=rel["edge_type"],
                source=source_id,
                target=target_id,
                confidence=rel["confidence"],
                evidence=rel.get("evidence", f"Extracted from {rel_path}"),
            )

            try:
                graph.add_edge(edge)
                edges_added += 1
            except KeyError:
                pass

    return nodes_added, edges_added


def print_stats(graph: InMemoryGraphStore):
    """Print a summary of the graph contents."""
    stats = graph.stats()
    print(f"\n{'='*60}")
    print(f"  NEXUS Ingestion Pipeline -- Results")
    print(f"{'='*60}")
    print(f"  Total nodes:  {graph.node_count}")
    print(f"  Total edges:  {graph.edge_count}")
    print(f"\n  Nodes by type:")
    for ntype, count in sorted(stats["node_types"].items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 40)
        print(f"    {ntype:<20} {count:>4}  {bar}")

    # Edge type breakdown
    edge_type_counts: dict[str, int] = defaultdict(int)
    edge_confidence_sum: dict[str, float] = defaultdict(float)
    for nid in graph._nodes:
        for edge in graph._edges_out.get(nid, []):
            edge_type_counts[edge.type] += 1
            edge_confidence_sum[edge.type] += edge.confidence

    print(f"\n  Edges by type:")
    for etype, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        avg_conf = edge_confidence_sum[etype] / count if count > 0 else 0
        bar = "#" * min(count, 30)
        print(f"    {etype:<20} {count:>4}  (avg conf: {avg_conf:.2f})  {bar}")

    # Show some examples
    print(f"\n  Sample Experiment nodes:")
    experiments = graph.nodes_of_type("Experiment")
    for node in experiments[:5]:
        print(f"    {node.id}")

    print(f"\n  Sample Concept nodes:")
    concepts = graph.nodes_of_type("Concept")[:5]
    for node in concepts:
        print(f"    {node.id}  <- {len(node.sources)} source(s)")

    # Top connected nodes
    print(f"\n  Top connected nodes:")
    node_edge_count = {}
    for nid in graph._nodes:
        node_edge_count[nid] = len(graph._edges_out.get(nid, [])) + len(graph._edges_in.get(nid, []))
    top_connected = sorted(node_edge_count.items(), key=lambda x: -x[1])[:10]
    for nid, count in top_connected:
        node = graph.get_node(nid)
        print(f"    {node.type:<15} {nid:<50} {count} edges")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NEXUS ingestion pipeline")
    parser.add_argument(
        "--dirs", nargs="+",
        default=["sam-lm/docs", "sam-lm/experiments"],
        help="Directories to scan for .md files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-file extraction stats"
    )
    args = parser.parse_args()

    graph = InMemoryGraphStore()
    total_nodes = 0
    total_edges = 0

    for dir_path in args.dirs:
        full_path = _project_root / dir_path
        if not full_path.exists():
            print(f"Warning: directory not found: {full_path}")
            continue
        
        print(f"\nScanning: {dir_path}")
        nodes, edges = ingest_directory(full_path, graph, verbose=args.verbose)
        total_nodes += nodes
        total_edges += edges
        print(f"  -> {nodes} nodes added, {edges} edges added")

    print(f"\nIngestion complete: {total_nodes} nodes, {total_edges} edges added to graph")
    print_stats(graph)


if __name__ == "__main__":
    main()
