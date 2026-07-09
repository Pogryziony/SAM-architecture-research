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
from nexus.ingestion.entity_extractor import extract_from_markdown, _is_valid_entity, _extract_auto_aliases
from nexus.ingestion.relation_extractor import extract_relations
from nexus.ingestion.normalizer import canonicalize, normalize_entity_name
from nexus.ingestion.deduplicator import merge_entity_lists


# ── Sub-experiment variant suffix patterns ──
# Matched case-insensitively against entity names to identify sub-run variants.
# Stripped iteratively: numeric/metric patterns first, then concept suffixes.
_SUB_VARIANT_PATTERNS: list[str] = [
    # Compound variant suffixes must be listed BEFORE their simpler components
    # so they match as a whole unit (e.g., _oracle_filter_top64 before _top64).
    r'[_\s]weighted[_\s]t\d{3}[_\s]top\d+',   # _weighted_t005_top32
    r'[_\s]oracle[_\s]filter[_\s]top\d+$',       # _oracle_filter_top64 (compound, must precede _topXX)
    r'[_\s]top\d+$',                             # _top64, _top8
    r'[_\s]external[_\s]text[_\s]query',         # _external_text_query
    r'[_\s]sam[_\s]chain[_\s]aware',             # _sam_chain_aware
    r'[_\s]hidden[_\s]adapter',                  # _hidden_adapter
    r'[_\s]dual[_\s]encoder',                    # _dual_encoder
    r'[_\s]retrieval[_\s]compact',               # _retrieval_compact
    r'[_\s]oracle[_\s]slots',                    # _oracle_slots
    r'[_\s]baseline$',                           # _baseline
    r'[_\s]hardneg$',                            # _hardneg
    r'[_\s]improved$',                           # _improved
]


def _clean_experiment_name(name: str) -> tuple[str, str | None]:
    """
    Strip sub-experiment variant suffixes from an entity name.

    Returns (clean_name, original_name) if stripping occurred,
    or (original_name, None) if no change needed.

    Examples:
        'Experiment 0.9 oracle_filter weighted_t005_top32'
            -> ('Experiment 0.9 oracle_filter', 'Experiment 0.9 oracle_filter weighted_t005_top32')
        'Exp_0_12_Selection_oracle_filter_top64'
            -> ('Exp_0_12_Selection', 'Exp_0_12_Selection_oracle_filter_top64')
    """
    original = name
    clean = name
    changed = True
    while changed:
        changed = False
        for pattern in _SUB_VARIANT_PATTERNS:
            m = re.search(pattern, clean, re.IGNORECASE)
            if m:
                clean = clean[:m.start()] + clean[m.end():]
                changed = True
                break
    clean = re.sub(r'[_\s]{2,}', '_', clean).strip('_').strip()
    if clean and clean != original:
        return clean, original
    return name, None


def _slugify(name: str, entity_type: str = "Entity") -> str:
    """Create a normalized ID from an entity name using the canonicalizer."""
    return canonicalize(name, entity_type)


def _make_node(entity: dict, aliases: list[str] | None = None) -> Node:
    """Convert an entity dict from the extractor to a Node."""
    name = entity["name"]
    etype = entity.get("type", "Entity")
    node_id = _slugify(name, etype)
    # Populate aliases: if the raw name normalizes differently from the node_id,
    # add it as an alias for recall
    raw_alias = name.lower().replace(" ", "_")
    final_aliases = list(aliases) if aliases else []
    if raw_alias != node_id and raw_alias not in final_aliases:
        final_aliases.append(raw_alias)
    props = {
        "name": name,
        "display_name": name,
    }
    if "properties" in entity:
        props.update(entity["properties"])
    return Node(
        id=node_id,
        type=entity.get("type", "Entity"),
        properties=props,
        sources=[entity.get("source", "")],
        aliases=final_aliases,
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

    # ── Noise filter: reject entities that fail _is_valid_entity ──
    entities = [e for e in entities if _is_valid_entity(e["name"])]

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

        # ── Layer 1: Clean experiment names & merge sub-run variants ──
        exp_groups: dict[str, dict] = {}         # clean_name -> merged entity
        exp_order: list[str] = []                 # preserve order
        non_exp_entities: list[dict] = []

        for entity in entities:
            etype = entity.get("type", "Entity")
            if etype == "Experiment":
                name = entity["name"]
                clean_name, original = _clean_experiment_name(name)
                entity["name"] = clean_name
                # Collect original variant name as an alias
                if original:
                    entity.setdefault("_aliases", []).append(original)

                norm_key = normalize_entity_name(clean_name)
                if norm_key in exp_groups:
                    # Merge into existing parent: combine sources, aliases, properties
                    parent = exp_groups[norm_key]
                    if "_aliases" in entity:
                        parent.setdefault("_aliases", []).extend(entity["_aliases"])
                    if "properties" in entity:
                        parent.setdefault("properties", {}).update(entity["properties"])
                    if entity.get("source") and entity["source"] not in parent.setdefault("sources", []):
                        parent.setdefault("sources", []).append(entity["source"])
                else:
                    exp_groups[norm_key] = entity
                    exp_order.append(norm_key)
            else:
                non_exp_entities.append(entity)

        # Rebuild entity list with merged experiments + non-experiments
        entities = non_exp_entities + [exp_groups[k] for k in exp_order]

        if verbose and entities:
            print(f"  {rel_path}: {len(entities)} entities")

        # Add entity nodes to graph
        # Also track name -> node_id mapping for edge creation
        entity_node_map: dict[str, str] = {}  # normalized_name -> node_id
        for entity in entities:
            extra_aliases = entity.pop("_aliases", []) if "_aliases" in entity else []
            node = _make_node(entity, aliases=extra_aliases)
            entity_node_map[normalize_entity_name(entity["name"])] = node.id
            if not graph.has_node(node.id):
                graph.add_node(node)
                nodes_added += 1
            else:
                existing = graph.get_node(node.id)
                if existing and rel_path not in existing.sources:
                    existing.sources.append(rel_path)
                # Merge aliases into existing node
                if extra_aliases:
                    for alias in extra_aliases:
                        if alias not in existing.aliases:
                            existing.aliases.append(alias)

        # Extract relations (also returns newly discovered entities)
        relations, new_entities = extract_relations(text, rel_path, entities)
        
        # Clean experiment names in relation-discovered entities
        for entity in new_entities:
            if entity.get("type") == "Experiment":
                name = entity["name"]
                clean_name, original = _clean_experiment_name(name)
                entity["name"] = clean_name
                if original:
                    entity.setdefault("_aliases", []).append(original)
        
        # Add any entities discovered during relation extraction
        for entity in new_entities:
            extra_aliases = entity.pop("_aliases", []) if "_aliases" in entity else []
            node = _make_node(entity, aliases=extra_aliases)
            entity_node_map[normalize_entity_name(entity["name"])] = node.id
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
        def _resolve_edge_node(name: str) -> str | None:
            """Resolve an entity name to a node ID using the local map or store lookup."""
            # Try local map first
            norm = normalize_entity_name(name)
            if norm in entity_node_map:
                return entity_node_map[norm]
            # Fall back to store's find_entity (includes fuzzy + alias matching)
            store_result = graph.find_entity(name)
            if store_result:
                return store_result
            # Last resort: try slugify with store's node ID matching
            slug = _slugify(name)
            if graph.has_node(slug):
                return slug
            return None

        for rel in relations:
            source_id = _resolve_edge_node(rel["source_name"])
            target_id = _resolve_edge_node(rel["target_name"])

            if source_id is None or target_id is None:
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
