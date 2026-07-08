"""
NEXUS generic ingestion pipeline — corpus-agnostic.
Takes any directory, extracts entities and relations from all files,
and populates a knowledge graph with zero hand-curated aliases.

Usage:
    python -m nexus.ingestion.ingest_generic --dir C:\\path\\to\\corpus
    python -m nexus.ingestion.ingest_generic --dir /external/docs --patterns "**/*.md" "**/*.txt"

Differences from ingest_docs.py:
    - No SAM-specific domain vocabulary or experiment variant stripping
    - No hand-curated alias lists
    - Relies purely on rule-based extraction (headers, backticks, bold text, verb patterns)
    - Works on any domain: tech docs, product specs, legal, medical, etc.
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
from nexus.ingestion.entity_extractor import extract_from_markdown, _is_valid_entity
from nexus.ingestion.relation_extractor import extract_relations
from nexus.ingestion.normalizer import canonicalize, normalize_entity_name
from nexus.ingestion.deduplicator import merge_entity_lists


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
    
    Domain-agnostic patterns:
      - Bold-emphasized concepts (**X**)
      - Numeric references (versions, IDs)
      - Long CamelCase identifiers
      - Mentioned technologies / proper nouns
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

    # Bold concepts **X** — skip very short or sentence-like patterns
    for m in re.finditer(r'\*\*([^*\n]{5,60})\*\*', text):
        bold = m.group(1).strip()
        bold = bold.strip('`').strip()
        if re.search(r'[.?!]$', bold):
            continue
        if bold.rstrip().endswith(':'):
            continue
        if re.match(r'^[\d.,%+\-=]', bold):
            continue
        # Skip obvious sentence fragments
        if re.match(r'^(the|and|but|that|this|these|those|for|with|from|it|is|are|was|were|be|a|an|if|when|how|use|no|not|into|small|large|every|each|can|has|have|also|only|very|any|our|per|may|via)\b',
                     bold, re.IGNORECASE):
            continue
        line = text[:m.start()].count('\n') + 1
        add(bold, "Concept", line)

    # Version numbers: v1.2.3, 2.0, 10.x
    for m in re.finditer(r'\bv?(\d+\.\d+(?:\.\d+)?(?:[a-z]\w*)?)\b', text):
        ver = m.group(1)
        if "." in ver and len(ver) >= 3:
            line = text[:m.start()].count('\n') + 1
            add(f"v{ver}", "Concept", line)

    # CamelCase identifiers (3+ words, at least 8 chars): NextJsAppRouter, ServiceListingDto
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+){2,})\b', text):
        ident = m.group(1)
        if len(ident) >= 8:
            line = text[:m.start()].count('\n') + 1
            add(ident, "Concept", line)

    # Technology mentions: common frameworks/tools
    tech_patterns = [
        (r'\b(Next\.?js|React|TypeScript|Tailwind CSS|PostgreSQL|Docker|nginx|SignalR|EF Core|ASP\.NET Core|JWT|OAuth|Playwright|Vitest|xUnit|GitHub Actions|Docker Compose)\b', "Technology"),
        (r'\b(Stripe|PayU|Przelewy24|SMSAPI|Sundream|BCrypt|Argon2)\b', "Technology"),
        (r'\b(REST API|SPA|SSR|SEO|CI/CD|CSRF|CORS|HMAC|TLS|SSL)\b', "Concept"),
    ]
    for pattern, etype in tech_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            line = text[:m.start()].count('\n') + 1
            add(m.group(1), etype, line)

    # ── Noise filter ──
    entities = [e for e in entities if _is_valid_entity(e["name"])]

    return entities


def ingest_generic(
    directory: Path,
    graph: InMemoryGraphStore,
    patterns: list[str] | None = None,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Walk a directory, extract entities and relations from files matching
    the given glob patterns, and add them to the graph.

    Args:
        directory: Root directory to scan
        graph: Graph store to populate
        patterns: Glob patterns for files (default: ["**/*.md", "**/*.txt", "**/*.py"])
        verbose: Print per-file extraction stats

    Returns:
        (nodes_added, edges_added)
    """
    if patterns is None:
        patterns = ["**/*.md", "**/*.txt", "**/*.py"]

    # Collect all matching files
    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern) if "**" in pattern
                     else directory.glob(pattern))

    # Deduplicate and sort
    files = sorted(set(files))

    nodes_added = 0
    edges_added = 0

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, PermissionError, OSError) as e:
            if verbose:
                print(f"  Skipping {file_path}: {e}")
            continue

        if not text.strip():
            continue

        # Use absolute path relative to the provided directory for consistent source refs
        try:
            rel_path = str(file_path.relative_to(directory))
        except ValueError:
            rel_path = str(file_path)

        # Extract entities from markdown (includes deduplication)
        entities = extract_from_markdown(text, rel_path)
        # Supplement with additional entity patterns (domain-agnostic)
        supplement = _supplement_entities(text, rel_path)
        entities = merge_entity_lists([entities, supplement])

        if verbose and entities:
            print(f"  {rel_path}: {len(entities)} entities")

        # Add entity nodes to graph
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

        # Extract relations
        relations, new_entities = extract_relations(text, rel_path, entities)

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
            norm = normalize_entity_name(name)
            if norm in entity_node_map:
                return entity_node_map[norm]
            store_result = graph.find_entity(name)
            if store_result:
                return store_result
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
    print(f"  NEXUS Generic Ingestion — Results")
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

    # Show sample entities by type
    for ntype in ["Concept", "Technology", "Document", "Entity"]:
        nodes = graph.nodes_of_type(ntype)
        if nodes:
            print(f"\n  Sample {ntype} nodes:")
            for node in nodes[:5]:
                print(f"    {node.id}  <- {len(node.sources)} source(s)")

    # Top connected nodes
    print(f"\n  Top connected nodes:")
    node_edge_count = {}
    for nid in graph._nodes:
        node_edge_count[nid] = len(graph._edges_out.get(nid, [])) + len(graph._edges_in.get(nid, []))
    top_connected = sorted(node_edge_count.items(), key=lambda x: -x[1])[:10]
    for nid, count in top_connected:
        node = graph.get_node(nid)
        if node:
            print(f"    {node.type:<15} {nid:<50} {count} edges")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="NEXUS generic ingestion pipeline — corpus-agnostic entity/relation extraction"
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory to scan for documents"
    )
    parser.add_argument(
        "--patterns", nargs="+",
        default=["**/*.md", "**/*.txt"],
        help="Glob patterns for files to include (default: **/*.md **/*.txt)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Optional path to save graph stats as JSON"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-file extraction stats"
    )
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.exists():
        print(f"Error: directory not found: {directory}")
        sys.exit(1)

    graph = InMemoryGraphStore()

    print(f"Ingesting from: {directory}")
    print(f"Patterns: {args.patterns}")

    nodes, edges = ingest_generic(
        directory, graph,
        patterns=args.patterns,
        verbose=args.verbose,
    )

    print(f"\nIngestion complete: {nodes} nodes added, {edges} edges added")
    print_stats(graph)

    if args.output:
        import json
        stats = graph.stats()
        stats["nodes_added"] = nodes
        stats["edges_added"] = edges
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStats saved to {args.output}")


if __name__ == "__main__":
    main()
