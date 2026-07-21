"""Build the canonical NEXUS benchmark graph and emit a content hash.

Excludes volatile timestamps so two clean builds on the same tree produce the
same ``content_hash``. Does not rewrite historical benchmark artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.run_benchmark import build_benchmark_graph
from nexus.utils.config import NEXUSConfig


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def graph_content_payload(graph: Any, provenance: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic payload for hashing (no timestamps)."""
    node_ids = sorted(graph._nodes)
    nodes = []
    for node_id in node_ids:
        node = graph.get_node(node_id)
        props = {
            key: _jsonable(node.properties[key])
            for key in sorted(node.properties)
            if key not in {"created_at", "updated_at", "timestamp"}
        }
        nodes.append({
            "id": node.id,
            "type": node.type,
            "aliases": sorted(node.aliases or []),
            "sources": sorted(str(s) for s in (node.sources or [])),
            "properties": props,
        })

    edges = []
    for node_id in node_ids:
        for edge in graph.get_outgoing(node_id):
            edges.append({
                "source": edge.source,
                "type": edge.type,
                "target": edge.target,
                "confidence": round(float(edge.confidence), 6),
                "evidence": str(getattr(edge, "evidence", "") or ""),
            })
    edges.sort(key=lambda item: (item["source"], item["type"], item["target"], item["confidence"]))

    return {
        "schema_version": "nexus-canonical-graph-v1",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "edge_type_counts": dict(sorted((provenance.get("edge_type_counts") or {}).items())),
        "effective_config": provenance.get("effective_config") or {},
        "build_steps": provenance.get("build_steps") or [],
        "nodes": nodes,
        "edges": edges,
    }


def content_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_canonical_graph(
    *,
    enable_cooccurrence_edges: bool = False,
) -> tuple[Any, dict[str, Any]]:
    config = NEXUSConfig(enable_cooccurrence_edges=enable_cooccurrence_edges)
    graph, provenance = build_benchmark_graph(config)
    payload = graph_content_payload(graph, provenance)
    digest = content_hash(payload)
    manifest = {
        "schema_version": "nexus-canonical-graph-manifest-v1",
        "content_hash": digest,
        "node_count": payload["node_count"],
        "edge_count": payload["edge_count"],
        "edge_type_counts": payload["edge_type_counts"],
        "effective_config": payload["effective_config"],
        "build_steps": payload["build_steps"],
        "cooccurrence_enabled": bool(enable_cooccurrence_edges),
        "note": (
            "Production builds should keep enable_cooccurrence_edges=false; "
            "co-occurrence edges inflate related_to noise (see relation_eval)."
        ),
    }
    return graph, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/canonical_graph_manifest.json"),
        help="Manifest path (refuses overwrite unless --force)",
    )
    parser.add_argument(
        "--enable-cooccurrence",
        action="store_true",
        help="Opt-in experimental co-occurrence edges (not for production)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-hash-only", action="store_true")
    args = parser.parse_args()

    _graph, manifest = build_canonical_graph(
        enable_cooccurrence_edges=args.enable_cooccurrence,
    )
    if args.print_hash_only:
        print(manifest["content_hash"])
        return 0

    output = args.output
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if not args.force and (output.exists() or sidecar.exists()):
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
