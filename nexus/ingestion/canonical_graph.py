"""Canonical SAM/NEXUS graph construction (runtime-owned, not benchmark-owned)."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from nexus.graph.store import InMemoryGraphStore
from nexus.utils.config import DEFAULT_CONFIG, NEXUSConfig

_ROOT = Path(__file__).resolve().parents[2]


def build_canonical_sam_graph(
    config: NEXUSConfig | None = None,
) -> tuple[InMemoryGraphStore, dict[str, Any]]:
    """Build the deterministic SAM research graph used by evaluation and runtime.

    This is the authoritative construction path. Benchmark scripts must call
    this function rather than owning graph construction.
    """
    cfg = config or DEFAULT_CONFIG
    from nexus.ingestion.populate_from_experiments import EXPERIMENTS_DIR, populate_graph
    from nexus.ingestion.ingest_docs import ingest_directory
    from nexus.graph.family_curations import apply_oracle_family_curations

    graph = InMemoryGraphStore()
    if EXPERIMENTS_DIR.exists():
        graph = populate_graph(EXPERIMENTS_DIR, graph)

    for rel in ("docs", "sam-lm/docs", "sam-lm/experiments"):
        path = _ROOT / rel
        if path.exists():
            ingest_directory(path, graph, config=cfg)

    family_curation_stats = apply_oracle_family_curations(graph)

    edge_type_counts: dict[str, int] = {}
    for node_id in graph._nodes:  # noqa: SLF001 — enumeration for provenance
        for edge in graph.get_outgoing(node_id):
            edge_type_counts[edge.type] = edge_type_counts.get(edge.type, 0) + 1

    provenance = {
        "schema_version": "nexus-graph-snapshot-v1",
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "edge_type_counts": edge_type_counts,
        "family_curations": family_curation_stats,
        "effective_config": {
            "enable_cooccurrence_edges": cfg.enable_cooccurrence_edges,
            "enable_embedding_er": cfg.enable_embedding_er,
            "enable_associative_encoder": cfg.enable_associative_encoder,
            "enable_normalization": cfg.enable_normalization,
        },
        "build_module": "nexus.ingestion.canonical_graph.build_canonical_sam_graph",
        "build_steps": [
            "populate_from_experiments",
            "ingest_directory(docs)",
            "ingest_directory(sam-lm/docs)",
            "ingest_directory(sam-lm/experiments)",
            "apply_oracle_family_curations",
        ],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    provenance["graph_snapshot_id"] = graph_snapshot_id(graph, provenance)
    return graph, provenance


def graph_snapshot_id(graph: InMemoryGraphStore, provenance: dict[str, Any] | None = None) -> str:
    """Deterministic snapshot identity from nodes, edges, properties, and temporal data.

    Strengthened identity includes:
    - Node IDs and counts
    - Edge types and counts
    - Edge endpoint pairs (source→target)
    - Node property keys (not values, to avoid timestamp drift)
    - Temporal metadata when available
    """
    node_ids = sorted(graph._nodes.keys())  # noqa: SLF001

    # Collect edge endpoints for identity
    edge_endpoints: list[str] = []
    edge_type_counts: dict[str, int] = {}
    for node_id in node_ids:
        for edge in graph.get_outgoing(node_id):
            edge_type_counts[edge.type] = edge_type_counts.get(edge.type, 0) + 1
            # Edge.target is the target node ID (Edge uses 'target' not 'target_id')
            target = getattr(edge, 'target', None) or getattr(edge, 'target_id', '')
            edge_endpoints.append(f"{node_id}->{edge.type}->{target}")

    # Collect node property keys (but not values to avoid timestamp-induced drift)
    property_keys_by_type: dict[str, set[str]] = {}
    for node_id, node in graph._nodes.items():  # noqa: SLF001
        node_type = getattr(node, "type", "unknown")
        if node_type not in property_keys_by_type:
            property_keys_by_type[node_type] = set()
        if hasattr(node, "properties") and isinstance(node.properties, dict):
            property_keys_by_type[node_type].update(node.properties.keys())
        elif hasattr(node, "__dict__"):
            property_keys_by_type[node_type].update(
                k for k in node.__dict__.keys() if not k.startswith("_")
            )

    # Serialize property keys consistently
    property_signature = {
        k: sorted(v) for k, v in sorted(property_keys_by_type.items())
    }

    payload = {
        "schema_version": "nexus-graph-snapshot-id-v2",
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "node_ids_sha256": hashlib.sha256(
            "\n".join(node_ids).encode("utf-8")
        ).hexdigest(),
        "edge_endpoints_sha256": hashlib.sha256(
            "\n".join(sorted(edge_endpoints)).encode("utf-8")
        ).hexdigest(),
        "edge_type_counts": edge_type_counts,
        "property_keys_signature": property_signature,
        "build_module": (provenance or {}).get("build_module", "unknown"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


# Backward-compatible alias used while benchmark wrappers migrate.
build_benchmark_graph = build_canonical_sam_graph
