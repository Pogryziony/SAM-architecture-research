"""Canonical graph ownership and process resource helpers."""

from __future__ import annotations

from nexus.evaluation.process_resources import (
    nvidia_vram_used_mb,
    process_tree_rss_mb,
    snapshot_llm_server_resources,
)
from nexus.ingestion.canonical_graph import build_canonical_sam_graph, graph_snapshot_id


def test_canonical_graph_builds_and_has_snapshot_id():
    graph, prov = build_canonical_sam_graph()
    assert graph.node_count > 0
    assert prov["build_module"].endswith("build_canonical_sam_graph")
    assert prov["graph_snapshot_id"]
    assert graph_snapshot_id(graph, prov) == prov["graph_snapshot_id"]


def test_benchmark_wrapper_delegates():
    from benchmarks.run_benchmark import build_benchmark_graph

    g1, p1 = build_benchmark_graph()
    g2, p2 = build_canonical_sam_graph()
    assert g1.node_count == g2.node_count
    assert p1["graph_snapshot_id"] == p2["graph_snapshot_id"]


def test_resource_snapshot_schema():
    snap = snapshot_llm_server_resources()
    assert snap["schema_version"] == "nexus-llm-server-resources-v1"
    assert "ollama_rss_mb" in snap
    assert "vram" in snap
    vram = nvidia_vram_used_mb()
    assert "available" in vram
    # process RSS may be None without psutil; must not raise
    _ = process_tree_rss_mb()
