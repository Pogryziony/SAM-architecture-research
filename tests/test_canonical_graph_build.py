"""Deterministic canonical graph build."""

from __future__ import annotations

from benchmarks.build_canonical_graph import build_canonical_graph, content_hash, graph_content_payload


def test_canonical_graph_hash_is_stable_across_two_builds():
    _g1, m1 = build_canonical_graph(enable_cooccurrence_edges=False)
    _g2, m2 = build_canonical_graph(enable_cooccurrence_edges=False)
    assert m1["content_hash"] == m2["content_hash"]
    assert m1["node_count"] == m2["node_count"]
    assert m1["edge_count"] == m2["edge_count"]
    assert m1["cooccurrence_enabled"] is False


def test_payload_excludes_timestamp_volatility():
    graph, provenance = __import__(
        "benchmarks.run_benchmark", fromlist=["build_benchmark_graph"]
    ).build_benchmark_graph()
    payload = graph_content_payload(graph, provenance)
    assert "timestamp" not in payload
    digest = content_hash(payload)
    assert len(digest) == 64
