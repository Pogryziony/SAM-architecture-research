"""Production ingest stamps bi-temporal fields on edges."""

from __future__ import annotations

from pathlib import Path

from nexus.graph.bitemporal import DEFAULT_INGEST_EPOCH
from nexus.graph.store import InMemoryGraphStore
from nexus.ingestion.populate_from_experiments import populate_graph


def test_populate_stamps_edges_with_stable_observed_at():
    experiments_dir = Path("sam-lm/experiments")
    assert experiments_dir.exists(), "expected sam-lm/experiments for stamp test"
    graph = populate_graph(experiments_dir, InMemoryGraphStore())
    stamped = 0
    for node_id in graph._nodes:
        for edge in graph.get_outgoing(node_id):
            assert edge.observed_at, f"missing observed_at on {edge}"
            assert edge.valid_from, f"missing valid_from on {edge}"
            assert edge.observed_at.startswith("2026-07-08")
            stamped += 1
    assert stamped > 0
    assert DEFAULT_INGEST_EPOCH.startswith("2026-07-08")
