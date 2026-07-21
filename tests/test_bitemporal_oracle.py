"""Stage 6 bi-temporal schema + oracle gold replay."""

from __future__ import annotations

from pathlib import Path

from benchmarks.run_bitemporal_replay import evaluate, _read_jsonl
from nexus.graph import Edge


def test_edge_exposes_bitemporal_fields():
    edge = Edge(
        type="depends_on",
        source="A",
        target="B",
        valid_from="2020-01-01T00:00:00+00:00",
        observed_at="2020-02-01T00:00:00+00:00",
    )
    stamp = edge.bitemporal_stamp()
    assert stamp["valid_from"].startswith("2020")
    assert stamp["observed_at"].startswith("2020")
    assert stamp["relation"] == "depends_on"


def test_bitemporal_oracle_replay_passes():
    report = evaluate(_read_jsonl(Path("benchmarks/qa-dataset/bitemporal_oracle_v1.jsonl")))
    assert report["status"] == "PASS", report.get("errors")
    assert report["n_records"] >= 5
