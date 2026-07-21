"""Structured provenance adapters."""

from __future__ import annotations

from nexus.graph.provenance import (
    SourceRecord,
    attach_provenance_to_properties,
    normalize_sources,
    parse_freeform_source,
    provenance_coverage,
    provenance_dicts_for_sources,
    source_id_for,
)
from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR


def test_parse_freeform_source_is_deterministic():
    a = parse_freeform_source("docs/graph-memory.md:42")
    b = parse_freeform_source("docs/graph-memory.md:42")
    assert a == b
    assert a.source_id.startswith("src_")
    assert a.locator == "docs/graph-memory.md"
    assert a.extraction_method == "legacy_freeform"


def test_normalize_sources_deduplicates():
    records = normalize_sources([
        "docs/a.md:1",
        "docs/a.md:1",
        SourceRecord(source_id="src_custom", locator="docs/b.md", reliability=0.9),
    ])
    assert len(records) == 2
    assert provenance_coverage(records) == 1.0


def test_empty_source_has_zero_reliability():
    record = parse_freeform_source("  ")
    assert record.reliability == 0.0
    assert provenance_coverage([record]) == 0.0


def test_source_id_changes_with_content():
    assert source_id_for("x", "a") != source_id_for("x", "b")


def test_attach_provenance_to_properties_keeps_legacy_compat():
    props = attach_provenance_to_properties({"name": "X"}, ["docs/a.md:12"])
    assert props["name"] == "X"
    assert len(props["provenance"]) == 1
    assert props["provenance"][0]["locator"] == "docs/a.md"
    assert provenance_dicts_for_sources(["docs/a.md:12"])[0]["source_id"].startswith("src_")


def test_populated_nodes_carry_structured_provenance():
    from nexus.graph.store import InMemoryGraphStore

    graph = populate_graph(EXPERIMENTS_DIR, InMemoryGraphStore())
    sample = graph.get_node("Decision_PivotToNEXUS")
    assert sample is not None
    assert sample.sources
    assert "provenance" in sample.properties
    assert provenance_coverage(
        normalize_sources([
            SourceRecord(**row) if isinstance(row, dict) else row
            for row in sample.properties["provenance"]
        ])
    ) == 1.0
