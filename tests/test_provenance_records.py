"""Structured provenance adapters."""

from __future__ import annotations

from nexus.graph.provenance import (
    SourceRecord,
    normalize_sources,
    parse_freeform_source,
    provenance_coverage,
    source_id_for,
)


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
