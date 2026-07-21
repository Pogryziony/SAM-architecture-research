"""Stage 6 bi-temporal replay tests."""

from __future__ import annotations

from nexus.graph.bitemporal import assert_no_lookahead, filter_facts_bitemporal


def test_filter_rejects_future_observed_facts():
    facts = [
        {
            "source": "A",
            "relation": "depends_on",
            "target": "B",
            "observed_at": "2026-01-01T00:00:00+00:00",
            "valid_from": "2025-01-01T00:00:00+00:00",
        },
        {
            "source": "A",
            "relation": "depends_on",
            "target": "C",
            "observed_at": "2026-06-01T00:00:00+00:00",
            "valid_from": "2025-01-01T00:00:00+00:00",
        },
    ]
    kept = filter_facts_bitemporal(facts, as_known_at="2026-03-01T00:00:00+00:00")
    assert len(kept) == 1
    assert kept[0]["target"] == "B"


def test_lookahead_errors_for_future_knowledge():
    facts = [
        {
            "source": "A",
            "target": "C",
            "observed_at": "2026-06-01T00:00:00+00:00",
        }
    ]
    errors = assert_no_lookahead(facts, as_known_at="2026-03-01T00:00:00+00:00")
    assert errors
