"""Synthetic traversal budget campaign guards."""

from __future__ import annotations

from benchmarks.run_traversal_budget_campaign import (
    GRAPH_SPECS,
    build_synthetic_graph,
    run_size_campaign,
    run_traversal_samples,
    validate_campaign_artifact,
)
from nexus.utils.config import NEXUSConfig


def test_build_synthetic_graph_is_deterministic():
    a = build_synthetic_graph(20, branching=2)
    b = build_synthetic_graph(20, branching=2)
    assert a.node_count == b.node_count == 20
    assert a.edge_count == b.edge_count
    assert a.edge_count > 0


def test_tight_budgets_truncate_on_small_graph():
    graph = build_synthetic_graph(40, branching=3)
    metrics = run_traversal_samples(
        graph,
        NEXUSConfig(max_depth=4, beam_width=10, max_expanded_edges=5, max_expanded_nodes=5),
        starts=["n0", "n5"],
        repeats=1,
    )
    assert metrics["truncated_runs"] > 0
    assert metrics["truncation_reasons"]


def test_validate_campaign_artifact_accepts_pass_shape():
    # Hand-built metrics: do not sample live process RSS (may already include
    # torch/other suite fixtures when this test runs late in CI).
    sizes = [
        {
            "size": name,
            "default_budgets": {
                "truncated_runs": 0,
                "latency_p50_ms": 1.0,
                "latency_p95_ms": 2.0,
            },
            "tight_budgets": {"truncated_runs": 3},
            "peak_rss_mb": 40.0,
        }
        for name in GRAPH_SPECS
    ]
    artifact = {
        "schema_version": "nexus-traversal-budget-campaign-v1",
        "preregistration_id": "traversal-budgets-v1",
        "reference_cpu_label": "github-actions-ubuntu-latest-x86_64",
        "platform": "linux",
        "python_version": "3.12.0",
        "cpu_model": "test",
        "source_sha": "abc",
        "sizes": sizes,
        "errors": [],
    }
    assert validate_campaign_artifact(artifact) == []


def test_run_size_campaign_reports_tight_truncation():
    row = run_size_campaign("small", {"nodes": 40, "branching": 3})
    assert row["default_budgets"]["truncated_runs"] == 0
    assert row["tight_budgets"]["truncated_runs"] > 0


def test_validate_campaign_artifact_fails_when_tight_does_not_truncate():
    artifact = {
        "schema_version": "nexus-traversal-budget-campaign-v1",
        "preregistration_id": "traversal-budgets-v1",
        "reference_cpu_label": "github-actions-ubuntu-latest-x86_64",
        "platform": "linux",
        "python_version": "3.12.0",
        "cpu_model": "test",
        "source_sha": "abc",
        "sizes_requested": ["small"],
        "sizes": [
            {
                "size": "small",
                "default_budgets": {
                    "truncated_runs": 0,
                    "latency_p50_ms": 1.0,
                    "latency_p95_ms": 2.0,
                },
                "tight_budgets": {"truncated_runs": 0},
                "peak_rss_mb": 10.0,
            },
        ],
        "errors": [],
    }
    errors = validate_campaign_artifact(artifact)
    assert any("tight budgets must truncate" in error for error in errors)


def test_validate_requires_preregistration_identity():
    artifact = {
        "schema_version": "nexus-traversal-budget-campaign-v1",
        "sizes_requested": ["small"],
        "sizes": [
            {
                "size": "small",
                "default_budgets": {
                    "truncated_runs": 0,
                    "latency_p50_ms": 1.0,
                    "latency_p95_ms": 2.0,
                },
                "tight_budgets": {"truncated_runs": 1},
                "peak_rss_mb": 10.0,
            },
        ],
        "errors": [],
    }
    errors = validate_campaign_artifact(artifact)
    assert any("preregistration_id" in error for error in errors)
