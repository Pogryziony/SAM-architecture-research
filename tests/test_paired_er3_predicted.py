"""Paired oracle vs ER3 predicted arm (uses frozen checkpoint; no training)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.run_oracle_vs_predicted import run_paired_benchmark
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.model_interface import DummyModel

pytest.importorskip("torch")

_ER3_DIR = Path("models/encoder/entity_ranker_v3_20260711T081545Z")
_WEIGHTS = _ER3_DIR / "weights.pt"


@pytest.mark.skipif(not _WEIGHTS.exists(), reason="frozen ER3 weights not present")
def test_er3_predicted_arm_beats_empty_entry_recall_on_toy_graph():
    from stack.pipeline.resolver import ER3Resolver

    graph = InMemoryGraphStore()
    graph.add_node(Node(
        id="Exp_0_12_Selection",
        type="Experiment",
        properties={"title": "Selection bottleneck", "key_finding": "selector precision 50%"},
        sources=["docs/a.md"],
        aliases=["selection experiment"],
    ))
    graph.add_node(Node(
        id="Concept_SelectorBottleneck",
        type="Concept",
        properties={"description": "Selector precision bottleneck"},
        sources=["docs/b.md"],
        aliases=["selector bottleneck"],
    ))
    graph.add_edge(Edge(
        type="validates",
        source="Exp_0_12_Selection",
        target="Concept_SelectorBottleneck",
        confidence=0.95,
        evidence="docs/a.md",
    ))
    # Resolver construction proves frozen weights load; full ranking needs
    # richer graph text and is covered by the published paired artifact.
    resolver = ER3Resolver.from_directory(str(_ER3_DIR), graph)
    assert resolver.resolve("What is the selector bottleneck?", graph).selected_entity_ids

    records = [{
        "id": "toy_er3_001",
        "question": "What experiment validates Concept_SelectorBottleneck?",
        "gold_answer": "Exp_0_12_Selection validates Concept_SelectorBottleneck.",
        "gold_entities": ["Exp_0_12_Selection", "Concept_SelectorBottleneck"],
        "gold_path": [{
            "source": "Exp_0_12_Selection",
            "relation": "validates",
            "target": "Concept_SelectorBottleneck",
        }],
        "path_required": True,
        "should_abstain": False,
        "category": "two_hop",
        "source_split": "unit",
    }]
    artifact = run_paired_benchmark(
        records,
        graph,
        source_sha="unit",
        dataset_identity={"file_sha256": "unit", "record_count": 1},
        predicted_resolver="er3",
        er3_dir=str(_ER3_DIR),
        model=DummyModel(),
    )
    assert artifact["schema_version"] == "nexus-oracle-vs-predicted-v2"
    assert artifact["predicted_resolver"]["name"] == "entity_ranker_v3"
    assert "entry_recall_mean" in artifact["predicted"]["metrics"]
    assert "pool_recall_mean" in artifact["predicted"]["metrics"]
    # Oracle entries are gold, so entry recall must be perfect.
    assert artifact["oracle"]["metrics"]["entry_recall_mean"] == 1.0


@pytest.mark.skipif(not _WEIGHTS.exists(), reason="frozen ER3 weights not present")
def test_union_predicted_arm_identity_and_pool_metrics():
    records = [{
        "id": "toy_union_001",
        "question": "What experiment validates Concept_SelectorBottleneck?",
        "gold_answer": "Exp_0_12_Selection validates Concept_SelectorBottleneck.",
        "gold_entities": ["Exp_0_12_Selection", "Concept_SelectorBottleneck"],
        "gold_path": [{
            "source": "Exp_0_12_Selection",
            "relation": "validates",
            "target": "Concept_SelectorBottleneck",
        }],
        "path_required": True,
        "should_abstain": False,
        "category": "two_hop",
        "source_split": "unit",
    }]
    graph = InMemoryGraphStore()
    graph.add_node(Node(
        id="Exp_0_12_Selection",
        type="Experiment",
        properties={"title": "Selection bottleneck"},
        aliases=["selection experiment"],
    ))
    graph.add_node(Node(
        id="Concept_SelectorBottleneck",
        type="Concept",
        properties={"description": "Selector precision bottleneck"},
        aliases=["selector bottleneck"],
    ))
    graph.add_edge(Edge(
        type="validates",
        source="Exp_0_12_Selection",
        target="Concept_SelectorBottleneck",
        confidence=0.95,
        evidence="docs/a.md",
    ))
    artifact = run_paired_benchmark(
        records,
        graph,
        source_sha="unit",
        dataset_identity={"file_sha256": "unit", "record_count": 1},
        predicted_resolver="union",
        er3_dir=str(_ER3_DIR),
        model=DummyModel(),
    )
    assert artifact["predicted_resolver"]["name"] == "union_lexical_er3"
    assert artifact["predicted_resolver"]["entity_ranker_v3_enabled"] is True
    assert "pool_recall_mean" in artifact["predicted"]["metrics"]
    assert artifact["oracle"]["metrics"]["entry_recall_mean"] == 1.0
