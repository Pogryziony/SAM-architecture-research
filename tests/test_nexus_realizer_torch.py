"""CPU PyTorch preflight for the actual NEXUS Realizer model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from benchmarks.build_distillation_dataset import build_distillation_dataset
from benchmarks.train_nexus_realizer import run
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore


def test_actual_realizer_forward_backward_preflight_writes_no_weights(tmp_path: Path):
    graph = InMemoryGraphStore()
    questions = []
    for index in range(5):
        source = f"Source{index}"
        target = f"Target{index}"
        graph.add_node(Node(
            source, "Concept",
            properties={"description": f"{source} achieved 90% accuracy."},
            sources=[f"docs/{source}.md"],
        ))
        graph.add_node(Node(
            target, "Concept",
            properties={"description": f"{target} supports the result."},
            sources=[f"docs/{target}.md"],
        ))
        graph.add_edge(Edge(
            "related_to", source, target, evidence=f"docs/edge-{index}.md"
        ))
        questions.append({
            "id": f"q{index}",
            "question": f"What accuracy did {source} achieve?",
            "answer": f"{source} achieved 90% accuracy.",
            "entities": [source],
            "source_split": "train",
        })

    dataset_root = tmp_path / "dataset"
    build_distillation_dataset(
        questions, graph, str(dataset_root), "a" * 40, min_pairs=5
    )
    result = run(
        dataset_root / "manifest.json",
        Path("training/nexus_realizer_v1.json"),
        mode="preflight",
    )
    assert result["status"] == "PREFLIGHT_PASS"
    assert 0 < result["parameter_count"] <= 50_000_000
    assert result["loss"] > 0
    assert result["priority_evidence_coverage"] == {"mean": 1.0, "min": 1.0}
    assert result["weights_written"] is False
    assert not list(tmp_path.rglob("*.pt"))
