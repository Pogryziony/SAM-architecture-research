"""Tests for the canonical entity mapping module.

Covers: one-to-one, many-to-one, missing-parent, cyclic, and ambiguous mappings.
"""
from __future__ import annotations

import pytest

from nexus.graph import Node, Edge
from nexus.graph.store import InMemoryGraphStore
from stack.encoder.canonical_mapping import (
    build_canonical_mapping,
    apply_canonical_mapping,
    _find_canonical,
)


def _graph() -> InMemoryGraphStore:
    """Build a minimal test graph with known canonical structure."""
    g = InMemoryGraphStore()

    # Canonical experiments
    g.add_node(Node(id="Exp_0_6_Validation", type="Experiment"))
    g.add_node(Node(id="Exp_0_7_ExternalText", type="Experiment"))
    g.add_node(Node(id="Concept_RetrievalMismatch", type="Concept"))

    # Sub-experiments – derive from canonical
    g.add_node(Node(id="Exp_0_6_Validation_core_only", type="Experiment"))
    g.add_edge(Edge(type="derived_from", source="Exp_0_6_Validation_core_only",
                    target="Exp_0_6_Validation"))

    # Granular metric nodes – derive from sub-experiments
    g.add_node(Node(id="Metric_Exp_0_6_Validation_core_only_accuracy", type="Metric"))
    g.add_edge(Edge(type="derived_from", source="Metric_Exp_0_6_Validation_core_only_accuracy",
                    target="Exp_0_6_Validation_core_only"))

    g.add_node(Node(id="Metric_Exp_0_6_Validation_dense_baseline_accuracy", type="Metric"))
    # No derived_from — missing-parent node

    # Cyclic structure
    g.add_node(Node(id="Cycle_A", type="Experiment"))
    g.add_node(Node(id="Cycle_B", type="Experiment"))
    g.add_edge(Edge(type="derived_from", source="Cycle_A", target="Cycle_B"))
    g.add_edge(Edge(type="derived_from", source="Cycle_B", target="Cycle_A"))

    # Non-canonical node with no derived_from
    g.add_node(Node(id="Unrelated_Entity", type="Entity"))

    # Concept variant
    g.add_node(Node(id="Concept_RetrievalMismatch_variant", type="Concept"))
    g.add_edge(Edge(type="derived_from", source="Concept_RetrievalMismatch_variant",
                    target="Concept_RetrievalMismatch"))

    return g


class TestCanonicalMapping:
    """T5: Canonical mapping determinism, edge cases, and correctness."""

    def test_one_to_one_canonical_maps_to_self(self):
        """Canonical nodes map to themselves."""
        g = _graph()
        mapping = build_canonical_mapping(g)
        assert mapping.get("Exp_0_6_Validation") == "Exp_0_6_Validation"
        assert mapping.get("Concept_RetrievalMismatch") == "Concept_RetrievalMismatch"

    def test_many_to_one_granular_to_canonical(self):
        """Two granular metrics map to the same canonical experiment."""
        g = _graph()
        mapping = build_canonical_mapping(g)
        # Sub-experiment maps to canonical
        assert mapping.get("Exp_0_6_Validation_core_only") == "Exp_0_6_Validation"
        # Metric maps to canonical (two hops)
        assert mapping.get("Metric_Exp_0_6_Validation_core_only_accuracy") == "Exp_0_6_Validation"

    def test_missing_parent_node_excluded(self):
        """Nodes with no derived_from to canonical are not in mapping."""
        g = _graph()
        mapping = build_canonical_mapping(g)
        assert "Metric_Exp_0_6_Validation_dense_baseline_accuracy" not in mapping
        assert "Unrelated_Entity" not in mapping

    def test_cyclic_mapping_is_safe(self):
        """Cyclic derived_from chains do not cause infinite recursion."""
        g = _graph()
        mapping = build_canonical_mapping(g)
        # Cycle_A and Cycle_B are not canonical patterns, so should not map
        assert "Cycle_A" not in mapping
        assert "Cycle_B" not in mapping

    def test_ambiguous_mapping_deterministic(self):
        """Same graph always produces the same mapping (deterministic)."""
        g = _graph()
        m1 = build_canonical_mapping(g)
        m2 = build_canonical_mapping(g)
        assert m1 == m2

    def test_concept_variant_maps_to_canonical_concept(self):
        """Concept_<name>_variant maps to Concept_<name>."""
        g = _graph()
        mapping = build_canonical_mapping(g)
        assert mapping.get("Concept_RetrievalMismatch_variant") == "Concept_RetrievalMismatch"

    def test_apply_canonical_mapping_deduplicates(self):
        """apply_canonical_mapping deduplicates after mapping and caps at K."""
        mapping = {
            "M1": "E1",
            "M2": "E1",  # same target
            "M3": "E2",
            "M4": "E2",  # same target
            "M5": "E3",
        }
        ranked = ["M1", "M2", "M3", "M4", "M5"]
        result = apply_canonical_mapping(ranked, mapping, top_k=3)
        assert result == ["E1", "E2", "E3"]
        assert len(result) == 3

    def test_apply_canonical_mapping_preserves_order(self):
        """First occurrence of a canonical target claims that position."""
        mapping = {"M1": "E1", "M2": "E2", "M3": "E1"}
        assert apply_canonical_mapping(["M1", "M2", "M3"], mapping, 10) == ["E1", "E2"]
        # Reversed input should give different order
        assert apply_canonical_mapping(["M3", "M2", "M1"], mapping, 10) == ["E1", "E2"]

    def test_apply_canonical_mapping_caps_at_k(self):
        """Result length does not exceed top_k."""
        mapping = {f"N{i}": f"E{i}" for i in range(20)}
        ranked = [f"N{i}" for i in range(20)]
        assert len(apply_canonical_mapping(ranked, mapping, top_k=5)) == 5
        assert len(apply_canonical_mapping(ranked, mapping, top_k=10)) == 10

    def test_find_canonical_depth_limit(self):
        """Deep chains are bounded by depth limit."""
        g = InMemoryGraphStore()
        prev = None
        for i in range(15):
            nid = f"Chain_{i}"
            g.add_node(Node(id=nid, type="Metric"))
            if prev:
                g.add_edge(Edge(type="derived_from", source=nid, target=prev))
            prev = nid
        # Add a canonical at the end
        g.add_node(Node(id="Exp_0_1_Test", type="Experiment"))
        g.add_edge(Edge(type="derived_from", source="Chain_0", target="Exp_0_1_Test"))
        # Chain_14 is 15 hops away — should not resolve due to depth limit
        assert _find_canonical("Chain_14", g) is None
        # Chain_9 is 10 hops away — should resolve
        result = _find_canonical("Chain_9", g)
        assert result == "Exp_0_1_Test"
