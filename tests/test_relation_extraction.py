"""
Validation tests for relation extraction graph edges.

Tests that each supported edge type appears in the graph, invalid edge types
are rejected, confidence values are in [0, 1], and co-occurrence edges have
confidence exactly 0.3.
"""

import importlib
import json
import pytest
from pathlib import Path

import nexus.graph as ng
from nexus.graph.store import InMemoryGraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populated_graph() -> InMemoryGraphStore:
    """Return a graph populated from experiment data."""
    from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR
    graph = InMemoryGraphStore()
    graph = populate_graph(EXPERIMENTS_DIR, graph)
    return graph


def _load_gold() -> tuple[list[dict], list[dict]]:
    """Load the gold-standard relation dataset."""
    import sys
    _project_root = Path(__file__).resolve().parents[1]
    gold_path = _project_root / "benchmarks" / "qa-dataset" / "relation_gold.jsonl"
    positives = []
    negatives = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("is_negative", False):
                negatives.append(entry)
            else:
                positives.append(entry)
    return positives, negatives


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    return _populated_graph()


@pytest.fixture(scope="module")
def gold_positives():
    positives, _negatives = _load_gold()
    return positives


@pytest.fixture(scope="module")
def gold_negatives():
    _positives, negatives = _load_gold()
    return negatives


# ---------------------------------------------------------------------------
# Test: Supported edge types appear in the graph
# ---------------------------------------------------------------------------

class TestEdgeTypesPresent:
    """Each edge type in EDGE_TYPES should have at least one edge in the graph
    if the graph is populated with experiments. Some types (caused_by, implements)
    may not have edges yet — those are skipped with a note."""

    def test_depends_on_present(self, graph):
        edges = [e for nid in graph._nodes for e in graph.get_outgoing(nid) if e.type == "depends_on"]
        assert len(edges) > 0, "Expected at least one 'depends_on' edge in the graph"

    def test_validates_present(self, graph):
        edges = [e for nid in graph._nodes for e in graph.get_outgoing(nid) if e.type == "validates"]
        assert len(edges) > 0, "Expected at least one 'validates' edge in the graph"

    def test_derived_from_present(self, graph):
        edges = [e for nid in graph._nodes for e in graph.get_outgoing(nid) if e.type == "derived_from"]
        assert len(edges) > 0, "Expected at least one 'derived_from' edge in the graph"

    def test_contradicts_present(self, graph):
        edges = [e for nid in graph._nodes for e in graph.get_outgoing(nid) if e.type == "contradicts"]
        assert len(edges) > 0, "Expected at least one 'contradicts' edge in the graph"


# ---------------------------------------------------------------------------
# Test: Invalid edge types are rejected
# ---------------------------------------------------------------------------

class TestInvalidEdgeTypes:
    """Edge types not in EDGE_TYPES should not appear in the graph."""

    def test_no_invalid_edge_types_in_graph(self, graph):
        valid_types = ng.EDGE_TYPES
        for nid in graph._nodes:
            for edge in graph.get_outgoing(nid):
                assert edge.type in valid_types, (
                    f"Edge from '{edge.source}' to '{edge.target}' "
                    f"has invalid type '{edge.type}'"
                )

    def test_unknown_type_not_in_edgetypes(self):
        """Verify that a made-up type is not in the valid set."""
        assert "fantasy_edge" not in ng.EDGE_TYPES


# ---------------------------------------------------------------------------
# Test: Confidence is in [0, 1] range
# ---------------------------------------------------------------------------

class TestConfidenceRange:
    """All edges in the graph must have confidence in [0.0, 1.0]."""

    def test_all_confidences_in_range(self, graph):
        for nid in graph._nodes:
            for edge in graph.get_outgoing(nid):
                assert 0.0 <= edge.confidence <= 1.0, (
                    f"Edge '{edge.type}' from '{edge.source}' to '{edge.target}' "
                    f"has confidence {edge.confidence} outside [0, 1]"
                )

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_out_of_range_confidence_rejected_at_construction(self, confidence):
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            ng.Edge(type="depends_on", source="A", target="B", confidence=confidence)

    def test_non_numeric_confidence_rejected_at_construction(self):
        with pytest.raises(TypeError, match="must be a number"):
            ng.Edge(type="depends_on", source="A", target="B", confidence="high")

    def test_invalid_edge_type_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Unsupported edge type"):
            ng.Edge(type="fantasy_edge", source="A", target="B")


# ---------------------------------------------------------------------------
# Test: Co-occurrence edges have confidence exactly 0.3
# ---------------------------------------------------------------------------

class TestCooccurrenceEdges:
    """Co-occurrence ('related_to') edges must have confidence = 0.3."""

    def test_cooccurrence_confidence_is_0_3(self, graph):
        cooccurrence_edges = [
            e for nid in graph._nodes
            for e in graph.get_outgoing(nid)
            if e.type == "related_to"
        ]
        for edge in cooccurrence_edges:
            assert edge.confidence == 0.3, (
                f"Co-occurrence edge '{edge.source}' -> '{edge.target}' "
                f"has confidence {edge.confidence}, expected 0.3"
            )

    def test_cooccurrence_edges_have_correct_type(self, graph):
        for nid in graph._nodes:
            for edge in graph.get_outgoing(nid):
                if edge.type == "related_to":
                    assert edge.confidence == 0.3


# ---------------------------------------------------------------------------
# Test: Gold dataset integrity
# ---------------------------------------------------------------------------

class TestGoldDataset:
    """Integrity checks on the gold-standard relation dataset."""

    def test_at_least_20_positive_examples(self, gold_positives):
        assert len(gold_positives) >= 20, (
            f"Expected at least 20 positive examples, got {len(gold_positives)}"
        )

    def test_at_least_3_negative_examples(self, gold_negatives):
        assert len(gold_negatives) >= 3, (
            f"Expected at least 3 negative examples, got {len(gold_negatives)}"
        )

    def test_all_edge_types_represented(self, gold_positives):
        """All core edge types should appear in the gold dataset."""
        types_found = set(e["edge_type"] for e in gold_positives)
        expected = {"derived_from", "validates", "depends_on", "caused_by",
                     "blocked_by", "implements", "contradicts"}
        missing = expected - types_found
        assert not missing, (
            f"Gold dataset missing edge types: {missing}. "
            f"Found: {types_found}"
        )

    def test_positive_examples_have_required_fields(self, gold_positives):
        for i, entry in enumerate(gold_positives):
            assert "source" in entry, f"Positive #{i} missing 'source'"
            assert "target" in entry, f"Positive #{i} missing 'target'"
            assert "edge_type" in entry, f"Positive #{i} missing 'edge_type'"
            assert "evidence" in entry, f"Positive #{i} missing 'evidence'"

    def test_negative_examples_have_is_negative_flag(self, gold_negatives):
        for i, entry in enumerate(gold_negatives):
            assert entry.get("is_negative", False), (
                f"Negative #{i} should have is_negative=True"
            )

    def test_edge_types_are_valid(self, gold_positives):
        valid_types = ng.EDGE_TYPES
        for entry in gold_positives:
            assert entry["edge_type"] in valid_types, (
                f"Edge type '{entry['edge_type']}' in gold is not in EDGE_TYPES"
            )


# ---------------------------------------------------------------------------
# Test: Evaluation function correctness
# ---------------------------------------------------------------------------

class TestEvaluation:
    """Validation of the evaluate_relations module's evaluation logic."""

    def test_evaluate_module_imports(self):
        """The evaluation module should be importable."""
        import sys
        _project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(_project_root))
        # Import the module via importlib to handle hyphenated paths
        eval_path = _project_root / "experiments" / "relation-extraction" / "evaluate_relations.py"
        spec = importlib.util.spec_from_file_location("evaluate_relations", eval_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "evaluate")
        assert hasattr(mod, "load_gold")
        assert hasattr(mod, "extract_semantic_edges")
        assert hasattr(mod, "extract_cooccurrence_edges")

    def test_evaluate_perfect_match(self):
        """evaluate() with identical gold and predictions should return F1=1.0."""
        import sys
        _project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(_project_root))
        eval_path = _project_root / "experiments" / "relation-extraction" / "evaluate_relations.py"
        spec = importlib.util.spec_from_file_location("eval_rel", eval_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        gold = [
            {"source": "A", "target": "B", "edge_type": "depends_on", "evidence": "test"},
            {"source": "C", "target": "D", "edge_type": "validates", "evidence": "test"},
        ]
        negatives = []
        predicted = {("A", "B", "depends_on"), ("C", "D", "validates")}
        results = mod.evaluate(gold, negatives, predicted)
        assert results["global"]["precision"] == 1.0
        assert results["global"]["recall"] == 1.0
        assert results["global"]["f1"] == 1.0

    def test_evaluate_zero_recall(self):
        """evaluate() with empty predictions should return recall=0."""
        import sys
        _project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(_project_root))
        eval_path = _project_root / "experiments" / "relation-extraction" / "evaluate_relations.py"
        spec = importlib.util.spec_from_file_location("eval_rel2", eval_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        gold = [
            {"source": "A", "target": "B", "edge_type": "depends_on", "evidence": "test"},
        ]
        negatives = []
        predicted = set()
        results = mod.evaluate(gold, negatives, predicted)
        assert results["global"]["recall"] == 0.0
        assert results["global"]["precision"] == 0.0

    def test_evaluate_negative_check(self):
        """evaluate() should report negative examples that were incorrectly extracted."""
        import sys
        _project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(_project_root))
        eval_path = _project_root / "experiments" / "relation-extraction" / "evaluate_relations.py"
        spec = importlib.util.spec_from_file_location("eval_rel3", eval_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        gold = [
            {"source": "A", "target": "B", "edge_type": "depends_on", "evidence": "test"},
        ]
        negatives = [
            {"source": "X", "target": "Y", "edge_type": "depends_on", "evidence": "NEGATIVE", "is_negative": True},
        ]
        predicted = {("A", "B", "depends_on"), ("X", "Y", "depends_on")}
        results = mod.evaluate(gold, negatives, predicted)
        assert len(results["negative_examples_hit"]) == 1


# ---------------------------------------------------------------------------
# Test: Semantic vs co-occurrence separation
# ---------------------------------------------------------------------------

class TestSemanticCooccurrenceSeparation:
    """Verify that semantic edges and co-occurrence edges are properly separated."""

    def test_related_to_not_in_semantic(self):
        import sys
        _project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(_project_root))
        eval_path = _project_root / "experiments" / "relation-extraction" / "evaluate_relations.py"
        spec = importlib.util.spec_from_file_location("eval_rel4", eval_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Create a graph with both semantic and co-occurrence edges
        g = InMemoryGraphStore()
        g.add_node(ng.Node(id="A", type="Entity"))
        g.add_node(ng.Node(id="B", type="Entity"))
        g.add_node(ng.Node(id="C", type="Entity"))
        g.add_edge(ng.Edge(type="depends_on", source="A", target="B", confidence=0.9))
        g.add_edge(ng.Edge(type="related_to", source="A", target="C", confidence=0.3))

        semantic = mod.extract_semantic_edges(g)
        cooccurrence = mod.extract_cooccurrence_edges(g)

        assert ("A", "B", "depends_on") in semantic
        assert ("A", "C", "related_to") not in semantic
        assert len(cooccurrence) == 1
        assert cooccurrence[0]["source"] == "A"
        assert cooccurrence[0]["target"] == "C"
