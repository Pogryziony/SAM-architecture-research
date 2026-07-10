"""
Test multi-stage entity candidate pipeline for abstract entities.

Verifies that Decision-type and Concept-type entities appear in the
candidate pool when semantic retrieval (Stage 3) is enabled, even when
there is no lexical substring overlap with the question text.
"""

from __future__ import annotations

from nexus.graph import Node, Edge
from nexus.graph.store import InMemoryGraphStore
from stack.encoder.eval_gates import _stage1_exact_name_alias, _stage4_graph_expansion
from nexus.query.parser import _rank_entities, parse_question
from nexus.utils.config import NEXUSConfig


def _make_decision_graph():
    """Build a small graph with abstract Decision and Concept entities."""
    g = InMemoryGraphStore()

    # Decision node: abstract — no surface overlap with questions below
    g.add_node(Node(
        id="Decision_PivotToNEXUS",
        type="Decision",
        properties={
            "description": "Pivot from SAM to NEXUS graph reasoning",
            "rationale": "Selector bottleneck is structural",
        },
        aliases=["nexus system", "nexus approach", "graph reasoning", "pivot to graph"],
    ))

    # Concept node: abstract
    g.add_node(Node(
        id="Concept_OracleMemory",
        type="Concept",
        properties={
            "description": "Oracle memory proves SAM core can use external memory",
            "key_finding": "99.87% accuracy with oracle",
        },
        aliases=["oracle memory concept", "oracle experiment"],
    ))

    # Regular Experiment entity with surface overlap
    g.add_node(Node(
        id="Exp_0_6_Validation",
        type="Experiment",
        properties={
            "description": "Validation experiment for SAM core architecture",
            "key_finding": "oracle memory achieves 99.87%",
        },
        aliases=["validation experiment"],
    ))

    # Edges: link Decision to Concept
    g.add_edge(Edge(type="depends_on", source="Decision_PivotToNEXUS", target="Concept_OracleMemory"))
    g.add_edge(Edge(type="derived_from", source="Concept_OracleMemory", target="Exp_0_6_Validation"))

    return g


class TestAbstractEntityCoverage:
    """Verify Decision and Concept entities reach the candidate pool."""

    QUESTIONS = [
        {
            "question": "What is the significance of the oracle memory experiment achieving 99.87% accuracy?",
            "gt_entity": "Decision_PivotToNEXUS",
        },
        {
            "question": "What is the significance of the chain-set BCE retriever achieving 100% all_required@32?",
            "gt_entity": "Decision_PivotToNEXUS",
        },
        {
            "question": "What is the significance of the pivot from SAM to NEXUS?",
            "gt_entity": "Decision_PivotToNEXUS",
        },
        {
            "question": "What is the significance of the oracle memory concept?",
            "gt_entity": "Concept_OracleMemory",
        },
        {
            "question": "What is the significance of the architecture validation findings?",
            "gt_entity": "Concept_OracleMemory",
        },
    ]

    def test_stage1_exact_name_alias_matches_direct_aliases(self):
        """Stage 1 should catch entities whose aliases appear as substrings."""
        g = _make_decision_graph()

        # Question with explicit "nexus" substring → Decision_PivotToNEXUS alias match
        q_with_nexus = "What is the significance of the pivot from SAM to NEXUS?"
        s1 = _stage1_exact_name_alias(q_with_nexus, g)
        assert "Decision_PivotToNEXUS" in s1, (
            f"Stage 1 should match 'Decision_PivotToNEXUS' via alias 'nexus system' "
            f"in question containing 'nexus'. Got: {s1}"
        )

    def test_stage1_does_not_match_without_surface_overlap(self):
        """Stage 1 should NOT catch entities with zero substring overlap."""
        g = _make_decision_graph()

        q_no_overlap = "What is the significance of the oracle memory experiment achieving 99.87% accuracy?"
        s1 = _stage1_exact_name_alias(q_no_overlap, g)
        assert "Decision_PivotToNEXUS" not in s1, (
            f"Stage 1 should NOT match 'Decision_PivotToNEXUS' without surface overlap. Got: {s1}"
        )

    def test_semantic_stage_adds_missing_abstract_entities(self):
        """Semantic retrieval (Stage 3) should find Decision having no surface overlap."""
        from nexus.query.embedding_resolver import NodeEmbeddingIndex

        g = _make_decision_graph()
        emb_idx = NodeEmbeddingIndex()
        emb_idx.build_index(g)

        q = "What is the significance of the oracle memory experiment achieving 99.87% accuracy?"

        # Stage 1: exact/alias — should NOT match Decision_PivotToNEXUS
        s1 = _stage1_exact_name_alias(q, g)

        # Stage 3: semantic — should find it via embedding similarity
        s3 = [eid for eid, _ in emb_idx.query(q, top_k=10)]

        # Combined pool (before expansion)
        seen: set[str] = set()
        combined: list[str] = []
        for c in s1 + s3:
            if c not in seen:
                seen.add(c)
                combined.append(c)

        assert "Decision_PivotToNEXUS" in combined, (
            f"Stage 3 semantic retrieval should add 'Decision_PivotToNEXUS' "
            f"to the candidate pool even without surface overlap. "
            f"Stage 1: {s1}, Stage 3 top-3: {s3[:3]}"
        )

    def test_all_five_questions_gt_in_semantic_pool(self):
        """Every abstract-concept question should have GT entity in the combined pool."""
        from nexus.query.embedding_resolver import NodeEmbeddingIndex

        g = _make_decision_graph()
        emb_idx = NodeEmbeddingIndex()
        emb_idx.build_index(g)

        for entry in self.QUESTIONS:
            q = entry["question"]
            gt = entry["gt_entity"]

            s1 = _stage1_exact_name_alias(q, g)
            s3 = [eid for eid, _ in emb_idx.query(q, top_k=10)]

            seen: set[str] = set()
            combined: list[str] = []
            for c in s1 + s3:
                if c not in seen:
                    seen.add(c)
                    combined.append(c)

            assert gt in combined, (
                f"GT entity '{gt}' NOT found in combined pool for question:\n"
                f"  '{q}'\n"
                f"  Stage 1: {s1}\n"
                f"  Stage 3 (top-3): {s3[:3]}"
            )

    def test_stage4_graph_expansion_adds_neighbors(self):
        """Stage 4 should add 1-hop neighbors to the candidate pool."""
        g = _make_decision_graph()

        # Start with only Decision_PivotToNEXUS
        candidates = ["Decision_PivotToNEXUS"]
        expanded = _stage4_graph_expansion(candidates, g)

        # Concept_OracleMemory is a 1-hop neighbor (depends_on edge)
        assert "Concept_OracleMemory" in expanded, (
            f"Stage 4 should add Concept_OracleMemory as neighbor. Got: {expanded}"
        )

    def test_encoder_selected_entity_is_not_displaced_by_final_cap(self):
        """The full parser must preserve a selected encoder baseline entity."""
        g = _make_decision_graph()
        config = NEXUSConfig(max_entry_nodes=1)
        ranked = _rank_entities(
            g,
            ["Exp_0_6_Validation", "Decision_PivotToNEXUS"],
            question="What is the significance of the result?",
            config=config,
            protected_ids={"Decision_PivotToNEXUS"},
        )
        assert ranked == ["Decision_PivotToNEXUS"]

    def test_final_pipeline_recall_is_not_lower_than_encoder_selection(self):
        """A selected encoder entity must survive the complete parser handoff."""
        class FakeEncoder:
            def predict(self, question, entity_threshold, entity_candidates, entity_descriptions):
                selected = "Decision_PivotToNEXUS"
                return {
                    "entity_ids": [selected],
                    "entity_scores": [(selected, 0.99)],
                    "candidate_scores": {selected: 0.99},
                    "intent": "factual_lookup",
                    "category": "factual",
                }

        g = _make_decision_graph()
        config = NEXUSConfig(max_entry_nodes=1, enable_associative_encoder=True)
        result = parse_question(
            "What is the significance of the oracle memory experiment?",
            g,
            config=config,
            encoder_model=FakeEncoder(),
            encoder_entity_threshold=0.55,
        )
        assert result.entity_ids[:1] == ["Decision_PivotToNEXUS"]

    def test_stage4_respects_max_neighbors(self):
        """Stage 4 should respect the max_neighbors limit."""
        g = _make_decision_graph()

        candidates = ["Decision_PivotToNEXUS"]
        expanded = _stage4_graph_expansion(candidates, g, max_neighbors=0)
        assert len(expanded) == 0, f"With max_neighbors=0, should return empty. Got: {expanded}"

        expanded_1 = _stage4_graph_expansion(candidates, g, max_neighbors=1)
        assert len(expanded_1) <= 1, f"With max_neighbors=1, should return at most 1. Got: {len(expanded_1)}"
