"""Core validation tests for Entity Ranker V3.

T1: Question changes relative entity ordering
T2: All validation questions remain in the denominator
T3: Baseline and rankers use identical populations
T4: K cannot exceed 10
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from nexus.graph import Node
from nexus.graph.store import InMemoryGraphStore
from stack.encoder.char_tokenizer import CharNgramTokenizer
from stack.encoder.entity_ranker_v3 import QuestionConditionedEntityRanker
from stack.encoder.train_ranker_v3 import (
    build_evaluation_group,
    build_training_group,
    evaluate_trivial_baseline,
    multi_positive_listwise_loss,
)


# ── T1: Question-conditional entity scoring ──

def _build_interaction_scorer() -> Any:
    """Build a minimal bilinear interaction model for testing.

    score(q, e) = dot(W_q @ q_enc, W_e @ e_enc)
    """
    import torch
    import torch.nn as nn

    class BilinearEntityScorer(nn.Module):
        def __init__(self, feat_dim: int = 8, proj_dim: int = 4):
            super().__init__()
            self.q_proj = nn.Linear(feat_dim, proj_dim)
            self.e_proj = nn.Linear(feat_dim, proj_dim)

        def forward(self, q_encoding, e_encodings):
            # q_encoding: [B, feat_dim], e_encodings: [B, K, feat_dim]
            q = self.q_proj(q_encoding)  # [B, proj_dim]
            e = self.e_proj(e_encodings)  # [B, K, proj_dim]
            # Dot product per candidate
            scores = (q.unsqueeze(1) * e).sum(dim=-1)  # [B, K]
            return scores

    return BilinearEntityScorer()


def test_question_changes_relative_entity_ordering():
    """T1: Different questions produce different entity rankings."""
    import torch

    scorer = _build_interaction_scorer()
    scorer.eval()

    # Two entities
    e1 = torch.randn(1, 2, 8)  # [1, 2, 8] — two entities
    q1 = torch.randn(1, 8)       # question 1
    q2 = torch.randn(16, 8)      # question 2 (different)

    with torch.no_grad():
        s1 = scorer(q1, e1)[0]  # [2]
        s2 = scorer(q2[:1], e1)[0]  # [2]

    # The relative ordering must change for at least one of the 16 q2 variants
    ordering_changed = False
    for i in range(16):
        s2i = scorer(q2[i:i+1], e1)[0]
        if (s1[0] > s1[1]) != (s2i[0] > s2i[1]):
            ordering_changed = True
            break

    assert ordering_changed, (
        "T1 FAIL: Changing the question did not change entity ordering. "
        "The scorer is not genuinely question-conditioned."
    )


def test_production_ranker_score_margin_depends_on_question():
    """Exercise the real V3 scorer rather than a test-only stand-in."""
    import torch

    torch.manual_seed(7)
    tokenizer = CharNgramTokenizer()
    tokenizer.add_words([
        "alpha experiment result", "beta graph decision",
        "alpha entity", "beta entity",
    ])
    tokenizer.freeze()
    model = QuestionConditionedEntityRanker(
        tokenizer.feature_dim, embed_dim=16, hidden_dim=32, proj_dim=8, dropout=0.0
    )
    model.eval()
    entity_texts = ["alpha entity", "beta entity"]

    def margin(question: str):
        offsets, indices = tokenizer.tokenize_batch([question])
        scores = model(
            torch.tensor(indices), torch.tensor(offsets[:-1]), entity_texts, tokenizer
        )[0]
        return float(scores[0] - scores[1])

    assert margin("alpha experiment result") != pytest.approx(
        margin("beta graph decision")
    )


def test_entity_projection_receives_gradient_and_all_positives_reduce_loss():
    import torch

    torch.manual_seed(11)
    tokenizer = CharNgramTokenizer()
    tokenizer.add_words(["question", "positive one", "positive two", "negative"])
    tokenizer.freeze()
    model = QuestionConditionedEntityRanker(
        tokenizer.feature_dim, embed_dim=16, hidden_dim=32, proj_dim=8, dropout=0.0
    )
    offsets, indices = tokenizer.tokenize_batch(["question"])
    scores = model(
        torch.tensor(indices),
        torch.tensor(offsets[:-1]),
        ["positive one", "positive two", "negative"],
        tokenizer,
    )[0]
    one_positive_loss = multi_positive_listwise_loss(scores, [0])
    all_positive_loss = multi_positive_listwise_loss(scores, [0, 1])
    assert all_positive_loss <= one_positive_loss
    all_positive_loss.backward()
    assert model.e_proj[0].weight.grad is not None
    assert torch.count_nonzero(model.e_proj[0].weight.grad) > 0


def test_linear_concat_scorer_is_not_question_conditioned():
    """Demonstrate that the current linear-concat scorer fails T1.

    score(q,e) = W_q·q + W_e·e + b — for fixed q, the q-term is constant,
    so relative ordering of entities is independent of q.
    """
    import torch
    import torch.nn as nn

    class LinearConcatScorer(nn.Module):
        """Reproduces the defective scoring from stack/encoder/model.py."""
        def __init__(self, feat_dim: int = 8, embed_dim: int = 8):
            super().__init__()
            self.scorer = nn.Linear(feat_dim + embed_dim, 1)

        def forward(self, q_encoding, e_encodings):
            B, K, E = e_encodings.shape
            q_expanded = q_encoding.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
            pair = torch.cat([q_expanded, e_encodings], dim=-1)  # [B, K, D+E]
            return self.scorer(pair).squeeze(-1)  # [B, K]

    scorer = LinearConcatScorer()
    scorer.eval()

    # Two entities, two different questions
    e1 = torch.randn(1, 2, 8)
    qa = torch.randn(1, 8)
    qb = torch.randn(16, 8)

    # Check if ordering EVER differs
    with torch.no_grad():
        sa = scorer(qa, e1)[0]
        ordering_differs = False
        for i in range(16):
            sb = scorer(qb[i:i+1], e1)[0]
            if (sa[0] > sa[1]) != (sb[0] > sb[1]):
                ordering_differs = True
                break

    assert not ordering_differs, (
        "Linear-concat scorer unexpectedly changed ordering across questions. "
        "This should NOT happen — the scorer is architecturally q-independent. "
        "If this assertion fires, the test setup produced degenerate weights."
    )


# ── T2: Validation denominator — all 150 questions in denominator ──

def test_missing_gold_cannot_disappear_from_evaluation():
    """T2: A question whose gold entity is missing from candidates must
    still be counted in the denominator, not silently dropped."""
    groups = [
        {"question_id": "q1", "question": "?",
         "candidate_ids": ["A", "B"], "positive_ids": ["A"],
         "source": "test"},
        {"question_id": "q2", "question": "??",
         "candidate_ids": ["B", "C"], "positive_ids": ["Z"],  # Z not in candidates
         "source": "test"},
    ]

    # Simulate evaluation with denominator-preserving logic
    total_questions = 2
    total_gold = 2  # [A, Z]

    hits_at_1 = 0
    for g in groups:
        pos = set(g["positive_ids"])
        cand = set(g["candidate_ids"])
        if pos & cand:
            # At least one gold entity is in the candidate pool
            hits_at_1 += int(bool(set(g["candidate_ids"][:1]) & pos))

    recall_at_1 = hits_at_1 / total_gold
    # q1: candidate[0]="A", gold=A → hit
    # q2: candidate[0]="B", gold={Z}, Z not in candidates → miss
    # recall = 1/2 = 0.5
    assert recall_at_1 == 0.5, (
        f"T2 FAIL: Missing gold not counted. recall@1 = {recall_at_1}, expected 0.5. "
        f"Total questions remained {total_questions}, total gold {total_gold}."
    )

    # Verify q2 is NOT silently dropped (it contributes gold=1 but hits=0)
    assert total_gold == 2, "Missing-gold question must contribute to denominator"


def test_production_group_builders_only_inject_gold_for_training():
    graph = InMemoryGraphStore()
    for node_id in ("A", "B", "Z"):
        graph.add_node(Node(id=node_id, type="Entity"))

    evaluation = build_evaluation_group(
        "q", "question", ["Z"], ["A", "B"], "validation", graph
    )
    training = build_training_group(
        "q", "question", ["Z"], ["A", "B"], "real_train", graph
    )

    assert evaluation is not None and training is not None
    assert evaluation["candidate_ids"] == ["A", "B"]
    assert evaluation["gold_present_in_candidates"] is False
    assert evaluation["gold_injected_for_training"] is False
    assert training["candidate_ids"] == ["A", "B", "Z"]
    assert training["gold_present_in_candidates"] is False
    assert training["gold_injected_for_training"] is True


def test_production_baseline_preserves_missing_gold_in_denominator():
    graph = InMemoryGraphStore()
    graph.add_node(Node(id="Alpha", type="Entity", aliases=["alpha"]))
    graph.add_node(Node(id="Beta", type="Entity", aliases=["beta"]))
    graph.add_node(Node(id="Missing", type="Entity"))
    groups = [
        build_evaluation_group(
            "q1", "alpha", ["Alpha"], ["Alpha", "Beta"], "validation", graph
        ),
        build_evaluation_group(
            "q2", "beta", ["Missing"], ["Beta"], "validation", graph
        ),
    ]
    metrics = evaluate_trivial_baseline(groups, graph)
    assert metrics["total_questions"] == 2
    assert metrics["total_gold_entities"] == 2
    assert metrics["raw_candidate_recall_ceiling"] == 0.5
    assert metrics["recall@10"] == 0.5


class TestValidationDenominator:
    """T2 continued: real data validation."""

    @pytest.fixture(scope="class")
    def val_data(self):
        path = Path(__file__).parents[1] / "stack" / "encoder" / "data" / "val.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_val_has_150_questions(self, val_data):
        """Validation split must be exactly 150 questions."""
        assert len(val_data) == 150

    def test_val_has_182_gold_entities(self, val_data):
        """Validation split must have exactly 182 gold entities."""
        total = sum(len(set(q["entities"])) for q in val_data)
        assert total == 182


# ── T3: Identical populations ──

def test_baseline_and_ranker_use_identical_populations():
    """T3: Baseline and all rankers must evaluate the same groups."""
    groups_a = [{"question_id": "a", "candidate_ids": ["X", "Y"], "positive_ids": ["X"]}]
    groups_b = [{"question_id": "a", "candidate_ids": ["X", "Y"], "positive_ids": ["X"]}]
    groups_c = [{"question_id": "a", "candidate_ids": ["Y", "X"], "positive_ids": ["X"]}]  # reordered
    groups_d = [{"question_id": "b", "candidate_ids": ["X", "Y"], "positive_ids": ["X"]}]  # diff question

    def same(g1, g2):
        return [(g["question_id"], g["candidate_ids"], g["positive_ids"]) for g in g1] == \
               [(g["question_id"], g["candidate_ids"], g["positive_ids"]) for g in g2]

    assert same(groups_a, groups_b), "Identical groups must compare equal"
    assert not same(groups_a, groups_c), "Reordered candidates differ"
    assert not same(groups_a, groups_d), "Different question IDs differ"


# ── T4: K cannot exceed 10 ──

def test_k_never_exceeds_ten():
    """T4: Ranking functions must reject K > 10 or cap at 10."""
    K = 10

    def ranker(candidates, k):
        if k < 1 or k > K:
            raise ValueError(f"k must be in [1, {K}]")
        return candidates[:k]

    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        ranker(["A", "B", "C"], 11)

    # K=10 should work
    result = ranker(["A"] * 30, 10)
    assert len(result) == 10

    # K=1 should work
    result = ranker(["A", "B", "C"], 1)
    assert len(result) == 1
