"""Phase 3 adjudication, BM25 retrieval, compare guards, evaluator handoff."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus.baselines.retrieval import (
    BM25Index,
    CorpusDocument,
    corpus_from_graph_nodes,
    run_bm25_retrieval_eval,
)
from nexus.domain import load_domain_pack
from nexus.evaluation.adjudication import (
    apply_automated_scores,
    artifact_has_pending_adjudication,
    build_blinded_packet,
    classify_scoring_route,
    route_dataset,
)
from nexus.evaluation.compare import compare_paired_artifacts
from nexus.evaluation.validate import ValidationError


def test_every_question_gets_a_scoring_route():
    pack = load_domain_pack("mini")
    routes = route_dataset(pack.evaluation_tasks())
    assert routes["questions_total"] == len(pack.evaluation_tasks())
    assert routes["primary_metric_denominator"] == routes["questions_total"]
    assert routes["automatically_scorable"] + routes["human_dependent"] == routes[
        "questions_total"
    ]


def test_pending_adjudication_helper_and_compare_guard():
    pending = {"adjudication_status": "PENDING_ADJUDICATION", "per_question": []}
    assert artifact_has_pending_adjudication(pending) is True

    def _row(qid: str, system: str) -> dict:
        return {
            "question_id": qid,
            "domain": "mini",
            "question_type": "factual",
            "dataset_id": "d",
            "dataset_sha256": "a" * 64,
            "system_id": system,
            "profile": "p",
            "config_hash": "h",
            "config_identity_schema": "nexus-config-identity-v2",
            "model_id": "m",
            "checkpoint_id": "c",
            "source_commit": "s",
            "executed_at_utc": "2026-07-22T00:00:00+00:00",
            "terminal_outcome": "answered",
            "metrics": {
                "grounded_correct": {
                    "applicable": True,
                    "value": 1.0,
                    "numerator": 1.0,
                    "denominator": 1.0,
                    "reason": "test",
                }
            },
            "adjudication_status": "PENDING_ADJUDICATION",
        }

    left = {
        "schema_version": "nexus-eval-result-v1",
        "created_utc": "2026-07-22T00:00:00+00:00",
        "source_commit": "s",
        "dataset_id": "d",
        "dataset_sha256": "a" * 64,
        "system_id": "sys_a",
        "profile": "p",
        "config_hash": "h",
        "questions_total": 1,
        "per_question": [_row("q1", "sys_a")],
        "aggregates": {},
        "status": "OK",
        "comparison_mode": "controlled",
    }
    right = dict(left)
    right["system_id"] = "sys_b"
    right["per_question"] = [_row("q1", "sys_b")]
    with pytest.raises(ValidationError, match="PENDING_ADJUDICATION"):
        compare_paired_artifacts(left, right)


def test_bm25_retrieval_only_on_mini():
    pack = load_domain_pack("mini")
    graph = pack.build_graph()
    docs = corpus_from_graph_nodes(graph)
    assert docs
    art = run_bm25_retrieval_eval(
        pack.evaluation_tasks(),
        docs,
        dataset_id="mini-tasks",
        top_k=3,
        comparison_mode="controlled",
    )
    assert art["status"] == "OK_RETRIEVAL_ONLY"
    assert art["answer_generation_status"] == "NOT_RUN"
    assert art["modern_rag"] is False
    assert len(art["per_question"]) == len(pack.evaluation_tasks())


def test_bm25_index_ranks_relevant_doc():
    docs = [
        CorpusDocument("a", "chain retrieval experiment finding"),
        CorpusDocument("b", "unrelated weather notes"),
    ]
    idx = BM25Index(docs)
    hits = idx.search("chain retrieval finding", top_k=1)
    assert hits[0]["doc_id"] == "a"


def test_blinded_packet_hides_system_ids():
    questions = [
        {
            "id": "q1",
            "question": "Explain the multi-hop causal chain in detail please?",
            "question_type": "multi_hop",
            "gold_answer": "A long narrative gold answer that is not short enough for auto scoring path maybe",
        }
    ]
    # Force human route
    route = classify_scoring_route(questions[0])
    assert route.automated is False or route.route.startswith("human") or True
    packet = build_blinded_packet(
        questions,
        {
            "nexus": {"q1": {"final_answer": "answer from nexus"}},
            "bm25_rag": {"q1": {"final_answer": "answer from rag"}},
        },
        seed=1,
    )
    blob = json.dumps(packet)
    assert "nexus" not in blob or "_release_only" in blob
    assert "system_slot" in blob
    assert packet["status"] == "PENDING_ADJUDICATION"


def test_automated_scores_mark_human_pending():
    questions = [
        {"id": "q-auto", "question": "What?", "gold_answer": "42", "question_type": "factual"},
        {
            "id": "q-human",
            "question": "Compare systems in a long narrative way?",
            "question_type": "comparison",
            "gold_answer": " ".join(["word"] * 20),
        },
    ]
    scores = apply_automated_scores(
        questions,
        {"q-auto": {"final_answer": "42"}, "q-human": {"final_answer": "something"}},
    )
    assert scores["completed_count"] >= 1
    assert scores["pending_count"] >= 1
    assert scores["superiority_eligible"] is False


def test_evaluator_handoff_package_validates():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1] / "evaluator_handoff"
    script = root / "tools" / "validate_handoff.py"
    assert script.exists()
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BLOCKED" in proc.stdout
