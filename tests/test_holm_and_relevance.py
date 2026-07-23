"""Holm correction, relevance mapping, agreement, packet evidence."""

from __future__ import annotations

import pytest

from nexus.evaluation.agreement import cohen_kappa
from nexus.evaluation.multiple_comparison import apply_holm_to_comparisons, holm_adjust
from nexus.evaluation.packet_evidence import build_permitted_evidence_map
from nexus.evaluation.relevance import build_relevance_for_question, build_relevance_table
from nexus.evaluation.adjudication_io import export_dual_packets


def test_holm_adjust_monotone_and_stricter():
    raw = [0.01, 0.04, 0.03]
    adj = holm_adjust(raw)
    # sorted p: 0.01, 0.03, 0.04 → adj 0.03, 0.06, 0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)


def test_apply_holm_family_updates_verdicts():
    comps = [
        {
            "mcnemar": {"p_value": 0.01},
            "bootstrap": {"ci_low": 0.02, "ci_high": 0.1},
        },
        {
            "mcnemar": {"p_value": 0.04},
            "bootstrap": {"ci_low": 0.01, "ci_high": 0.05},
        },
    ]
    out = apply_holm_to_comparisons(comps)
    assert out[0]["multiple_comparison"]["n_tests"] == 2
    assert out[0]["multiple_comparison"]["adjusted_p_value"] == pytest.approx(0.02)


def test_cohen_kappa_perfect_and_chance():
    perfect = cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0])
    assert perfect["kappa"] == pytest.approx(1.0)
    empty = cohen_kappa([], [])
    assert empty["status"] == "NOT_RUN"


def test_relevance_nonzero_fixture():
    q = {
        "id": "q_rel",
        "question": "What about NEXUS traversal?",
        "gold_answer": "NEXUS uses bounded graph traversal",
        "gold_entities": ["nexus_system"],
    }
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "The nexus_system performs bounded graph traversal for evidence.",
            "source_path": "docs/graph-memory.md",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "text": "Unrelated cooking recipe with pasta.",
            "source_path": "other.md",
        },
    ]
    m = build_relevance_for_question(q, chunks)
    assert "c1" in m.relevant_chunk_ids
    table = build_relevance_table([q], {"chunks": chunks, "corpus_sha256": "x" * 64})
    assert table["questions_with_relevant_chunks"] == 1


def test_dual_export_requires_evidence(tmp_path):
    questions = [
        {
            "id": "q1",
            "question": "Compare two long narrative systems in detail please?",
            "category": "comparative",
            "gold_answer": " ".join(["word"] * 20),
        }
    ]
    with pytest.raises(ValueError, match="lacks structured evidence"):
        export_dual_packets(
            questions,
            {
                "nexus": {"q1": {"final_answer": "n"}},
                "qwen": {"q1": {"final_answer": "q"}},
            },
            tmp_path,
            require_evidence=True,
        )


def test_dual_export_with_evidence(tmp_path):
    questions = [
        {
            "id": "q1",
            "question": "Compare two long narrative systems in detail please?",
            "category": "comparative",
            "gold_answer": " ".join(["word"] * 20),
            "gold_entities": ["sys_a"],
        }
    ]
    answers = {
        "nexus": {
            "q1": {
                "final_answer": "n",
                "citations": ["chunk:1"],
                "structured_evidence": {"facts": ["a"]},
            }
        },
        "qwen": {
            "q1": {
                "final_answer": "q",
                "retrieved_documents": ["chunk:2"],
            }
        },
    }
    manifest = export_dual_packets(questions, answers, tmp_path, require_evidence=True)
    packet = (tmp_path / "annotator_A_packet.json").read_text(encoding="utf-8")
    assert "citation:" in packet or "retrieved:" in packet
    assert manifest["item_count_A"] >= 1
    ev = build_permitted_evidence_map(questions, answers)
    assert ev["q1"]
