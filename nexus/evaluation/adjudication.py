"""Correctness adjudication routing for internal oracle questions.

Separates automatically scorable routes from human-dependent adjudication.
Pending human scores cannot produce superiority verdicts.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from nexus.evaluation.metrics import compute_grounded_correct


AUTO_ROUTES = frozenset(
    {
        "exact_match",
        "key_fact_fuzzy",
        "structured_gold",
        "should_abstain",
        "relation_presence",
    }
)
HUMAN_ROUTES = frozenset(
    {
        "human_blinded",
        "human_temporal",
        "human_comparison_narrative",
        "human_multihop_narrative",
    }
)


@dataclass(frozen=True)
class ScoringRoute:
    question_id: str
    route: str
    automated: bool
    reason: str
    primary_metric: str = "grounded_correct"


@dataclass
class AdjudicationDimensionScores:
    conclusion_correctness: float | None = None
    material_claim_support: float | None = None
    citation_entailment: float | None = None
    completeness: float | None = None
    temporal_correctness: float | None = None
    unsupported_claims: float | None = None
    abstention_appropriate: float | None = None


@dataclass
class AdjudicationPacketItem:
    packet_item_id: str
    question_id: str
    question: str
    system_slot: str  # anonymized A/B/C
    candidate_answer: str
    permitted_reference_evidence: list[str] = field(default_factory=list)
    dimensions: list[str] = field(
        default_factory=lambda: [
            "conclusion_correctness",
            "material_claim_support",
            "citation_entailment",
            "completeness",
            "temporal_correctness",
            "unsupported_claims",
            "abstention_appropriate",
        ]
    )


def classify_scoring_route(record: Mapping[str, Any]) -> ScoringRoute:
    """Assign a scoring route for one oracle/eval question record."""
    qid = str(record.get("id") or record.get("question_id") or "")
    qtype = str(
        record.get("question_type")
        or record.get("category")
        or record.get("type")
        or ""
    ).casefold()
    gold = record.get("gold_answer")
    gold_words = len(str(gold or "").split())
    should_abstain = bool(record.get("should_abstain", False))
    structured = record.get("structured_gold") or record.get("gold_structured")

    if should_abstain:
        return ScoringRoute(qid, "should_abstain", True, "gold marks abstention")
    if structured:
        return ScoringRoute(qid, "structured_gold", True, "structured gold present")
    if qtype in {"temporal", "bitemporal", "as_of"}:
        return ScoringRoute(
            qid,
            "human_temporal",
            False,
            "temporal narrative not safely auto-scored alone",
        )
    if qtype in {
        "comparison",
        "comparative",
        "causal",
        "multi_hop",
        "multi-hop",
        "multihop",
        "two_hop",
        "three_hop",
        "diagnostic",
        "contradiction",
    }:
        if gold and gold_words <= 12:
            return ScoringRoute(
                qid, "key_fact_fuzzy", True, "short gold allows key-fact proxy"
            )
        return ScoringRoute(
            qid,
            "human_blinded",
            False,
            f"narrative type={qtype or 'unknown'} needs human adjudication",
        )
    if qtype in {
        "factual",
        "factual_lookup",
        "direct_lookup",
        "metric",
        "relation",
        "negative_relation",
        "yes_no",
        "no_answer",
    }:
        # Long factual gold remains fuzzy-proxy but still automated for
        # internal regression; human packet can still be requested later.
        return ScoringRoute(
            qid,
            "key_fact_fuzzy",
            True,
            f"structured-friendly type={qtype or 'factual'}",
        )
    if gold is not None and str(gold).strip() and gold_words <= 12:
        return ScoringRoute(
            qid, "key_fact_fuzzy", True, "short gold_answer; key-fact proxy route"
        )
    if gold is not None and str(gold).strip():
        return ScoringRoute(
            qid,
            "human_blinded",
            False,
            "long gold without safe typed route; human adjudication",
        )
    return ScoringRoute(
        qid,
        "human_blinded",
        False,
        "no safe automated gold representation",
    )


def route_dataset(questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build machine-readable routing table for a question set."""
    routes = [classify_scoring_route(q) for q in questions]
    auto = [r for r in routes if r.automated]
    human = [r for r in routes if not r.automated]
    return {
        "schema_version": "nexus-adjudication-routes-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "questions_total": len(routes),
        "automatically_scorable": len(auto),
        "human_dependent": len(human),
        "primary_metric_denominator": len(routes),
        "routes": [asdict(r) for r in routes],
        "status": "ROUTES_DEFINED",
        "note": (
            "Human-dependent metrics remain PENDING_ADJUDICATION until two "
            "independent annotators complete the blinded packet."
        ),
    }


def apply_automated_scores(
    questions: Sequence[Mapping[str, Any]],
    answers_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score automatically routable questions; leave others pending."""
    rows: list[dict[str, Any]] = []
    for q in questions:
        route = classify_scoring_route(q)
        qid = str(q.get("id") or q.get("question_id") or "")
        ans_row = answers_by_id.get(qid) or {}
        answer = str(ans_row.get("final_answer") or ans_row.get("answer") or "")
        should_abstain = bool(q.get("should_abstain", False))
        gold = str(q.get("gold_answer") or "")
        status = "PENDING_ADJUDICATION"
        grounded = None
        if route.automated:
            answer_correct = None
            if route.route == "should_abstain":
                answer_correct = "insufficient" in answer.casefold() or not answer.strip()
            elif gold:
                try:
                    from benchmarks.scoring import compute_fact_score

                    fuzzy = float(
                        compute_fact_score(answer, gold).get("fuzzy_accuracy") or 0.0
                    )
                    answer_correct = fuzzy >= 0.5
                except Exception:
                    answer_correct = (
                        gold.casefold() in answer.casefold() if gold else None
                    )
            grounded = compute_grounded_correct(
                answer=answer,
                gold_answer=gold,
                should_abstain=should_abstain,
                answer_correct=answer_correct,
                material_claims_supported=None,
                citations_entail=None,
                temporal_ok=None,
            )
            status = "SCORED_AUTOMATED"
        rows.append(
            {
                "question_id": qid,
                "route": asdict(route),
                "status": status,
                "grounded_correct": None if grounded is None else grounded.to_dict(),
                "llm_judge_diagnostic_only": None,
            }
        )
    completed = sum(1 for r in rows if r["status"] == "SCORED_AUTOMATED")
    pending = sum(1 for r in rows if r["status"] == "PENDING_ADJUDICATION")
    return {
        "schema_version": "nexus-adjudication-scores-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "questions_total": len(rows),
        "completed_count": completed,
        "pending_count": pending,
        "primary_metric_denominator": len(rows),
        "agreement": {
            "annotators": 0,
            "cohen_kappa": None,
            "status": "NOT_RUN",
            "reason": "no human annotators available in this phase",
        },
        "rows": rows,
        "superiority_eligible": False,
        "reason_not_eligible": (
            "PENDING_ADJUDICATION remains for human-dependent questions; "
            "do not issue superiority verdicts"
            if pending
            else "automated-only subset; full-oracle superiority still requires complete protocol"
        ),
    }


def build_blinded_packet(
    questions: Sequence[Mapping[str, Any]],
    system_answers: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    seed: int = 0,
    permitted_evidence_by_qid: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Build a blinded human adjudication packet.

    ``system_answers`` maps system_id -> question_id -> {final_answer,...}.
    System IDs are anonymized; answer order is randomized per question.
    """
    rng = random.Random(seed)
    system_ids = sorted(system_answers)
    anon_map = {
        sid: chr(ord("A") + i) for i, sid in enumerate(system_ids)
    }
    reverse = {v: k for k, v in anon_map.items()}
    items: list[dict[str, Any]] = []
    for q in questions:
        route = classify_scoring_route(q)
        if route.automated:
            continue
        qid = str(q.get("id") or q.get("question_id") or "")
        slots = list(anon_map.values())
        rng.shuffle(slots)
        evidence = list((permitted_evidence_by_qid or {}).get(qid) or [])
        for slot in slots:
            real_system = reverse[slot]
            ans = (system_answers.get(real_system) or {}).get(qid) or {}
            item = AdjudicationPacketItem(
                packet_item_id=hashlib.sha256(
                    f"{qid}:{slot}:{seed}".encode("utf-8")
                ).hexdigest()[:16],
                question_id=qid,
                question=str(q.get("question") or ""),
                system_slot=slot,
                candidate_answer=str(
                    ans.get("final_answer") or ans.get("answer") or ""
                ),
                permitted_reference_evidence=evidence,
            )
            items.append(asdict(item))
    return {
        "schema_version": "nexus-adjudication-packet-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "anonymization": {
            "mapping_held_separately": True,
            "note": "Do not disclose system identity to annotators",
        },
        "annotator_requirements": {
            "independent_annotators": 2,
            "disagreement_tracking": True,
            "third_adjudicator_or_resolution_process": True,
            "inter_annotator_agreement_required": True,
        },
        "status": "PENDING_ADJUDICATION",
        "items": items,
        "item_count": len(items),
        # Mapping stored hashed-side for release tooling; not for annotators.
        "_release_only_system_map_sha256": hashlib.sha256(
            json.dumps(anon_map, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def artifact_has_pending_adjudication(artifact: Mapping[str, Any]) -> bool:
    """True when any primary metric depends on unfinished human adjudication."""
    if artifact.get("adjudication_status") == "PENDING_ADJUDICATION":
        return True
    for row in artifact.get("per_question") or []:
        metrics = row.get("metrics") or {}
        gc = metrics.get("grounded_correct") or {}
        if gc.get("reason") == "pending_adjudication":
            return True
        if row.get("adjudication_status") == "PENDING_ADJUDICATION":
            return True
    return False
