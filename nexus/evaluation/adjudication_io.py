"""Import/export workflow for dual-human adjudication packets."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from nexus.evaluation.adjudication import build_blinded_packet, route_dataset


DIMENSIONS = (
    "conclusion_correctness",
    "material_claim_support",
    "citation_entailment",
    "completeness",
    "temporal_correctness",
    "unsupported_claims",
    "abstention_appropriate",
)


def export_dual_packets(
    questions: Sequence[Mapping[str, Any]],
    system_answers: Mapping[str, Mapping[str, Mapping[str, Any]]],
    out_dir: Path,
    *,
    seed_a: int = 11,
    seed_b: int = 29,
) -> dict[str, Any]:
    """Write two independently randomized blinded packets for annotators A/B."""
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_a = build_blinded_packet(questions, system_answers, seed=seed_a)
    packet_b = build_blinded_packet(questions, system_answers, seed=seed_b)
    path_a = out_dir / "annotator_A_packet.json"
    path_b = out_dir / "annotator_B_packet.json"
    for path, packet, label in (
        (path_a, packet_a, "A"),
        (path_b, packet_b, "B"),
    ):
        packet = dict(packet)
        packet["annotator_slot"] = label
        packet["response_template"] = {
            "packet_item_id": "",
            "scores": {d: None for d in DIMENSIONS},
            "notes": "",
        }
        path.write_text(
            json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": "nexus-adjudication-dual-export-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "annotator_A_packet": str(path_a),
        "annotator_B_packet": str(path_b),
        "item_count_A": packet_a["item_count"],
        "item_count_B": packet_b["item_count"],
        "status": "PENDING_HUMAN_RESPONSES",
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def import_annotator_responses(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    responses = data.get("responses") or data.get("items") or []
    if not isinstance(responses, list):
        raise ValueError("responses must be a list")
    return {
        "annotator": data.get("annotator_slot") or data.get("annotator") or "unknown",
        "responses": responses,
        "path": str(path),
    }


def validate_completeness(
    packet: Mapping[str, Any], responses: Sequence[Mapping[str, Any]]
) -> list[str]:
    needed = {item["packet_item_id"] for item in packet.get("items") or []}
    got = {
        str(r.get("packet_item_id") or "")
        for r in responses
        if r.get("scores")
    }
    missing = sorted(needed - got)
    errors = []
    if missing:
        errors.append(f"missing {len(missing)} responses; e.g. {missing[:3]}")
    return errors


def agreement_on_dimension(
    a_scores: Mapping[str, Mapping[str, Any]],
    b_scores: Mapping[str, Mapping[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    pairs = []
    for item_id, sa in a_scores.items():
        if item_id not in b_scores:
            continue
        va = (sa.get("scores") or {}).get(dimension)
        vb = (b_scores[item_id].get("scores") or {}).get(dimension)
        if va is None or vb is None:
            continue
        pairs.append((float(va), float(vb)))
    if not pairs:
        return {"n": 0, "agreement_rate": None, "status": "NOT_RUN"}
    agree = sum(1 for x, y in pairs if abs(x - y) < 1e-9)
    return {
        "n": len(pairs),
        "agreement_rate": round(agree / len(pairs), 6),
        "status": "COMPUTED",
        "note": "Simple exact-agreement; Cohen kappa optional later",
    }


def merge_dual_responses(
    packet: Mapping[str, Any],
    resp_a: Sequence[Mapping[str, Any]],
    resp_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge A/B; disagreements require resolution packet (not auto-resolved by LLM)."""
    a_map = {str(r["packet_item_id"]): r for r in resp_a if r.get("packet_item_id")}
    b_map = {str(r["packet_item_id"]): r for r in resp_b if r.get("packet_item_id")}
    disagreements = []
    agreements = []
    for item in packet.get("items") or []:
        iid = item["packet_item_id"]
        if iid not in a_map or iid not in b_map:
            continue
        sa = a_map[iid].get("scores") or {}
        sb = b_map[iid].get("scores") or {}
        diffs = [d for d in DIMENSIONS if sa.get(d) != sb.get(d)]
        if diffs:
            disagreements.append({"packet_item_id": iid, "dimensions": diffs})
        else:
            agreements.append(iid)
    by_dim = {
        d: agreement_on_dimension(a_map, b_map, d) for d in DIMENSIONS
    }
    return {
        "schema_version": "nexus-adjudication-merge-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "agreements": len(agreements),
        "disagreements": disagreements,
        "disagreement_count": len(disagreements),
        "agreement_by_dimension": by_dim,
        "status": (
            "NEEDS_RESOLUTION" if disagreements else "READY_FOR_METRIC_BIND"
        ),
        "resolution_required": bool(disagreements),
        "note": "Do not use diagnostic_model_judgment to resolve disagreements",
    }
