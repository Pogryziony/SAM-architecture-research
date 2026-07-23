"""Exploratory auto-scorable-subset stats (not full primary superiority)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nexus.evaluation.adjudication import classify_scoring_route
from nexus.evaluation.aggregate import aggregate_question_records
from nexus.evaluation.compare import compare_paired_artifacts
from nexus.evaluation.validate import ValidationError


def _filter_auto(artifact: dict, questions_by_id: dict) -> dict:
    rows = []
    for row in artifact.get("per_question") or []:
        qid = str(row.get("question_id") or "")
        q = questions_by_id.get(qid) or {
            "id": qid,
            "category": row.get("question_type"),
            "gold_answer": "",
            "question": row.get("question"),
        }
        # Prefer original oracle record fields when present
        route = classify_scoring_route(questions_by_id.get(qid) or q)
        if route.automated:
            rows.append(row)
    out = dict(artifact)
    out["per_question"] = rows
    out["questions_total"] = len(rows)
    out["aggregates"] = aggregate_question_records(rows)
    out["subset"] = "automatically_scorable"
    out["claim_eligibility"] = {
        "full_primary_superiority": False,
        "exploratory_auto_subset": True,
    }
    out.pop("adjudication_status", None)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    questions = []
    for line in args.questions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    by_id = {str(q["id"]): q for q in questions}
    left = _filter_auto(json.loads(args.left.read_text(encoding="utf-8")), by_id)
    right = _filter_auto(json.loads(args.right.read_text(encoding="utf-8")), by_id)
    # Remove adjudication_status pending if present
    left.pop("adjudication_status", None)
    right.pop("adjudication_status", None)
    for row in left["per_question"] + right["per_question"]:
        row.pop("adjudication_status", None)
    # Exploratory subset compares must stay within one family; if modes differ,
    # refuse rather than silently coerce (caller should pick same-family arms).
    if left.get("comparison_mode") != right.get("comparison_mode"):
        report = {
            "analysis_level": "exploratory_auto_scorable_subset",
            "full_primary_superiority_eligible": False,
            "error": (
                "comparison_mode mismatch between arms; refuse mixing "
                f"{left.get('comparison_mode')!r} vs {right.get('comparison_mode')!r}"
            ),
            "interpretation": (
                "NO FULL SUPERIORITY VERDICT — human adjudication incomplete."
            ),
        }
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(args.output), "refused": True}, sort_keys=True))
        return 0

    try:
        report = compare_paired_artifacts(left, right)
        report["analysis_level"] = "exploratory_auto_scorable_subset"
        report["full_primary_superiority_eligible"] = False
        report["interpretation"] = (
            "Exploratory only on automatically scorable subset. "
            "NO FULL SUPERIORITY VERDICT — human adjudication incomplete."
        )
    except ValidationError as exc:
        report = {
            "analysis_level": "exploratory_auto_scorable_subset",
            "full_primary_superiority_eligible": False,
            "error": str(exc),
            "interpretation": (
                "NO FULL SUPERIORITY VERDICT — human adjudication incomplete."
            ),
        }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "keys": list(report)[:8]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
