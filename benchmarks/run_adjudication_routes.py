"""Generate adjudication routes and optional blinded packet for oracle questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nexus.domain import load_domain_pack
from nexus.evaluation.adjudication import (
    apply_automated_scores,
    build_blinded_packet,
    route_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="sam", choices=("mini", "sam"))
    parser.add_argument("--answers", type=Path, default=None, help="eval-result-v1 JSON")
    parser.add_argument("--routes-output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument("--packet-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    pack = load_domain_pack(args.domain)
    questions = pack.evaluation_tasks()
    routes = route_dataset(questions)
    if args.routes_output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.routes_output}")
    args.routes_output.parent.mkdir(parents=True, exist_ok=True)
    args.routes_output.write_text(
        json.dumps(routes, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    answers_by_id: dict[str, dict] = {}
    system_answers: dict[str, dict[str, dict]] = {}
    if args.answers and args.answers.exists():
        art = json.loads(args.answers.read_text(encoding="utf-8"))
        sid = str(art.get("system_id") or "system")
        system_answers[sid] = {}
        for row in art.get("per_question") or []:
            qid = str(row.get("question_id") or "")
            answers_by_id[qid] = row
            system_answers[sid][qid] = row

    if args.scores_output:
        scores = apply_automated_scores(questions, answers_by_id)
        if args.scores_output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.scores_output}")
        args.scores_output.write_text(
            json.dumps(scores, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if args.packet_output:
        if not system_answers:
            system_answers = {"nexus": {}}
        packet = build_blinded_packet(questions, system_answers, seed=args.seed)
        if args.packet_output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.packet_output}")
        args.packet_output.write_text(
            json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "questions_total": routes["questions_total"],
                "automatically_scorable": routes["automatically_scorable"],
                "human_dependent": routes["human_dependent"],
                "routes_output": str(args.routes_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
