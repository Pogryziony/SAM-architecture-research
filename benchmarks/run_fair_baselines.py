"""Emit honest NOT_RUN / OK records for fair-comparison baseline arms.

Does not fabricate scores. Real LLM/RAG execution requires environment pins
documented in docs/external-evaluation-protocol.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from nexus.baselines import list_arms
from nexus.baselines.adapters import run_baseline_eval


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def run_arm(
    arm_id: str,
    questions: list[dict[str, Any]],
    *,
    dataset_id: str,
    dataset_sha256: str,
    comparison_mode: str = "system_level",
) -> dict[str, Any]:
    return run_baseline_eval(
        arm_id,
        questions,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        comparison_mode=comparison_mode,
        source_commit=_git_head(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        default="closed_book_llm",
        help="Baseline arm id (see --list)",
    )
    parser.add_argument("--list", action="store_true", help="List arms and exit")
    parser.add_argument(
        "--questions-json",
        type=Path,
        help="Optional JSON list of {id,question,...}; default one smoke question",
    )
    parser.add_argument("--output", type=Path, required=False)
    args = parser.parse_args(argv)

    if args.list:
        print(
            json.dumps(
                [
                    {
                        "arm_id": a.arm_id,
                        "family": a.family,
                        "is_placeholder": a.is_placeholder,
                        "modern_rag": a.modern_rag,
                    }
                    for a in list_arms()
                ],
                indent=2,
            )
        )
        return 0

    if args.questions_json:
        questions = json.loads(args.questions_json.read_text(encoding="utf-8"))
        dataset_id = args.questions_json.name
        dataset_sha256 = __import__("hashlib").sha256(
            args.questions_json.read_bytes()
        ).hexdigest()
    else:
        questions = [
            {
                "id": "smoke_q0",
                "question": "Smoke question for baseline harness.",
                "domain": "smoke",
                "question_type": "smoke",
            }
        ]
        dataset_id = "smoke_inline_v1"
        dataset_sha256 = "0" * 64

    artifact = run_arm(
        args.arm,
        questions,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
    )
    text = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "arm": args.arm,
                "questions_total": artifact["questions_total"],
                "is_placeholder": artifact["arm_metadata"]["is_placeholder"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
