"""Record honest AnswerPlan pilot / full-training status (no new training).

Graph/proof quality is already saturated on the paired DummyModel artifact.
AnswerPlan full training remains blocked by existing readiness gates; this
script publishes that status instead of pretending pilots passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = (
    _project_root / "benchmarks" / "results" / "answer_plan_pilot_status_20260721T151000Z.json"
)


def build_status(*, ci: bool = False) -> dict:
    source_sha = ""
    try:
        source_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(_project_root)
        ).strip()
    except Exception:
        source_sha = ""
    return {
        "schema_version": "nexus-answer-plan-status-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "ci_mode": bool(ci),
        "copy_edit_pilots": {
            "status": "NOT_RUN",
            "reason": (
                "Overfit-64 / 2048 copy-edit pilots are prepared in "
                "benchmarks/train_answer_plan_edit_transducer.py but are not "
                "required while graph/proof gates are saturated under DummyModel."
            ),
        },
        "full_training": {
            "status": "FULL_TRAINING_BLOCKED",
            "reason": (
                "benchmarks/train_answer_plan_pilots.py and "
                "check_answer_plan_full_training_readiness.py keep autoregressive "
                "full training sealed until copy/edit pilots and generation gates pass."
            ),
        },
        "realizer_training_decision": {
            "train_now": False,
            "reason": (
                "Paired union_cov artifact shows proof_valid_rate≈0.98 and "
                "gold_path_recall≈0.97; low fact_accuracy under DummyModel is not "
                "evidence that new Realizer weights are required."
            ),
        },
        "next_action": (
            "Run copy/edit overfit+2048 pilots only after entry_recall recovery "
            "and when surface realization (non-dummy) is the binding constraint."
        ),
        "status": "RECORDED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    payload = build_status(ci=args.ci)
    if args.output.exists() and not args.force and not args.ci:
        # CI may rewrite a temp path; committed default requires --force.
        if args.output.resolve() == _DEFAULT_OUTPUT.resolve():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
    if args.ci:
        # Deterministic stdout-only path for CI; still write when output is temp/custom.
        print(json.dumps(payload, sort_keys=True))
        if args.output != _DEFAULT_OUTPUT:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "RECORDED", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
