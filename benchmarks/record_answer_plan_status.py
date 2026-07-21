"""Record honest AnswerPlan pilot / full-training status (no new training).

Binding constraint for copy/edit overfit+2048 is a non-dummy realization gap
after graph/proof saturation. The current union_recall DummyModel artifact
shows saturated proof/path metrics; low fact_accuracy is DummyModel noise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = (
    _project_root
    / "benchmarks"
    / "results"
    / "answer_plan_pilot_status_20260721T153000Z.json"
)
_EVIDENCE_ARTIFACT = (
    "benchmarks/results/oracle_vs_predicted_union_recall_full_20260721T151500Z.json"
)


def _load_evidence() -> dict:
    path = _project_root / _EVIDENCE_ARTIFACT
    if not path.exists():
        return {}
    art = json.loads(path.read_text(encoding="utf-8"))
    metrics = (art.get("predicted") or {}).get("metrics") or {}
    return {
        "artifact": _EVIDENCE_ARTIFACT,
        "proof_valid_rate": metrics.get("proof_valid_rate"),
        "gold_path_recall_mean": metrics.get("gold_path_recall_mean"),
        "entry_recall_mean": metrics.get("entry_recall_mean"),
        "gold_entity_coverage_mean": metrics.get("gold_entity_coverage_mean"),
        "fact_accuracy_mean": metrics.get("fact_accuracy_mean"),
        "model": "DummyModel",
    }


def build_status(*, ci: bool = False) -> dict:
    source_sha = ""
    try:
        source_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(_project_root)
        ).strip()
    except Exception:
        source_sha = ""
    evidence = _load_evidence()
    binding = False
    reason = (
        "Non-dummy realization is NOT the binding constraint: "
        f"proof_valid_rate={evidence.get('proof_valid_rate')}, "
        f"path_recall={evidence.get('gold_path_recall_mean')}, "
        f"entry_recall={evidence.get('entry_recall_mean')} under DummyModel. "
        f"fact_accuracy_mean={evidence.get('fact_accuracy_mean')} is DummyModel "
        "surface noise, not evidence that AnswerPlan weights are required. "
        "No non-dummy paired baseline exists yet."
    )
    return {
        "schema_version": "nexus-answer-plan-status-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "ci_mode": bool(ci),
        "evidence": evidence,
        "binding_constraint_non_dummy_realization": binding,
        "copy_edit_pilots": {
            "status": "NOT_RUN",
            "reason": reason,
        },
        "full_training": {
            "status": "FULL_TRAINING_BLOCKED",
            "reason": (
                "Sealed until copy/edit pilots pass and a non-dummy/grounded "
                "paired run shows realization is binding."
            ),
        },
        "realizer_training_decision": {
            "train_now": False,
            "reason": reason,
        },
        "next_action": (
            "Publish a grounded/non-dummy paired oracle_vs_predicted artifact; "
            "only then run "
            "`python benchmarks/train_answer_plan_edit_transducer.py "
            "--stages overfit small` if realization remains the limiter."
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
        if args.output.resolve() == _DEFAULT_OUTPUT.resolve():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
    if args.ci:
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
