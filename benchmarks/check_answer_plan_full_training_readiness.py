"""Fail-closed readiness gate for a future full AnswerPlan training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_corpus_v2_contracts import sha256_json


def _stage(
    path: Path | None, expected: str, data_readiness_sha256: str,
) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("readiness_sha256") != data_readiness_sha256:
        return None
    matches = [item for item in report.get("stages", []) if item.get("name") == expected]
    return matches[-1] if matches else None


def check(
    data_readiness: Path, overfit_report: Path | None,
    small_report: Path | None, representative_report: Path | None,
) -> dict[str, Any]:
    data = json.loads(data_readiness.read_text(encoding="utf-8"))
    readiness_sha256 = str(data.get("canonical_sha256", ""))
    overfit = _stage(overfit_report, "overfit", readiness_sha256)
    small = _stage(small_report, "small", readiness_sha256)
    representative = _stage(representative_report, "representative", readiness_sha256)

    def passes(stage: dict[str, Any] | None, *, exact: float, f1: float) -> bool:
        if stage is None:
            return False
        metrics = stage.get("generation", {})
        return bool(
            metrics.get("exact_match", 0.0) >= exact
            and metrics.get("token_f1", 0.0) >= f1
            and metrics.get("eos_rate", 0.0) >= 0.95
            and metrics.get("empty_rate", 1.0) == 0.0
            and metrics.get("unsupported_number_rate", 1.0) <= 0.01
        )

    checks = {
        "data_ready_for_bounded_pilot": data.get("status") == "READY_FOR_BOUNDED_PILOT",
        "test_still_sealed": data.get("baselines", {}).get("test_split_accessed") is False,
        "full_training_not_already_launched": data.get("pilot_protocol", {}).get("full_training_authorized") is False,
        "overfit_generation_gate": passes(overfit, exact=0.80, f1=0.85),
        "small_generation_gate": passes(small, exact=0.50, f1=0.70),
        "representative_generation_gate": passes(representative, exact=0.70, f1=0.85),
    }
    report = {
        "schema_version": "nexus-answer-plan-full-training-readiness-v1",
        "status": "READY_FOR_FULL_TRAINING" if all(checks.values()) else "FULL_TRAINING_BLOCKED",
        "checks": checks,
        "blocking_checks": sorted(key for key, value in checks.items() if not value),
        "data_readiness_sha256": data.get("canonical_sha256"),
        "full_training_launched": False,
        "pilot_evidence": {
            "overfit": overfit,
            "small": small,
            "representative": representative,
        },
    }
    report["canonical_sha256"] = sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-readiness", type=Path, required=True)
    parser.add_argument("--overfit-report", type=Path)
    parser.add_argument("--small-report", type=Path)
    parser.add_argument("--representative-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(
        args.data_readiness, args.overfit_report, args.small_report,
        args.representative_report,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "blocking_checks": report["blocking_checks"],
        "canonical_sha256": report["canonical_sha256"],
    }, indent=2))
    return 0 if report["status"] == "READY_FOR_FULL_TRAINING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
