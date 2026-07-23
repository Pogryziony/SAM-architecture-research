"""Validate that the evaluator handoff package is complete."""

from __future__ import annotations

from pathlib import Path

REQUIRED = [
    "README.md",
    "PROTOCOL.md",
    "ACCEPTED_SOURCE_FORMATS.md",
    "HIDDEN_QUESTION_SCHEMA.md",
    "RESULT_SCHEMA.md",
    "SYSTEM_CONFIG_REGISTRY.md",
    "METRIC_DEFINITIONS.md",
    "PREREGISTRATION_TEMPLATE.md",
    "ADJUDICATION_RUBRIC.md",
    "LEAKAGE_CHECKLIST.md",
    "FINAL_RELEASE_PROCEDURE.md",
    "tools/hash_corpus.py",
    "tools/validate_handoff.py",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = [rel for rel in REQUIRED if not (root / rel).exists()]
    if missing:
        print("MISSING:", ", ".join(missing))
        return 1
    print("evaluator_handoff OK:", root)
    print("sealed_run_status: BLOCKED (requires independent evaluator + external corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
