"""INVALID HISTORICAL TOOLING — do not use for primary evidence.

This script previously performed metadata-only dataset_sha256 rebinding so
Phase-4 paired compares could share a question-only hash. That practice is
forbidden: primary artifacts must be regenerated from the generating checkout
with ``nexus.evaluation.dataset_identity.hash_dataset``.

Kept only so historical references resolve to an explicit refusal.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "REFUSED: metadata-only dataset rebinding is invalid.\n"
        "Regenerate NEXUS/Phase-4 arms from the evidence-repair checkout using:\n"
        "  python benchmarks/regenerate_evidence_identity.py\n"
        "See nexus.evaluation.dataset_identity.assert_primary_dataset_hash.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
