"""Export evidence-bearing dual adjudication packets from Phase-4 arm artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nexus.evaluation.adjudication_io import export_dual_packets


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ORACLE = ROOT / "benchmarks" / "qa-dataset" / "oracle_v1.jsonl"


def _load_answers(path: Path) -> dict[str, dict]:
    art = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in art.get("per_question") or []:
        out[str(row["question_id"])] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        nargs="+",
        default=[
            "eval_oracle_v1_grounded_evidence_repair.json",
            "phase4_qwen_closed_book_oracle_v1.json",
            "phase4_hybrid_rag_qwen_oracle_v1.json",
        ],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "phase4_adjudication_export_evidence_v1",
    )
    args = parser.parse_args()
    questions = [
        json.loads(line)
        for line in ORACLE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    system_answers = {}
    for fname in args.systems:
        path = RESULTS / fname if not Path(fname).is_absolute() else Path(fname)
        if not path.exists():
            raise SystemExit(f"missing arm artifact: {path}")
        system_answers[path.stem] = _load_answers(path)
    manifest = export_dual_packets(
        questions, system_answers, args.out, require_evidence=True
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
