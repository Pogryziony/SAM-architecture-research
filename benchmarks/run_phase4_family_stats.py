"""Family-wide Holm-corrected paired stats over Phase-4 arms.

Uses ``proxy_key_fact_correct`` for exploratory families only.
Refuses primary ``grounded_correct`` superiority while adjudication is pending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nexus.evaluation.compare import compare_paired_artifacts
from nexus.evaluation.multiple_comparison import apply_holm_to_comparisons
from nexus.evaluation.validate import ValidationError


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"

SYSTEM_LEVEL = [
    "phase4_qwen_closed_book_oracle_v1.json",
    "phase4_qwen_long_context_oracle_v1.json",
]
CONTROLLED = [
    "phase4_bm25_rag_qwen_oracle_v1.json",
    "phase4_dense_rag_qwen_oracle_v1.json",
    "phase4_hybrid_rag_qwen_oracle_v1.json",
    "phase4_hybrid_rerank_rag_qwen_oracle_v1.json",
    "phase4_nexus_graph_evidence_qwen_oracle_v1.json",
]


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        # accept repair filenames
        alt = RESULTS / name.replace(".json", "_repair.json")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(name)
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_family(
    reference: dict,
    others: list[tuple[str, dict]],
    *,
    metric: str,
    allow_placeholders: bool,
) -> list[dict]:
    comps = []
    for label, art in others:
        try:
            # Temporarily clear pending gate for exploratory proxy metric only
            left = dict(reference)
            right = dict(art)
            if metric == "proxy_key_fact_correct":
                left = {**left, "adjudication_status": "EXPLORATORY_PROXY_ONLY"}
                right = {**right, "adjudication_status": "EXPLORATORY_PROXY_ONLY"}
                for row in left.get("per_question") or []:
                    row.pop("adjudication_status", None)
                for row in right.get("per_question") or []:
                    row.pop("adjudication_status", None)
            comp = compare_paired_artifacts(
                left,
                right,
                metric_name=metric,
                allow_placeholders=allow_placeholders,
                family_size=1,  # adjusted later across family
            )
            comp["comparison_label"] = label
            comps.append(comp)
        except ValidationError as exc:
            comps.append(
                {
                    "comparison_label": label,
                    "status": "REFUSED",
                    "reason": str(exc),
                    "mcnemar": {"p_value": 1.0},
                }
            )
    runnable = [c for c in comps if c.get("status") != "REFUSED"]
    if runnable:
        adjusted = apply_holm_to_comparisons(runnable)
        by_label = {c["comparison_label"]: c for c in adjusted}
        out = []
        for c in comps:
            out.append(by_label.get(c["comparison_label"], c))
        return out
    return comps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="proxy_key_fact_correct",
        choices=("proxy_key_fact_correct", "grounded_correct"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.metric == "grounded_correct":
        raise SystemExit(
            "refusing grounded_correct family stats until human adjudication completes"
        )

    nexus = None
    for cand in (
        "eval_oracle_v1_grounded_evidence_repair.json",
        "eval_oracle_v1_grounded_phase3.json",
    ):
        p = RESULTS / cand
        if p.exists():
            nexus = json.loads(p.read_text(encoding="utf-8"))
            nexus_name = cand
            break
    if nexus is None:
        raise SystemExit("missing NEXUS grounded artifact")

    sys_arts = [(n, _load(n)) for n in SYSTEM_LEVEL if (RESULTS / n).exists() or (RESULTS / n.replace(".json", "_repair.json")).exists()]
    ctrl_arts = [(n, _load(n)) for n in CONTROLLED if (RESULTS / n).exists() or (RESULTS / n.replace(".json", "_repair.json")).exists()]

    report = {
        "schema_version": "nexus-phase4-family-stats-v1",
        "metric_name": args.metric,
        "claim_eligibility": {
            "full_primary_superiority": False,
            "reason": "exploratory proxy metric only; adjudication incomplete",
        },
        "reference": nexus_name,
        "system_level_family": _pair_family(
            nexus, sys_arts, metric=args.metric, allow_placeholders=False
        ),
        "controlled_family": _pair_family(
            nexus, ctrl_arts, metric=args.metric, allow_placeholders=False
        ),
        "note": "system_level and controlled families are corrected separately",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(args.output), "metric": args.metric}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
