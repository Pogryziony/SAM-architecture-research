"""Family-wide Holm-corrected paired stats over Phase-4 arms.

Uses ``proxy_key_fact_correct`` for exploratory families only.
Refuses primary ``grounded_correct`` superiority while adjudication is pending.

Family separation:
- System-level: closed-book and long-context (no retrieval), compared against each other
- Controlled: all RAG variants + NEXUS graph-evidence, compared against each other

Each family uses its own reference artifact (within the same comparison_mode).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from nexus.evaluation.compare import compare_paired_artifacts
from nexus.evaluation.dataset_identity import hash_dataset
from nexus.evaluation.multiple_comparison import apply_holm_to_comparisons
from nexus.evaluation.validate import ValidationError


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ORACLE = ROOT / "benchmarks" / "qa-dataset" / "oracle_v1.jsonl"

# Canonical dataset hash for Phase-4
CANONICAL_DATASET_SHA256 = "ca96877de86990e7757c18efe3576ec660b454d6984866cbebc5939ead1a63d5"

# System-level family: no retrieval, raw LLM capability
SYSTEM_LEVEL = [
    "phase4_qwen_closed_book_oracle_v1.json",
    "phase4_qwen_long_context_oracle_v1.json",
]

# Controlled family: retrieval-augmented arms (same mode)
CONTROLLED = [
    "phase4_bm25_rag_qwen_oracle_v1.json",
    "phase4_dense_rag_qwen_oracle_v1.json",
    "phase4_hybrid_rag_qwen_oracle_v1.json",
    "phase4_hybrid_rerank_rag_qwen_oracle_v1.json",
    "phase4_nexus_graph_evidence_qwen_oracle_v1.json",
]

# Reference artifacts for each family (must match comparison_mode)
CONTROLLED_REFERENCE = "phase4_nexus_graph_evidence_qwen_oracle_v1.json"
# System-level family uses pairwise comparison without a single reference


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


def _verify_dataset_hash(artifact: dict, name: str) -> None:
    """Verify artifact uses canonical dataset hash."""
    got = artifact.get("dataset_sha256", "")
    if got != CANONICAL_DATASET_SHA256:
        raise ValidationError(
            f"{name} has dataset_sha256 {got[:16]}..., expected {CANONICAL_DATASET_SHA256[:16]}..."
        )


def _prepare_for_proxy_metric(art: dict) -> dict:
    """Prepare artifact for proxy metric comparison (clear adjudication gates)."""
    result = {**art, "adjudication_status": "EXPLORATORY_PROXY_ONLY"}
    if "per_question" in result:
        result["per_question"] = [
            {k: v for k, v in row.items() if k != "adjudication_status"}
            for row in result["per_question"]
        ]
    return result


def _pair_family(
    reference: dict,
    others: list[tuple[str, dict]],
    *,
    metric: str,
    allow_placeholders: bool,
    reference_name: str,
) -> list[dict]:
    """Compare reference against all others, applying Holm correction."""
    comps = []
    for label, art in others:
        try:
            left = dict(reference)
            right = dict(art)
            if metric == "proxy_key_fact_correct":
                left = _prepare_for_proxy_metric(left)
                right = _prepare_for_proxy_metric(right)
            comp = compare_paired_artifacts(
                left,
                right,
                metric_name=metric,
                allow_placeholders=allow_placeholders,
                family_size=1,  # adjusted later across family
            )
            comp["comparison_label"] = label
            comp["reference"] = reference_name
            comps.append(comp)
        except ValidationError as exc:
            comps.append(
                {
                    "comparison_label": label,
                    "reference": reference_name,
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


def _pairwise_system_level(
    artifacts: list[tuple[str, dict]],
    *,
    metric: str,
) -> list[dict]:
    """Pairwise comparisons within system-level family (no single reference)."""
    if len(artifacts) < 2:
        return [{"status": "REFUSED", "reason": "need at least 2 system-level arms"}]

    comps = []
    for i, (name_a, art_a) in enumerate(artifacts):
        for name_b, art_b in artifacts[i + 1 :]:
            try:
                left = art_a
                right = art_b
                if metric == "proxy_key_fact_correct":
                    left = _prepare_for_proxy_metric(left)
                    right = _prepare_for_proxy_metric(right)
                comp = compare_paired_artifacts(
                    left,
                    right,
                    metric_name=metric,
                    allow_placeholders=False,
                    family_size=1,
                )
                comp["comparison_label"] = f"{name_a} vs {name_b}"
                comp["pair"] = [name_a, name_b]
                comps.append(comp)
            except ValidationError as exc:
                comps.append(
                    {
                        "comparison_label": f"{name_a} vs {name_b}",
                        "pair": [name_a, name_b],
                        "status": "REFUSED",
                        "reason": str(exc),
                        "mcnemar": {"p_value": 1.0},
                    }
                )

    runnable = [c for c in comps if c.get("status") != "REFUSED"]
    if runnable:
        adjusted = apply_holm_to_comparisons(runnable)
        by_label = {c["comparison_label"]: c for c in adjusted}
        return [by_label.get(c["comparison_label"], c) for c in comps]
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

    # Load and verify controlled family reference
    try:
        ctrl_ref = _load(CONTROLLED_REFERENCE)
        ctrl_ref_name = CONTROLLED_REFERENCE
        _verify_dataset_hash(ctrl_ref, ctrl_ref_name)
    except FileNotFoundError:
        raise SystemExit(f"missing controlled family reference: {CONTROLLED_REFERENCE}")
    except ValidationError as exc:
        raise SystemExit(f"controlled reference invalid: {exc}")

    # Load system-level artifacts
    sys_arts = []
    for name in SYSTEM_LEVEL:
        try:
            art = _load(name)
            _verify_dataset_hash(art, name)
            sys_arts.append((name, art))
        except FileNotFoundError:
            print(f"warning: system-level artifact missing: {name}")
        except ValidationError as exc:
            print(f"warning: system-level artifact invalid: {name}: {exc}")

    # Load controlled artifacts (excluding the reference)
    ctrl_arts = []
    for name in CONTROLLED:
        if name == CONTROLLED_REFERENCE:
            continue  # reference is not compared against itself
        try:
            art = _load(name)
            _verify_dataset_hash(art, name)
            ctrl_arts.append((name, art))
        except FileNotFoundError:
            print(f"warning: controlled artifact missing: {name}")
        except ValidationError as exc:
            print(f"warning: controlled artifact invalid: {name}: {exc}")

    # Build family stats
    system_level_stats = _pairwise_system_level(sys_arts, metric=args.metric)
    controlled_stats = _pair_family(
        ctrl_ref,
        ctrl_arts,
        metric=args.metric,
        allow_placeholders=False,
        reference_name=ctrl_ref_name,
    )

    # Count real comparisons
    sys_runnable = [c for c in system_level_stats if c.get("status") != "REFUSED"]
    ctrl_runnable = [c for c in controlled_stats if c.get("status") != "REFUSED"]

    report = {
        "schema_version": "nexus-phase4-family-stats-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric_name": args.metric,
        "dataset_sha256": CANONICAL_DATASET_SHA256,
        "claim_eligibility": {
            "full_primary_superiority": False,
            "reason": "exploratory proxy metric only; adjudication incomplete",
        },
        "system_level_family": {
            "design": "pairwise within family",
            "comparison_mode": "system_level",
            "arms": [name for name, _ in sys_arts],
            "comparisons": system_level_stats,
            "n_runnable": len(sys_runnable),
            "holm_applied": len(sys_runnable) > 0,
        },
        "controlled_family": {
            "design": "all vs reference",
            "comparison_mode": "controlled",
            "reference": ctrl_ref_name,
            "arms": [name for name, _ in ctrl_arts],
            "comparisons": controlled_stats,
            "n_runnable": len(ctrl_runnable),
            "holm_applied": len(ctrl_runnable) > 0,
        },
        "note": (
            "system_level and controlled families are separate comparison universes; "
            "Holm correction applied within each family independently"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "wrote": str(args.output),
                "metric": args.metric,
                "system_level_runnable": len(sys_runnable),
                "controlled_runnable": len(ctrl_runnable),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
