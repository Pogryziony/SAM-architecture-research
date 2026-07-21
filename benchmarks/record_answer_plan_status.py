"""Record honest AnswerPlan pilot / full-training status (no new training).

Binding constraint for copy/edit overfit+2048 is a non-dummy realization gap
after graph/proof saturation on a grounded/non-dummy paired artifact.
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
    / "answer_plan_pilot_status_20260721T160000Z.json"
)
_EVIDENCE_CANDIDATES = (
    "benchmarks/results/oracle_vs_predicted_union_l1_acceptance_full_20260721T234500Z.json",
    "benchmarks/results/oracle_vs_predicted_union_l1_acceptance_full_20260721T220000Z.json",
    "benchmarks/results/oracle_vs_predicted_union_l1_acceptance_full_20260721T200000Z.json",
    "benchmarks/results/oracle_vs_predicted_union_l1_acceptance_full_20260721T180000Z.json",
    "benchmarks/results/oracle_vs_predicted_union_l1_det_full_20260721T163000Z.json",
    "benchmarks/results/oracle_vs_predicted_union_grounded_full_20260721T163000Z.json",
    "benchmarks/results/oracle_vs_predicted_union_grounded_full_20260721T160000Z.json",
    "benchmarks/results/oracle_vs_predicted_union_recall_full_20260721T151500Z.json",
)

# Graph/ER considered saturated when these hold on the evidence artifact.
_GRAPH_ENTRY_MIN = 0.90
_GRAPH_PATH_MIN = 0.90
_GRAPH_PROOF_MIN = 0.90
# Realization is binding only when graph is saturated AND surface accuracy lags.
_FACT_ACC_BINDING_MAX = 0.50


def _load_evidence() -> dict:
    for rel in _EVIDENCE_CANDIDATES:
        path = _project_root / rel
        if not path.exists():
            continue
        art = json.loads(path.read_text(encoding="utf-8"))
        metrics = (art.get("predicted") or {}).get("metrics") or {}
        oracle_metrics = (art.get("oracle") or {}).get("metrics") or {}
        return {
            "artifact": rel,
            "proof_valid_rate": metrics.get("proof_valid_rate"),
            "gold_path_recall_mean": metrics.get("gold_path_recall_mean"),
            "entry_recall_mean": metrics.get("entry_recall_mean"),
            "gold_entity_coverage_mean": metrics.get("gold_entity_coverage_mean"),
            "fact_accuracy_mean": metrics.get("fact_accuracy_mean"),
            "oracle_fact_accuracy_mean": oracle_metrics.get("fact_accuracy_mean"),
            "model": art.get("model_name") or "unknown",
            "realizer_backend": art.get("realizer_backend")
            or (art.get("predicted_resolver") or {}).get("realizer_backend"),
            "dummy_model": str(art.get("model_name") or "").lower() == "dummymodel",
        }
    return {}


def _is_binding(evidence: dict) -> tuple[bool, str]:
    if not evidence:
        return False, "No paired evidence artifact found."
    try:
        proof = float(evidence.get("proof_valid_rate") or 0.0)
        path = float(evidence.get("gold_path_recall_mean") or 0.0)
        entry = float(evidence.get("entry_recall_mean") or 0.0)
        fact = float(evidence.get("fact_accuracy_mean") or 0.0)
        oracle_fact = float(evidence.get("oracle_fact_accuracy_mean") or 0.0)
    except (TypeError, ValueError):
        return False, "Evidence metrics are not numeric."
    backend = str(evidence.get("realizer_backend") or "synth")
    dummy = bool(evidence.get("dummy_model"))
    groundedish = backend in {
        "grounded_v1",
        "pointer_copy",
        "deterministic_render",
        "l1_acceptance",
    } and not dummy
    graph_saturated = (
        proof >= _GRAPH_PROOF_MIN
        and path >= _GRAPH_PATH_MIN
        and entry >= _GRAPH_ENTRY_MIN
    )
    if not groundedish:
        return False, (
            "Non-dummy realization is NOT established: evidence is still "
            f"model={evidence.get('model')} realizer_backend={backend}. "
            f"proof={proof}, path={path}, entry={entry}, fact_accuracy={fact}."
        )
    # Binding only when gold-entity oracle realizes well but predicted does not.
    # If oracle fact_accuracy is also low, the gap is shared surface/realizer
    # coverage — not AnswerPlan weights on the ER path.
    predicted_gap = oracle_fact - fact
    if (
        graph_saturated
        and oracle_fact >= _FACT_ACC_BINDING_MAX
        and fact < _FACT_ACC_BINDING_MAX
        and predicted_gap >= 0.15
    ):
        return True, (
            "Realization IS the binding constraint under grounded/non-dummy "
            f"artifact: oracle_fact={oracle_fact}, predicted_fact={fact}, "
            f"gap={predicted_gap:.4f}, proof={proof}, path={path}, entry={entry}, "
            f"realizer_backend={backend}."
        )
    return False, (
        "Non-dummy realization is NOT the binding constraint: "
        f"oracle_fact={oracle_fact}, predicted_fact={fact}, proof={proof}, "
        f"path={path}, entry={entry}, realizer_backend={backend}/"
        f"model={evidence.get('model')}. Predicted does not lag a strong "
        "oracle surface score, so AnswerPlan edit-transducer training is not "
        "justified."
    )


def build_status(*, ci: bool = False) -> dict:
    source_sha = ""
    try:
        source_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(_project_root)
        ).strip()
    except Exception:
        source_sha = ""
    evidence = _load_evidence()
    binding, reason = _is_binding(evidence)
    next_action = (
        "Run `python benchmarks/train_answer_plan_edit_transducer.py "
        "--stages overfit small` on an authorized prepared root."
        if binding
        else (
            "Keep AnswerPlan pilots deferred. Prefer graph/provenance/intent "
            "gaps over Realizer weight training unless a future grounded paired "
            "run shows fact_accuracy as the limiter."
        )
    )
    return {
        "schema_version": "nexus-answer-plan-status-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "ci_mode": bool(ci),
        "evidence": evidence,
        "binding_constraint_non_dummy_realization": binding,
        "copy_edit_pilots": {
            "status": "NOT_RUN" if not binding else "AUTHORIZED_BUT_NOT_RUN",
            "reason": reason,
        },
        "full_training": {
            "status": "FULL_TRAINING_BLOCKED",
            "reason": (
                "Sealed until copy/edit pilots pass under an authorized "
                "prepared-root readiness gate."
            ),
        },
        "realizer_training_decision": {
            "train_now": False,
            "reason": reason if not binding else (
                reason + " Pilots authorized but not auto-started in this slice "
                "(bounded overfit+2048 requires explicit prepared-root readiness)."
            ),
        },
        "next_action": next_action,
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
