"""Stage-4 style NEXUS vs RAG vs LLM-only campaign on frozen oracle_v1.

No training. Uses SynthesizingModel / EvidenceBlindModel so the run is
CI-safe without Ollama. Verdict thresholds come from
EXPERIMENT_NEXUS_ARCHITECTURE_VALIDATION.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.compare_arms import compare_paired
from benchmarks.run_nexus_oracle import _path_recall
from benchmarks.run_oracle_vs_predicted import (
    DEFAULT_ER3_DIR,
    build_predicted_runner,
    set_recall,
)
from benchmarks.scoring import compute_fact_score
from nexus.reasoning.model_interface import (
    EvidenceBlindModel,
    SynthesizingModel,
)

DEFAULT_DATASET = (
    _project_root / "benchmarks" / "qa-dataset" / "oracle_v1.jsonl"
)
ENTRY_MIN = 0.90
PATH_MIN = 0.90
PROOF_MIN = 0.90
FACT_MIN = 0.70
ANSWERPLAN_LAG_MIN = 0.15


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_abstain(answer: str) -> bool:
    text = (answer or "").casefold()
    return "insufficient evidence" in text or not text.strip()


def _mean(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not kept:
        return None
    return round(sum(kept) / len(kept), 4)


def _run_nexus_arm(
    records: list[dict[str, Any]],
    *,
    er3_dir: str,
) -> dict[str, Any]:
    from benchmarks.run_benchmark import build_benchmark_graph

    graph, provenance = build_benchmark_graph()
    model = SynthesizingModel()
    runner, identity = build_predicted_runner(
        graph,
        predicted_resolver="union",
        er3_dir=er3_dir,
        model=model,
        realizer_backend="l1_acceptance",
    )
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    pipeline = runner.run(records, source_sha=source_sha)
    if pipeline.errors:
        raise RuntimeError("NEXUS arm failed: " + "; ".join(pipeline.errors))

    facts: list[float | None] = []
    entries: list[float] = []
    paths: list[float | None] = []
    proofs: list[float] = []
    abstain_ok = 0
    abstain_total = 0
    rows: list[dict[str, Any]] = []
    for record, qr in zip(records, pipeline.per_question, strict=True):
        fact = compute_fact_score(qr.answer, record["gold_answer"])["fuzzy_accuracy"]
        entry_ids = list(
            getattr(qr, "selected_entry_nodes", None)
            or getattr(qr, "predicted_entities", None)
            or []
        )
        entry = set_recall(record.get("gold_entities") or [], entry_ids)
        path = _path_recall(record.get("gold_path") or [], qr.reasoning_audit)
        proof = 1.0 if bool(getattr(qr, "proof_valid", False)) else 0.0
        facts.append(fact)
        entries.append(entry)
        paths.append(path)
        proofs.append(proof)
        should = bool(record.get("should_abstain"))
        predicted_abstain = _is_abstain(qr.answer)
        if should:
            abstain_total += 1
            if predicted_abstain:
                abstain_ok += 1
        rows.append(
            {
                "question_id": record["id"],
                "answer": qr.answer,
                "fact_accuracy": fact,
                "entry_recall": round(entry, 4),
                "gold_path_recall": None if path is None else round(path, 4),
                "should_abstain": should,
                "predicted_abstain": predicted_abstain,
            }
        )

    return {
        "arm": "nexus_l1_union",
        "identity": identity,
        "graph": {
            "node_count": provenance.get("node_count"),
            "edge_count": provenance.get("edge_count"),
        },
        "metrics": {
            "fact_accuracy_mean": _mean(facts),
            "entry_recall_mean": _mean(entries),
            "gold_path_recall_mean": _mean(paths),
            "proof_valid_rate": _mean(proofs),
            "abstain_recall": (
                round(abstain_ok / abstain_total, 4) if abstain_total else None
            ),
            "answered": sum(1 for r in rows if not _is_abstain(r["answer"])),
            "questions": len(rows),
        },
        "per_question": rows,
    }


def _run_rag_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    from benchmarks.rag_baseline import initialize_rag_pipeline, run_rag_pipeline

    model = SynthesizingModel()
    rag = initialize_rag_pipeline(model, backend="lexical")
    facts: list[float | None] = []
    rows: list[dict[str, Any]] = []
    abstain_ok = 0
    abstain_total = 0
    for record in records:
        result = run_rag_pipeline(record["question"], rag)
        answer = str(result.get("answer") or "")
        fact = compute_fact_score(answer, record["gold_answer"])["fuzzy_accuracy"]
        facts.append(fact)
        should = bool(record.get("should_abstain"))
        predicted_abstain = _is_abstain(answer)
        if should:
            abstain_total += 1
            if predicted_abstain:
                abstain_ok += 1
        rows.append(
            {
                "question_id": record["id"],
                "answer": answer,
                "fact_accuracy": fact,
                "should_abstain": should,
                "predicted_abstain": predicted_abstain,
                "error": result.get("error") or "",
            }
        )
    return {
        "arm": "rag_lexical",
        "metrics": {
            "fact_accuracy_mean": _mean(facts),
            "abstain_recall": (
                round(abstain_ok / abstain_total, 4) if abstain_total else None
            ),
            "answered": sum(1 for r in rows if not _is_abstain(r["answer"])),
            "questions": len(rows),
        },
        "per_question": rows,
    }


def _run_llm_only_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    model = EvidenceBlindModel()
    facts: list[float | None] = []
    rows: list[dict[str, Any]] = []
    abstain_ok = 0
    abstain_total = 0
    for record in records:
        # Evidence-blind prompt: question only, no graph pack.
        prompt = (
            "QUESTION:\n"
            f"{record['question']}\n\n"
            "EVIDENCE:\n(No evidence found)\n"
        )
        answer = model.generate(prompt)
        fact = compute_fact_score(answer, record["gold_answer"])["fuzzy_accuracy"]
        facts.append(fact)
        should = bool(record.get("should_abstain"))
        predicted_abstain = _is_abstain(answer)
        if should:
            abstain_total += 1
            if predicted_abstain:
                abstain_ok += 1
        rows.append(
            {
                "question_id": record["id"],
                "answer": answer,
                "fact_accuracy": fact,
                "should_abstain": should,
                "predicted_abstain": predicted_abstain,
            }
        )
    return {
        "arm": "llm_only_evidence_blind",
        "metrics": {
            "fact_accuracy_mean": _mean(facts),
            "abstain_recall": (
                round(abstain_ok / abstain_total, 4) if abstain_total else None
            ),
            "answered": sum(1 for r in rows if not _is_abstain(r["answer"])),
            "questions": len(rows),
        },
        "per_question": rows,
    }


def _paired_fact_lists(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[list[float | None], list[float | None]]:
    by_id = {r["question_id"]: r for r in right["per_question"]}
    left_scores: list[float | None] = []
    right_scores: list[float | None] = []
    for row in left["per_question"]:
        other = by_id.get(row["question_id"])
        if other is None:
            continue
        left_scores.append(row.get("fact_accuracy"))
        right_scores.append(other.get("fact_accuracy"))
    return left_scores, right_scores


def decide_verdict(
    nexus: dict[str, Any],
    rag: dict[str, Any],
    llm_only: dict[str, Any],
    *,
    oracle_fact: float | None,
    predicted_fact: float | None,
) -> dict[str, Any]:
    m = nexus["metrics"]
    entry = float(m.get("entry_recall_mean") or 0.0)
    path = float(m.get("gold_path_recall_mean") or 0.0)
    proof = float(m.get("proof_valid_rate") or 0.0)
    fact = float(m.get("fact_accuracy_mean") or 0.0)
    rag_fact = float(rag["metrics"].get("fact_accuracy_mean") or 0.0)
    llm_fact = float(llm_only["metrics"].get("fact_accuracy_mean") or 0.0)

    graph_ok = entry >= ENTRY_MIN and path >= PATH_MIN and proof >= PROOF_MIN
    surface_ok = fact >= FACT_MIN
    beats_rag = fact > rag_fact
    beats_llm = fact > llm_fact
    lag = None
    if oracle_fact is not None and predicted_fact is not None:
        lag = round(float(oracle_fact) - float(predicted_fact), 4)
    answerplan_binding = bool(
        oracle_fact is not None
        and predicted_fact is not None
        and float(oracle_fact) >= 0.50
        and float(predicted_fact) < 0.50
        and (lag or 0.0) >= ANSWERPLAN_LAG_MIN
    )

    if graph_ok and surface_ok and beats_rag and beats_llm:
        decision = "VALIDATED"
        summary = (
            "NEXUS L1 meets graph/ER and surface thresholds and beats RAG and "
            "LLM-only on fact accuracy for this freeze."
        )
    elif graph_ok and surface_ok:
        decision = "CONDITIONAL"
        summary = (
            "NEXUS L1 meets saturation/surface thresholds but does not clearly "
            "beat both baselines on fact accuracy."
        )
    else:
        decision = "REJECTED"
        summary = (
            "NEXUS L1 misses graph/ER saturation and/or surface fact threshold "
            "on this freeze."
        )

    return {
        "decision": decision,
        "summary": summary,
        "thresholds": {
            "entry_min": ENTRY_MIN,
            "path_min": PATH_MIN,
            "proof_min": PROOF_MIN,
            "fact_min": FACT_MIN,
            "answerplan_lag_min": ANSWERPLAN_LAG_MIN,
        },
        "checks": {
            "graph_er_saturated": graph_ok,
            "surface_competence": surface_ok,
            "beats_rag_fact": beats_rag,
            "beats_llm_only_fact": beats_llm,
            "answerplan_binding": answerplan_binding,
            "oracle_predicted_lag": lag,
        },
        "next_training": (
            "Authorize bounded AnswerPlan overfit+2048."
            if answerplan_binding
            else (
                "No AnswerPlan training. If entry < 0.95 from new canonical "
                "nodes only, consider bounded ER3 refresh; otherwise keep "
                "lexical/L1 hygiene and baseline monitoring."
            )
        ),
    }


def run_campaign(
    *,
    dataset: Path,
    output: Path,
    er3_dir: str,
    limit: int | None = None,
    paired_evidence: Path | None = None,
) -> dict[str, Any]:
    records = _read_jsonl(dataset)
    if limit:
        records = records[:limit]
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    nexus = _run_nexus_arm(records, er3_dir=er3_dir)
    rag = _run_rag_arm(records)
    llm_only = _run_llm_only_arm(records)

    nexus_vs_rag = compare_paired(
        *_paired_fact_lists(nexus, rag), "NEXUS", "RAG"
    )
    nexus_vs_llm = compare_paired(
        *_paired_fact_lists(nexus, llm_only), "NEXUS", "LLM_ONLY"
    )

    oracle_fact = None
    predicted_fact = nexus["metrics"].get("fact_accuracy_mean")
    if paired_evidence and paired_evidence.exists():
        paired = json.loads(paired_evidence.read_text(encoding="utf-8"))
        oracle_fact = (paired.get("oracle") or {}).get("metrics", {}).get(
            "fact_accuracy_mean"
        )
        predicted_fact = (paired.get("predicted") or {}).get("metrics", {}).get(
            "fact_accuracy_mean", predicted_fact
        )

    verdict = decide_verdict(
        nexus,
        rag,
        llm_only,
        oracle_fact=oracle_fact,
        predicted_fact=predicted_fact,
    )

    artifact = {
        "schema_version": "nexus-architecture-validation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "dataset": str(dataset.as_posix()),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "questions_total": len(records),
        "er3_dir": er3_dir,
        "arms": {
            "nexus": {
                "arm": nexus["arm"],
                "identity": nexus["identity"],
                "graph": nexus["graph"],
                "metrics": nexus["metrics"],
            },
            "rag": {"arm": rag["arm"], "metrics": rag["metrics"]},
            "llm_only": {"arm": llm_only["arm"], "metrics": llm_only["metrics"]},
        },
        "paired_comparisons": {
            "nexus_vs_rag": nexus_vs_rag,
            "nexus_vs_llm_only": nexus_vs_llm,
        },
        "verdict": verdict,
        "status": "VALID",
    }

    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--er3-dir", default=DEFAULT_ER3_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--paired-evidence",
        type=Path,
        default=_project_root
        / "benchmarks"
        / "results"
        / "oracle_vs_predicted_union_l1_acceptance_full_20260721T241500Z.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Timestamped artifact path under benchmarks/results/",
    )
    args = parser.parse_args(argv)
    artifact = run_campaign(
        dataset=args.dataset,
        output=args.output,
        er3_dir=args.er3_dir,
        limit=args.limit,
        paired_evidence=args.paired_evidence,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "output": str(args.output),
                "decision": artifact["verdict"]["decision"],
                "nexus_fact": artifact["arms"]["nexus"]["metrics"]["fact_accuracy_mean"],
                "rag_fact": artifact["arms"]["rag"]["metrics"]["fact_accuracy_mean"],
                "llm_only_fact": artifact["arms"]["llm_only"]["metrics"][
                    "fact_accuracy_mean"
                ],
                "entry": artifact["arms"]["nexus"]["metrics"]["entry_recall_mean"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
