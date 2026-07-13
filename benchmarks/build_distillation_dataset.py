"""Build a deterministic, leakage-safe NEXUS Realizer v1 dataset.

Only train-split questions are accepted.  Gold entities drive the NEXUS
oracle path so the Realizer learns evidence-to-answer behavior instead of
entity-resolution mistakes.  The command refuses to overwrite artifacts and
emits a hash-verified manifest; it never writes model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.realizer_contracts import (
    SCHEMA_VERSION,
    canonical_json,
    jsonl_bytes,
    normalize_question,
    sha256_file,
    sha256_json,
    split_by_entity_family,
    stable_example_id,
    validate_dataset_manifest,
    validate_distillation_record,
)
from nexus.graph.store import InMemoryGraphStore
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner


def _verification_dict(result: Any) -> dict[str, Any]:
    return {
        "passed": bool(result.passed),
        "supported_count": int(result.supported_count),
        "unsupported_claims": list(result.unsupported_claims),
        "hallucination_rate": float(result.hallucination_rate),
    }


def _build_record(
    question: dict[str, Any],
    runner: NEXUSRunner,
    *,
    source_sha: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    entities = question.get("entities", question.get("gold_entities", []))
    entities = [str(value) for value in entities] if isinstance(entities, list) else []
    oracle_input = {
        "id": str(question.get("id", "")),
        "question": str(question.get("question", "")),
        "gold_entities": entities,
    }
    result = runner.run_oracle([oracle_input], source_sha=source_sha)
    if result.errors or not result.per_question:
        return None, ["oracle_pipeline_error"]
    qr = result.per_question[0]
    target_answer = str(question.get("answer", question.get("gold_answer", "")))
    target_verification = runner.verifier.verify(target_answer, qr.evidence_pack)
    normalized_hash = hashlib.sha256(
        normalize_question(oracle_input["question"]).encode("utf-8")
    ).hexdigest()
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": stable_example_id(oracle_input["question"], entities),
        "source_question_id": oracle_input["id"],
        "source_split": str(question.get("source_split", "")),
        "question": oracle_input["question"],
        "normalized_question_sha256": normalized_hash,
        "question_type": str(question.get("question_type", qr.parsed_intent)),
        "intent": str(question.get("intent", qr.parsed_intent)),
        "canonical_entities": sorted(set(entities)),
        "evidence_pack": qr.evidence_pack,
        "reasoning_audit": qr.reasoning_audit,
        "answer": target_answer,
        "target_verification": _verification_dict(target_verification),
        "pipeline_answer": qr.answer,
        "pipeline_verifier_passed": qr.verifier_passed,
        "source_sha": source_sha,
        "config_hash": result.config_hash,
        "generator_identity": "nexus_v1_oracle_evidence",
    }
    return record, validate_distillation_record(record)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def build_distillation_dataset(
    questions: list[dict[str, Any]],
    graph: InMemoryGraphStore,
    output_dir: str,
    source_sha: str,
    min_pairs: int = 5000,
    validation_fraction: float = 0.2,
    seed: int = 20260711,
    source_path: str = "",
) -> dict[str, Any]:
    """Build train/validation JSONL files and a self-validating manifest."""
    if not source_sha.strip():
        raise ValueError("source_sha is required")
    if min_pairs < 1:
        raise ValueError("min_pairs must be >= 1")
    if not questions:
        raise ValueError("questions must be non-empty")

    config = ProductionNEXUSConfig.lexical_only()
    runner = NEXUSRunner(graph, config)
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    rejected_records = 0
    seen_questions: set[str] = set()
    seen_ids: set[str] = set()

    for index, question in enumerate(questions):
        normalized = normalize_question(str(question.get("question", "")))
        if normalized in seen_questions:
            rejected["duplicate_normalized_question"] += 1
            rejected_records += 1
            continue
        record, errors = _build_record(question, runner, source_sha=source_sha)
        if errors or record is None:
            rejected_records += 1
            for error in errors or ["unknown_record_error"]:
                rejected[error] += 1
            continue
        if record["id"] in seen_ids:
            rejected["duplicate_stable_id"] += 1
            rejected_records += 1
            continue
        seen_questions.add(normalized)
        seen_ids.add(record["id"])
        accepted.append(record)
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(questions)}: {len(accepted)} accepted")

    if len(accepted) < 2:
        raise ValueError("fewer than two safe records; cannot create grouped splits")
    splits = split_by_entity_family(
        accepted, validation_fraction=validation_fraction, seed=seed
    )
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    split_meta: dict[str, dict[str, Any]] = {}
    for split_name, records in splits.items():
        path = out / f"{split_name}.jsonl"
        _write_exclusive(path, jsonl_bytes(records))
        split_meta[split_name] = {
            "path": path.name,
            "count": len(records),
            "sha256": sha256_file(path),
            "entity_families": len({record["entity_family"] for record in records}),
        }

    builder_config = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "min_pairs": min_pairs,
        "source_split": "train",
        "entity_resolution": "oracle",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "config_hash": config.config_hash,
        "builder_config": builder_config,
        "builder_config_sha256": sha256_json(builder_config),
        "source_path": source_path,
        "source_sha256": sha256_file(Path(source_path)) if source_path else "",
        "questions_processed": len(questions),
        "pairs_accepted": len(accepted),
        "pairs_rejected": rejected_records,
        "rejection_reasons": dict(sorted(rejected.items())),
        "min_pairs_target": min_pairs,
        "target_met": len(accepted) >= min_pairs,
        "splits": split_meta,
        "validation_fraction_actual": round(
            len(splits["validation"]) / len(accepted), 6
        ),
        "dataset_sha256": sha256_json(
            {name: meta["sha256"] for name, meta in sorted(split_meta.items())}
        ),
    }
    manifest_errors = validate_dataset_manifest(manifest, out)
    if manifest_errors:
        raise RuntimeError("invalid generated manifest: " + "; ".join(manifest_errors))
    _write_exclusive(
        out / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return manifest


def _load_train_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if path.name.casefold() in {"test.jsonl", "val.jsonl", "validation.jsonl", "holdout.jsonl"}:
        raise ValueError("distillation input must be the train split, never val/test/holdout")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["source_split"] = "train"
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="stack/encoder/data/train.jsonl")
    parser.add_argument("--output-dir", default="data/distillation/realizer_v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-pairs", type=int, default=5000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    input_path = Path(args.input)
    questions = _load_train_questions(input_path, args.limit)
    from benchmarks.run_benchmark import build_benchmark_graph
    graph, _ = build_benchmark_graph()
    manifest = build_distillation_dataset(
        questions,
        graph,
        args.output_dir,
        source_sha,
        min_pairs=args.min_pairs,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        source_path=str(input_path),
    )
    print(canonical_json({
        "status": "READY" if manifest["target_met"] else "BLOCKED",
        "pairs_accepted": manifest["pairs_accepted"],
        "target": manifest["min_pairs_target"],
        "manifest": str(Path(args.output_dir) / "manifest.json"),
    }))
    return 0 if manifest["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
